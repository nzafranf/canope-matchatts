
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model import PAD_CLASS


def build_targets(canon_ids: torch.Tensor, canon_len: torch.Tensor, M: int):
    B, Lc = canon_ids.shape
    device = canon_ids.device
    pos = torch.arange(M, device=device)
    idx = pos.clamp(max=max(Lc - 1, 0)).view(1, M).expand(B, M)
    bytes_at = canon_ids.gather(1, idx)
    is_content = pos.view(1, M) < canon_len.view(B, 1)
    return torch.where(is_content, bytes_at, torch.full_like(bytes_at, PAD_CLASS))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(-2) / m.sum(-2).clamp(min=eps)


def sentence_embedding(zc: torch.Tensor, content_len: torch.Tensor):
    M = zc.shape[2]
    pos = torch.arange(M, device=zc.device).view(1, 1, M)
    mask = pos < content_len.unsqueeze(-1)
    raw = masked_mean(zc.float(), mask)
    return raw, F.normalize(raw, dim=-1)


def cross_entropy_loss(logits, targets, sup_len):
    B, V, M, C = logits.shape
    tgt = targets.unsqueeze(1).expand(B, V, M)
    ce = F.cross_entropy(
        logits.reshape(-1, C).float(), tgt.reshape(-1), reduction="none"
    ).view(B, V, M)

    pos = torch.arange(M, device=logits.device).view(1, 1, M)
    sup = pos < sup_len.unsqueeze(-1)
    denom = sup.sum(-1).clamp(min=1).to(ce.dtype)
    per_view = (ce * sup).sum(-1) / denom

    with torch.no_grad():
        pred = logits.argmax(-1)
        correct = ((pred == tgt) & sup).sum().float()
        total = sup.sum().float().clamp(min=1)
        acc = correct / total
        ent = -(F.softmax(logits.float(), -1)
                * F.log_softmax(logits.float(), -1)).sum(-1)
        ent = (ent * sup).sum() / total

    return per_view.mean(), {"ce/acc": acc, "ce/entropy": ent}


def empty_penalty(logits, sup_len, canvas_len):
    B, V, M, _ = logits.shape
    pos = torch.arange(M, device=logits.device).view(1, 1, M)
    tail = (pos >= sup_len.unsqueeze(-1)) & (pos < canvas_len.unsqueeze(-1))
    n = tail.sum()
    if int(n) == 0:
        z = logits.sum() * 0.0
        return z, {"empty/frac": torch.zeros((), device=logits.device)}
    probs = F.softmax(logits.float(), dim=-1)
    norm_sq = (probs ** 2).sum(-1)
    val = (norm_sq * tail).sum() / n.clamp(min=1)
    return val, {"empty/frac": tail.float().mean()}


def length_loss(rho, canon_len, src_len, tau: float = 0.5):
    rho_star = torch.log(canon_len.float().clamp(min=1).unsqueeze(1)) \
             - torch.log(src_len.clamp(min=1.0))
    e = rho_star - rho
    val = torch.maximum(tau * e, (tau - 1.0) * e).mean()
    with torch.no_grad():
        aux = {
            "len/rho_hat": rho.mean(),
            "len/rho_star": rho_star.mean(),
            "len/abs_err": e.abs().mean(),
        }
    return val, aux


def latent_loss(zc, canvas_mask, noisy_views):
    tgt = zc[:, 0:1].detach().float()
    z = zc[:, noisy_views].float()
    m = canvas_mask[:, noisy_views].unsqueeze(-1).to(z.dtype)
    val = ((z - tgt) ** 2 * m).sum() / (m.sum().clamp(min=1.0) * zc.shape[-1])
    with torch.no_grad():
        cm = canvas_mask[:, noisy_views].to(z.dtype)
        cos = F.cosine_similarity(z, tgt.expand_as(z), dim=-1)
        aux = {
            "latent/mse": val.detach(),
            "latent/cos": (cos * cm).sum() / cm.sum().clamp(min=1.0),
            "latent/target_norm": tgt.norm(dim=-1).mean(),
        }
    return val, aux


def invariance_loss(q, noisy_views):
    anchor = q[:, 0:1, :]
    noisy = q[:, noisy_views, :]
    cos = (noisy * anchor).sum(-1)
    val = (1.0 - cos).mean()
    return val, {"inv/cos": cos.mean().detach()}


def negation_loss(q, noisy_views, n_offset, n_core, n_negation,
                  n_unrelated: int = 8, m1: float = 0.2, m2: float = 0.2,
                  generator=None):
    device = q.device
    R = int(n_negation)
    zero = q.sum() * 0.0
    empty = {
        "neg/s_pos": torch.zeros((), device=device),
        "neg/s_neg": torch.zeros((), device=device),
        "neg/s_unr": torch.zeros((), device=device),
        "neg/active_frac": torch.zeros((), device=device),
    }
    if R <= 0 or q.shape[0] < n_core + R or n_offset >= n_core or n_offset < 1:
        return zero, empty

    A = noisy_views
    P = q[n_offset:n_core]
    T = q[n_core:n_core + R]
    if P.shape[0] != R:
        return zero, empty

    n_reg = max(n_offset, 1)
    k = min(n_unrelated, n_reg)
    sel = torch.randperm(n_reg, device=device, generator=generator)[:k]
    U = q[sel, 0, :].detach()

    def side(X, Y):
        xa = X[:, A, :]
        s_pos = (xa * X[:, 0:1, :]).sum(-1)
        s_neg = (xa * Y[:, 0:1, :]).sum(-1)
        s_unr = torch.einsum("rad,ud->rau", xa, U).mean(-1)
        h1 = F.relu(s_neg - s_pos + m1)
        h2 = F.relu(s_unr - s_neg + m2)
        return h1 + h2, s_pos, s_neg, s_unr

    l_p, sp_p, sn_p, su_p = side(P, T)
    l_t, sp_t, sn_t, su_t = side(T, P)
    val = 0.5 * (l_p.mean() + l_t.mean())

    with torch.no_grad():
        aux = {
            "neg/s_pos": 0.5 * (sp_p.mean() + sp_t.mean()),
            "neg/s_neg": 0.5 * (sn_p.mean() + sn_t.mean()),
            "neg/s_unr": 0.5 * (su_p.mean() + su_t.mean()),
            "neg/active_frac": 0.5 * ((l_p > 0).float().mean()
                                      + (l_t > 0).float().mean()),
        }
    return val, aux


class VISReg(nn.Module):

    def __init__(self, num_projections: int = 256, scale_weight: float = 1.0,
                 shape_weight: float = 1.0, center_weight: float = 1.0,
                 projection_chunk: int = 256, shape_std_floor: float = 0.1):
        super().__init__()
        if num_projections <= 0 or projection_chunk <= 0 or shape_std_floor <= 0:
            raise ValueError("num_projections, projection_chunk, shape_std_floor "
                             "harus positif")
        self.K = int(num_projections)
        self.scale_weight = float(scale_weight)
        self.shape_weight = float(shape_weight)
        self.center_weight = float(center_weight)
        self.projection_chunk = int(projection_chunk)
        self.shape_std_floor = float(shape_std_floor)
        self._cached_B = -1
        self._cached_target = None

    def _target(self, batch_size: int, device):
        if self._cached_B != batch_size or self._cached_target is None:
            q = torch.linspace(1, batch_size, batch_size, device=device,
                               dtype=torch.float32) / (batch_size + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2.0))
            self._cached_B = batch_size
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor):
        if z.dim() != 3:
            raise ValueError(f"VISReg mengharapkan [V,B,D], dapat {tuple(z.shape)}")
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            _, B, D = z.shape

            mu = z.mean(dim=1, keepdim=True)
            center = mu.pow(2).mean()

            centered = z - mu
            std = centered.norm(dim=1).div(math.sqrt(B))
            scale = (std - 1.0).pow(2).mean()

            shape_std = std.detach().clamp_min(self.shape_std_floor)
            normalized = centered / shape_std.unsqueeze(1)
            target = self._target(B, z.device).view(1, B, 1)

            shape_sum = torch.zeros((), device=z.device, dtype=z.dtype)
            n_proj = 0
            for k0 in range(0, self.K, self.projection_chunk):
                w = min(self.projection_chunk, self.K - k0)
                dirs = F.normalize(torch.randn(D, w, device=z.device,
                                               dtype=z.dtype), dim=0)
                projected = (normalized @ dirs).sort(dim=1).values
                shape_sum = shape_sum + (projected - target).pow(2).mean() * w
                n_proj += w
            shape = shape_sum / max(1, n_proj)

            total = (self.scale_weight * scale + self.shape_weight * shape
                     + self.center_weight * center)
            return total, {"visreg/scale": scale.detach(),
                           "visreg/shape": shape.detach(),
                           "visreg/center": center.detach()}


@dataclass
class LossWeights:
    eta: float = 0.1
    gamma: float = 1.0
    alpha: float = 0.5
    beta: float = 0.3
    zeta: float = 0.0
    tau: float = 0.5
    m1: float = 0.2
    m2: float = 0.2
    n_unrelated: int = 8
    lam: float = 1.0


def compute_losses(out, batch_meta, weights: LossWeights, generator=None,
                   regularizer=None, objective: str = "ce"):
    logits = out["logits"]
    B, V, M, _ = logits.shape

    canon_ids = batch_meta["canon_ids"]
    canon_len = batch_meta["canon_len"]
    canvas_len = batch_meta["canvas_len"]
    sup_len = batch_meta["sup_len"]
    A = batch_meta["noisy_views"]

    targets = build_targets(canon_ids, canon_len, M)
    content_len = torch.minimum(canon_len.unsqueeze(1), canvas_len)
    q_raw, q = sentence_embedding(out["zc"], content_len)

    parts, aux = {}, {}

    l_ce, a = cross_entropy_loss(logits, targets, sup_len)
    parts["ce"] = l_ce; aux.update(a)

    l_p, a = empty_penalty(logits, sup_len, canvas_len)
    parts["empty"] = l_p; aux.update(a)

    l_len, a = length_loss(out["rho"], canon_len, out["src_len"], tau=weights.tau)
    parts["len"] = l_len; aux.update(a)

    l_inv, a = invariance_loss(q, A)
    parts["inv"] = l_inv; aux.update(a)

    l_n, a = negation_loss(
        q, A, batch_meta["n_offset"], batch_meta["n_core"],
        batch_meta["n_negation"], n_unrelated=weights.n_unrelated,
        m1=weights.m1, m2=weights.m2, generator=generator,
    )
    parts["neg"] = l_n; aux.update(a)

    if weights.zeta > 0 and regularizer is not None:
        l_vis, a = regularizer(q_raw.transpose(0, 1))
        parts["vis"] = l_vis; aux.update(a)
    else:
        parts["vis"] = q_raw.sum() * 0.0

    total = (parts["ce"]
             + weights.eta * parts["empty"]
             + weights.gamma * parts["len"]
             + weights.alpha * parts["inv"]
             + weights.beta * parts["neg"]
             + weights.zeta * parts["vis"])

    if objective == "jepa":
        l_lat, a = latent_loss(out["zc"], out["canvas_mask"], A)
        parts["latent"] = l_lat; aux.update(a)
        total = total + weights.lam * l_lat

    parts["total"] = total
    return total, parts, aux, {"targets": targets, "q": q}
