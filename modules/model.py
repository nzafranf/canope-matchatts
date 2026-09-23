import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modules.utils import length_to_mask

PAD_CLASS = 256
N_VOCAB_OUT = 257


def sinusoidal_PE(length: int, d_model: int, device=None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


def gaussian_resample(h: torch.Tensor,
                      source_mask: torch.Tensor,
                      target_lengths: torch.Tensor,
                      tau_r: float = 0.3,
                      target_max: Optional[int] = None,
                      chunk_size: int = 256):
    if h.dim() != 3:
        raise ValueError(f"h harus [B, L_src, D], dapat {tuple(h.shape)}")
    B, L_src, D = h.shape
    device = h.device

    if source_mask.shape != h.shape[:2]:
        raise ValueError("source_mask harus sama dengan dua dimensi awal h")
    if tau_r <= 0:
        raise ValueError(f"tau_r harus > 0, dapat {tau_r}")

    target_lengths = target_lengths.to(device=device, dtype=torch.long).clamp(min=1)
    src_count = source_mask.sum(dim=1)
    src_len_long = src_count.clamp(min=1)
    src_len = src_len_long.to(torch.float32)
    tgt_len = target_lengths.to(torch.float32)
    M = int(target_max if target_max is not None else int(target_lengths.max().item()))
    M = max(M, 1)

    i = torch.arange(L_src, device=device, dtype=torch.float32).view(1, 1, L_src)
    chunks = []
    for t0 in range(0, M, chunk_size):
        t1 = min(M, t0 + chunk_size)
        t = torch.arange(t0, t1, device=device, dtype=torch.float32).view(1, -1)
        denom = (tgt_len - 1.0).clamp(min=1.0).view(B, 1)
        c_t = t * (src_len - 1.0).clamp(min=0.0).view(B, 1) / denom
        c_t = torch.where(tgt_len.view(B, 1) > 1.0, c_t, torch.zeros_like(c_t))
        psi = -((i - c_t.unsqueeze(-1)) ** 2) / float(tau_r)
        psi = psi.masked_fill(~source_mask.view(B, 1, L_src), float("-inf"))
        omega = torch.softmax(psi, dim=-1)
        omega = torch.nan_to_num(omega, nan=0.0)
        chunks.append(torch.bmm(omega.to(h.dtype), h))

    c = torch.cat(chunks, dim=1)
    canvas_mask = length_to_mask(target_lengths, M)

    n_copy = min(L_src, M)
    ident = c.new_zeros(B, M, D)
    ident[:, :n_copy] = h[:, :n_copy].to(c.dtype)
    same = ((src_count > 0) & (src_len_long == target_lengths)).view(B, 1, 1)
    c = torch.where(same, ident, c)

    c = c * canvas_mask.unsqueeze(-1).to(c.dtype)
    return c, canvas_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, h, mask=None):
        kpm = ~mask if mask is not None else None
        qkv = self.norm1(h)
        attn_out, _ = self.self_attn(qkv, qkv, qkv, key_padding_mask=kpm,
                                     need_weights=False)
        h = h + self.dropout1(attn_out)
        h = h + self.dropout2(self.ffn(self.norm2(h)))
        return h


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False

    def forward(self, h, mask):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and h.requires_grad:
                h = checkpoint(layer, h, mask, use_reentrant=False)
            else:
                h = layer(h, mask=mask)
        h = self.final_norm(h)
        return h * mask.unsqueeze(-1).to(h.dtype)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, z, q_canon, query_mask, source_mask):
        self_kpm = ~query_mask if query_mask is not None else None
        q = self.norm1(q_canon)
        attn_out, _ = self.self_attn(q, q, q, key_padding_mask=self_kpm,
                                     need_weights=False)
        q_canon = q_canon + self.dropout1(attn_out)

        cross_kpm = ~source_mask if source_mask is not None else None
        cross_out, _ = self.cross_attn(self.norm2(q_canon), z, z,
                                       key_padding_mask=cross_kpm,
                                       need_weights=False)
        q_canon = q_canon + self.dropout2(cross_out)
        q_canon = q_canon + self.dropout3(self.ffn(self.norm3(q_canon)))
        return q_canon


class TransformerDecoder(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, n_layers, max_length=2048,
                 dropout=0.0, tau_r: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.tau_r = tau_r
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.input_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False

    def forward(self, z, l_star, source_mask, strict_max: bool = True):
        device = z.device
        l_star = l_star.to(device=device, dtype=torch.long)
        if strict_max and bool((l_star > self.max_length).any()):
            raise ValueError(
                f"panjang kanvas {int(l_star.max())} melampaui max_length="
                f"{self.max_length}. Potong teks lebih dulu -- memotong kanvas "
                f"secara diam-diam akan menghilangkan konten tanpa jejak."
            )
        l_star = l_star.clamp(min=1, max=self.max_length)
        l_star_max = int(l_star.max().item())

        c, query_mask = gaussian_resample(z, source_mask, l_star,
                                          tau_r=self.tau_r, target_max=l_star_max)
        pe = sinusoidal_PE(l_star_max, self.d_model, device=device, dtype=c.dtype)
        q = self.input_norm(c + pe.unsqueeze(0))
        q = q * query_mask.unsqueeze(-1).to(q.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and q.requires_grad:
                q = checkpoint(layer, z, q, query_mask, source_mask,
                               use_reentrant=False)
            else:
                q = layer(z, q, query_mask=query_mask, source_mask=source_mask)

        q = self.final_norm(q)
        q = q * query_mask.unsqueeze(-1).to(q.dtype)
        return q, query_mask


class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def forward(self, z, mask):
        logits = self.score(z).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (z * w).sum(dim=1)


class LengthPredictor(nn.Module):

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1,
                 rho_min: float = -2.5, rho_max: float = 2.5,
                 rho_init: float = 0.0):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.pool = MaskedAttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, float(rho_init))

    def set_rho_init(self, rho_init: float):
        with torch.no_grad():
            self.head[-1].bias.fill_(float(rho_init))

    def forward(self, z, mask, detach_input: bool = True):
        if detach_input:
            z = z.detach()
        pooled = self.pool(z, mask)
        lengths = mask.sum(dim=1).float().clamp(min=1.0)
        feat = torch.cat([pooled, torch.log(lengths).unsqueeze(-1)], dim=-1)
        rho = self.head(feat).squeeze(-1)
        rho_route = rho.detach().clamp(self.rho_min, self.rho_max)
        return rho, lengths * torch.exp(rho_route)


class CanonCanvas(nn.Module):
    ARCH_VERSION = "canon-ce-v1"

    def __init__(self, n_vocab_in: int = 256, d_model: int = 512,
                 n_attn_heads: int = 8, enc_layers: int = 4, dec_layers: int = 4,
                 max_length: int = 2048, dropout: float = 0.0,
                 tau_r: float = 0.3, rho_init: float = 0.0,
                 rho_route_min: float = -2.5, rho_route_max: float = 2.5):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.embedding = nn.Embedding(n_vocab_in, d_model, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        self.embed_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(d_model, n_attn_heads, 4 * d_model,
                                          enc_layers, dropout)
        self.decoder = TransformerDecoder(d_model, n_attn_heads, 4 * d_model,
                                          dec_layers, max_length=max_length,
                                          dropout=dropout, tau_r=tau_r)
        self.length_predictor = LengthPredictor(
            d_model, rho_init=rho_init,
            rho_min=rho_route_min, rho_max=rho_route_max,
        )
        self.head = nn.Linear(d_model, N_VOCAB_OUT)

    def architecture_signature(self) -> dict:
        return {
            "version": self.ARCH_VERSION,
            "d_model": self.d_model,
            "attention_heads": self.encoder.layers[0].self_attn.num_heads,
            "encoder_layers": len(self.encoder.layers),
            "decoder_layers": len(self.decoder.layers),
            "max_length": self.max_length,
            "n_vocab_out": N_VOCAB_OUT,
        }

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.encoder.gradient_checkpointing = enabled
        self.decoder.gradient_checkpointing = enabled
        return self

    def _embed(self, ids):
        emb = self.embedding(ids)
        pe = sinusoidal_PE(ids.shape[1], self.d_model, device=ids.device,
                           dtype=emb.dtype)
        return self.embed_norm(emb + pe.unsqueeze(0))

    def encode(self, ids, mask=None):
        ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        if mask is None:
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        return self.encoder(self._embed(ids), mask=mask), mask

    def forward_views(self, x, mask, canvas_len, detach_head: bool = False):
        B, V, L = x.shape
        flat_x = x.reshape(B * V, L)
        flat_m = mask.reshape(B * V, L)

        h = self.encoder(self._embed(flat_x), mask=flat_m)
        rho, _ = self.length_predictor(h, flat_m, detach_input=True)

        l_star = canvas_len.reshape(B * V)
        zc, cmask = self.decoder(h, l_star, source_mask=flat_m)
        M = zc.shape[1]
        head_in = zc.detach() if detach_head else zc

        return {
            "zc": zc.view(B, V, M, self.d_model),
            "logits": self.head(head_in).view(B, V, M, N_VOCAB_OUT),
            "canvas_mask": cmask.view(B, V, M),
            "rho": rho.view(B, V),
            "src_len": flat_m.sum(dim=1).float().view(B, V),
            "H": h,
            "src_mask": flat_m,
        }

    @torch.no_grad()
    def generate(self, ids, mask=None, delta: float = 0.15, c: int = 8,
                 max_tries: int = 3, growth: float = 1.5):
        h, mask = self.encode(ids, mask)
        _, l_hat = self.length_predictor(h, mask, detach_input=True)
        l_star = (torch.ceil(l_hat * (1.0 + delta)).long() + c).clamp(
            min=1, max=self.max_length)

        for attempt in range(1, max_tries + 1):
            zc, cmask = self.decoder(h, l_star, source_mask=mask, strict_max=False)
            pred = self.head(zc).argmax(-1)
            is_pad = (pred == PAD_CLASS) & cmask
            has_pad = is_pad.any(dim=1)
            if bool(has_pad.all()):
                break
            if bool(((~has_pad) & (l_star >= self.max_length)).any()):
                raise RuntimeError(
                    "kanvas jenuh dan melampaui max_length; teks perlu dipotong"
                )
            l_star = torch.where(
                has_pad, l_star,
                torch.ceil(l_star.to(torch.float32) * growth).long(),
            ).clamp(min=1, max=self.max_length)
        else:
            raise RuntimeError(
                f"kanvas jenuh setelah {max_tries} percobaan "
                f"(M={int(l_star.max())}); keluaran akan terpotong, jadi "
                f"dilaporkan sebagai galat."
            )

        outs = []
        row_len = cmask.sum(dim=1)
        for b in range(pred.shape[0]):
            pos = is_pad[b].nonzero()
            end = int(pos[0]) if len(pos) else int(row_len[b])
            outs.append(pred[b, :end].tolist())
        return outs, int(l_star.max()), attempt
