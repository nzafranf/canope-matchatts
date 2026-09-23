
import torch

TF, OE, UE = 0, 1, 2
REGIME_NAMES = {TF: "tf", OE: "oe", UE: "ue"}

_ANCHORS = [
    (0.00, 1.00, 0.00, 0.00),
    (0.15, 1.00, 0.00, 0.00),
    (0.40, 0.50, 0.50, 0.00),
    (0.65, 0.35, 0.45, 0.20),
    (1.00, 0.20, 0.60, 0.20),
]


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def regime_probs(progress: float, disable_ue: bool = False):
    p = min(1.0, max(0.0, float(progress)))
    out = tuple(_ANCHORS[-1][1:])
    for k in range(len(_ANCHORS) - 1):
        p0, *v0 = _ANCHORS[k]
        p1, *v1 = _ANCHORS[k + 1]
        if p <= p1:
            if p1 <= p0:
                out = tuple(v1)
            else:
                w = _smoothstep((p - p0) / (p1 - p0))
                out = tuple(a + (b - a) * w for a, b in zip(v0, v1))
            break
    if disable_ue:
        p_tf, p_oe, p_ue = out
        out = (p_tf, p_oe + p_ue, 0.0)
    return out


class CanvasRegimeSampler:

    def __init__(self, n_pad: int = 4, max_length: int = 2048,
                 tf_extra: int = 8,
                 oe_lo: float = 1.15, oe_hi: float = 1.80,
                 ue_lo: float = 0.50, ue_hi: float = 0.95,
                 disable_ue: bool = False):
        self.n_pad = int(n_pad)
        self.max_length = int(max_length)
        self.tf_extra = int(tf_extra)
        self.oe_lo, self.oe_hi = float(oe_lo), float(oe_hi)
        self.ue_lo, self.ue_hi = float(ue_lo), float(ue_hi)
        self.disable_ue = bool(disable_ue)
        self._progress = 0.0

    def set_progress(self, progress: float):
        self._progress = min(1.0, max(0.0, float(progress)))

    @property
    def probs(self):
        return regime_probs(self._progress, disable_ue=self.disable_ue)

    def sample(self, canon_len: torch.Tensor, generator=None,
               share_views: bool = False):
        device = canon_len.device
        shape = canon_len.shape
        L = canon_len.to(torch.float32)

        draw = shape
        if share_views and len(shape) == 2:
            draw = torch.Size((shape[0], 1))

        p_tf, p_oe, p_ue = self.probs
        probs = torch.tensor([p_tf, p_oe, p_ue], device=device, dtype=torch.float32)
        probs = probs / probs.sum().clamp(min=1e-8)
        regime = torch.multinomial(
            probs.expand(draw.numel(), 3), num_samples=1, replacement=True,
            generator=generator,
        ).view(draw).expand(shape)

        u = torch.rand(draw, device=device, generator=generator).expand(shape)
        m_tf = canon_len + torch.randint(
            0, self.tf_extra + 1, draw, device=device, generator=generator
        ).expand(shape)
        m_oe = torch.ceil(L * (self.oe_lo + (self.oe_hi - self.oe_lo) * u)).long()
        m_ue = torch.ceil(L * (self.ue_lo + (self.ue_hi - self.ue_lo) * u)).long()

        canvas = torch.where(regime == TF, m_tf,
                             torch.where(regime == OE, m_oe, m_ue))
        canvas = canvas.clamp(min=1, max=self.max_length)

        sup = torch.minimum(canvas, canon_len + self.n_pad)
        return canvas, sup, regime
