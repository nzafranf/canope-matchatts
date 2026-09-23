"""Frozen CanonCanvas text encoder adapted to MatchaTTS phoneme IDs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from modules.model import CanonCanvas, sinusoidal_PE


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for ``[B, C, T]`` tensors."""

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = ((x - mean) ** 2).mean(1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class DurationPredictor(nn.Module):
    """Matcha-compatible duration predictor without importing Matcha at module load."""

    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=padding)
        self.norm_1 = ChannelLayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=padding)
        self.norm_2 = ChannelLayerNorm(filter_channels)
        self.drop = nn.Dropout(p_dropout)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * mask))))
        return self.proj(x * mask) * mask


def _load_payload(path: str | Path) -> Mapping:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canon checkpoint not found: {path}")
    # These checkpoints are project-owned PyTorch files and include dict metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


def inspect_canon_checkpoint(path: str | Path) -> dict:
    payload = _load_payload(path)
    required = {"model", "signature", "args"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Canon checkpoint misses keys: {sorted(missing)}")
    state = payload["model"]
    signature = payload["signature"]
    embedding = state.get("embedding.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("Canon checkpoint has no valid embedding.weight")
    if int(signature["d_model"]) != int(embedding.shape[1]):
        raise ValueError("Checkpoint signature d_model disagrees with embedding")
    return {
        "path": str(Path(path).resolve()),
        "step": int(payload.get("step", -1)),
        "signature": dict(signature),
        "objective": payload["args"].get("objective"),
        "zeta": payload["args"].get("zeta"),
        "n_vocab_in": int(embedding.shape[0]),
    }


def _build_backbone(payload: Mapping) -> CanonCanvas:
    signature = payload["signature"]
    args = payload["args"]
    state = payload["model"]
    backbone = CanonCanvas(
        n_vocab_in=int(state["embedding.weight"].shape[0]),
        d_model=int(signature["d_model"]),
        n_attn_heads=int(signature["attention_heads"]),
        enc_layers=int(signature["encoder_layers"]),
        dec_layers=int(signature["decoder_layers"]),
        max_length=int(signature["max_length"]),
        dropout=float(args.get("dropout", 0.0)),
        tau_r=float(args.get("tau_r", 0.3)),
    )
    backbone.load_state_dict(state, strict=True)
    return backbone


def _symbol_bytes(symbol: str, n_vocab_in: int) -> list[int]:
    values = [value for value in symbol.encode("utf-8") if value < n_vocab_in]
    return values or [1]


class FrozenCanonTextEncoder(nn.Module):
    """Matcha text-encoder interface backed by a frozen Canon transformer.

    The phoneme adapter is the only vocabulary-changing layer. It is initialized
    from the mean Canon byte embedding of each Matcha symbol, then trained. The
    Canon embed norm and transformer remain frozen, while the mel projection and
    duration predictor remain trainable.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        n_vocab: int,
        n_feats: int,
        symbols: Sequence[str],
        duration_filter_channels: int = 256,
        duration_kernel_size: int = 3,
        duration_dropout: float = 0.1,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
    ):
        super().__init__()
        if n_spks != 1:
            raise NotImplementedError("The LJSpeech experiment supports n_spks=1 only")
        if len(symbols) != n_vocab:
            raise ValueError(f"Expected {n_vocab} Matcha symbols, got {len(symbols)}")

        payload = _load_payload(checkpoint_path)
        info = inspect_canon_checkpoint(checkpoint_path)
        backbone = _build_backbone(payload)
        self.checkpoint_info = info
        self.d_model = int(info["signature"]["d_model"])
        self.max_length = int(info["signature"]["max_length"])

        self.phoneme_adapter = nn.Embedding(n_vocab, self.d_model, padding_idx=0)
        self._initialize_adapter(backbone.embedding.weight.detach(), symbols)

        self.embed_norm = backbone.embed_norm
        self.encoder = backbone.encoder
        self.proj_m = nn.Conv1d(self.d_model, n_feats, 1)
        self.proj_w = DurationPredictor(
            self.d_model,
            duration_filter_channels,
            duration_kernel_size,
            duration_dropout,
        )
        self._freeze_backbone()

    def _initialize_adapter(self, byte_embedding: torch.Tensor, symbols: Iterable[str]) -> None:
        with torch.no_grad():
            for index, symbol in enumerate(symbols):
                if index == 0:
                    self.phoneme_adapter.weight[index].zero_()
                    continue
                ids = torch.tensor(
                    _symbol_bytes(str(symbol), byte_embedding.shape[0]), dtype=torch.long
                )
                self.phoneme_adapter.weight[index].copy_(byte_embedding.index_select(0, ids).mean(0))

    def _freeze_backbone(self) -> None:
        for module in (self.embed_norm, self.encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the checkpointed encoder deterministic even while its adapters train.
        self.embed_norm.eval()
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, spks=None):
        del spks
        if x.shape[1] > self.max_length:
            raise ValueError(
                f"Phoneme sequence length {x.shape[1]} exceeds Canon max_length={self.max_length}"
            )
        mask_bool = _sequence_mask(x_lengths, x.shape[1])
        mask = mask_bool.unsqueeze(1).to(dtype=self.phoneme_adapter.weight.dtype)

        adapted = self.phoneme_adapter(x)
        pe = sinusoidal_PE(
            x.shape[1], self.d_model, device=x.device, dtype=adapted.dtype
        )
        hidden = self.embed_norm(adapted + pe.unsqueeze(0))
        hidden = self.encoder(hidden, mask=mask_bool)
        hidden = hidden * mask_bool.unsqueeze(-1).to(hidden.dtype)
        hidden_ch = hidden.transpose(1, 2)

        mu = self.proj_m(hidden_ch) * mask
        logw = self.proj_w(hidden_ch.detach(), mask)
        return mu, logw, mask

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

