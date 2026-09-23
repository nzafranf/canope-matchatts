"""Hydra-instantiable MatchaTTS model for the three requested runs."""

from __future__ import annotations

from pathlib import Path

import torch

from matcha.models.matcha_tts import MatchaTTS
from matcha.text.symbols import symbols

from matcha_exp.frozen_canon import FrozenCanonTextEncoder
from matcha_exp.checkpoints import load_complete_state


class MatchaTTSExperiment(MatchaTTS):
    """MatchaTTS with either its native encoder or a frozen A0/A2 encoder."""

    VALID_VARIANTS = {"standard", "jepa_a0", "canopy_a2"}

    def __init__(
        self,
        *args,
        encoder_variant: str = "standard",
        canon_checkpoint: str | None = None,
        matcha_checkpoint: str | None = None,
        **kwargs,
    ):
        if encoder_variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"encoder_variant must be one of {sorted(self.VALID_VARIANTS)}, got {encoder_variant!r}"
            )
        super().__init__(*args, **kwargs)

        self.encoder_variant = encoder_variant
        if matcha_checkpoint:
            self._load_pretrained_matcha(matcha_checkpoint)

        if encoder_variant != "standard":
            if not canon_checkpoint:
                raise ValueError(f"canon_checkpoint is required for {encoder_variant}")
            encoder_cfg = self.hparams.encoder
            dp_cfg = encoder_cfg.duration_predictor_params
            self.encoder = FrozenCanonTextEncoder(
                checkpoint_path=canon_checkpoint,
                n_vocab=int(self.n_vocab),
                n_feats=int(self.n_feats),
                symbols=symbols,
                duration_filter_channels=int(dp_cfg.filter_channels_dp),
                duration_kernel_size=int(dp_cfg.kernel_size),
                duration_dropout=float(dp_cfg.p_dropout),
                n_spks=int(self.n_spks),
                spk_emb_dim=int(self.spk_emb_dim),
            )

        # Trainer resumes instantiate from Hydra config, but recording these is
        # still useful for auditability in the Lightning checkpoint.
        self.hparams.encoder_variant = encoder_variant
        self.hparams.canon_checkpoint = canon_checkpoint
        self.hparams.matcha_checkpoint = matcha_checkpoint

    def _load_pretrained_matcha(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Base Matcha checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        load_complete_state(self, state)
        print(f"[pretrained] loaded all {len(state)} tensors from {path}")

    def on_validation_end(self) -> None:
        # Upstream's visualization hook requires a logger. Fast development runs
        # disable loggers, and plotting is not part of model validation.
        if self.logger is not None:
            super().on_validation_end()
