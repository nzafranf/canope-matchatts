"""Validate complete pretrained initialization before mutating a model."""

from collections.abc import Mapping

import torch


def load_complete_state(model: torch.nn.Module, state: Mapping) -> None:
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    mismatched = [
        key for key in current.keys() & state.keys()
        if not isinstance(state[key], torch.Tensor) or state[key].shape != current[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Incompatible base Matcha checkpoint: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={sorted(mismatched)}"
        )
    model.load_state_dict(state, strict=True)
