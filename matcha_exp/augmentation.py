"""Deterministic, deliberately conservative offline text corruption."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from data.augmenter import RuleBasedAugmentor

SEED = 2027

@dataclass(frozen=True)
class AugmentationPolicy:
    version: str = "matcha-offline-conservative-v2"
    seed: int = 42
    augmentation_rate: float = 0.35
    max_corruption: float = 0.10
    max_attempts: int = 8
    lexicon_sha256: str = ""
    implementation_sha256: str = ""
    metadata_sha256: str = ""

    @property
    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def corruption_ratio(source: str, result: str) -> float:
    return 1.0 - SequenceMatcher(None, source, result).ratio()


class ConservativeOfflineAugmentor:
    """Apply medium rule and reject changes beyond the edit bound."""

    def __init__(self, lexicon_path: str | Path, policy: AugmentationPolicy):
        self.lexicon_path = str(lexicon_path)
        self.policy = policy

    def _rng_for(self, sample_id: str, stream: str = "corruption") -> random.Random:
        token = f"{self.policy.seed}:{sample_id}:{stream}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        return random.Random(seed)

    def augment(self, sample_id: str, text: str, force_change: bool = False) -> tuple[str, str, float]:
        rng = self._rng_for(sample_id)
        selected = self._rng_for(sample_id, "selection").random() < self.policy.augmentation_rate
        if not force_change and not selected:
            return text, 0.0

        augmentor = RuleBasedAugmentor(self.lexicon_path, seed=SEED)

        candidate = augmentor.augment_medium(text)
        ratio = corruption_ratio(text, candidate)
        if candidate.strip() and candidate != text and ratio <= self.policy.max_corruption:
            return candidate, ratio
        return text, 0.0
