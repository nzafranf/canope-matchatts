from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from matcha_exp.augmentation import AugmentationPolicy, ConservativeOfflineAugmentor
from matcha_exp.frozen_canon import FrozenCanonTextEncoder, inspect_canon_checkpoint
from modules.model import CanonCanvas


ROOT = Path(__file__).resolve().parents[1]


def make_checkpoint(path: Path) -> None:
    model = CanonCanvas(
        n_vocab_in=256,
        d_model=32,
        n_attn_heads=4,
        enc_layers=1,
        dec_layers=1,
        max_length=64,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "signature": model.architecture_signature(),
            "args": {"objective": "jepa", "zeta": 0.0, "dropout": 0.0, "tau_r": 0.3},
            "step": 17,
        },
        path,
    )


def test_checkpoint_inspection_and_frozen_adapter(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    make_checkpoint(checkpoint)
    info = inspect_canon_checkpoint(checkpoint)
    assert info["step"] == 17
    assert info["signature"]["d_model"] == 32

    symbols = ["_", "a", "b", "t", "ʃ", " "]
    encoder = FrozenCanonTextEncoder(
        checkpoint,
        n_vocab=len(symbols),
        n_feats=8,
        symbols=symbols,
        duration_filter_channels=16,
    )
    encoder.train()
    assert not encoder.encoder.training
    assert all(not p.requires_grad for p in encoder.encoder.parameters())
    assert encoder.phoneme_adapter.weight.requires_grad

    x = torch.tensor([[1, 2, 4, 0], [3, 5, 1, 2]])
    lengths = torch.tensor([3, 4])
    mu, logw, mask = encoder(x, lengths)
    assert mu.shape == (2, 8, 4)
    assert logw.shape == (2, 1, 4)
    assert mask.shape == (2, 1, 4)
    (mu.mean() + logw.mean()).backward()
    assert encoder.phoneme_adapter.weight.grad is not None
    assert all(p.grad is None for p in encoder.encoder.parameters())


def test_augmentation_is_deterministic_and_bounded():
    policy = AugmentationPolicy(seed=7, augmentation_rate=1.0, max_corruption=0.10)
    augmentor = ConservativeOfflineAugmentor(ROOT / "lexicon/abbrev-lexicon.json", policy)
    text = "The careful librarian couldn't locate the requested volume."
    first = augmentor.augment("LJ000-0001", text, force_change=True)
    second = augmentor.augment("LJ000-0001", text, force_change=True)
    assert first == second
    assert first[1] <= policy.max_corruption
    assert first[0].strip()


def test_apply_refuses_without_approval(tmp_path):
    dataset = tmp_path / "LJSpeech-1.1"
    (dataset / "wavs").mkdir(parents=True)
    (dataset / "wavs/LJ001-0001.wav").write_bytes(b"RIFF")
    (dataset / "metadata.csv").write_text(
        "LJ001-0001|Raw text|Normalized sample text.\n", encoding="utf-8"
    )
    preview = tmp_path / "preview"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "matcha_exp/scripts/prepare_ljspeech.py"),
            "preview",
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(preview),
            "--samples",
            "1",
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "matcha_exp/scripts/prepare_ljspeech.py"),
            "apply",
            "--dataset-root",
            str(dataset),
            "--preview-dir",
            str(preview),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "not approved" in (result.stdout + result.stderr).lower()

    policy = json.loads((preview / "policy.json").read_text(encoding="utf-8"))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "matcha_exp/scripts/prepare_ljspeech.py"),
            "approve",
            "--preview-dir",
            str(preview),
            "--digest",
            policy["policy_digest"],
        ],
        check=True,
    )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "matcha_exp/scripts/prepare_ljspeech.py"),
            "apply",
            "--dataset-root",
            str(dataset),
            "--preview-dir",
            str(preview),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "matcha_exp/scripts/render_filelists.py"),
            "--data-root",
            str(tmp_path),
            "--augmentation-root",
            str(output),
        ],
        check=True,
    )
    assert (output / "filelists/test_augmented.txt").is_file()


def test_three_slurm_gpu_assignments_and_config_paths():
    common = (ROOT / "matcha_exp/slurm/common.sh").read_text(encoding="utf-8")
    submit = (ROOT / "matcha_exp/slurm/submit_all.sh").read_text(encoding="utf-8")
    assert 'baseline) echo "$MATCHA_GPU_A100"' in common
    assert 'jepa_a0|canopy_a2) echo "$MATCHA_GPU_H100"' in common
    assert "RUNS=(baseline jepa_a0 canopy_a2)" in submit
    for name in ("augmented_baseline", "augmented_jepa_a0", "augmented_canopy_a2"):
        assert (ROOT / f"third_party/Matcha-TTS/configs/experiment/{name}.yaml").is_file()


def test_real_checkpoint_specs_are_expected():
    expected = {
        "a0_jepa_s42/latest.pt": ("jepa", 0.3),
        "a2_visreg_s42/latest.pt": ("ce", 0.1),
    }
    for relative, (objective, zeta) in expected.items():
        path = ROOT / relative
        if not path.is_file():
            pytest.skip(f"large checkpoint not present: {path}")
        info = inspect_canon_checkpoint(path)
        assert info["objective"] == objective
        assert info["zeta"] == zeta
        assert info["signature"]["d_model"] == 512
        assert info["signature"]["encoder_layers"] == 4
