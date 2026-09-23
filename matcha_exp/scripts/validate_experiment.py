#!/usr/bin/env python3
"""Read-only preflight validation suitable for local and HPC login nodes."""

from __future__ import annotations

import json
import os
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matcha_exp.frozen_canon import inspect_canon_checkpoint


def main() -> None:
    python_files = list((ROOT / "matcha_exp").rglob("*.py"))
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    data_root = Path(os.environ.get("MATCHA_DATA_ROOT", ROOT / "data_hpc"))
    def checkpoint_path(variable: str, staged: Path, local: Path) -> Path:
        if variable in os.environ:
            return Path(os.environ[variable])
        return staged if staged.is_file() else local

    reports = [
        inspect_canon_checkpoint(checkpoint_path("CANON_A0_CKPT", data_root / "assets/checkpoints/a0_jepa_s42/latest.pt", ROOT / "a0_jepa_s42/latest.pt")),
        inspect_canon_checkpoint(checkpoint_path("CANON_A2_CKPT", data_root / "assets/checkpoints/a2_visreg_s42/latest.pt", ROOT / "a2_visreg_s42/latest.pt")),
    ]
    assert reports[0]["objective"] == "jepa"
    assert reports[1]["objective"] == "ce" and reports[1]["zeta"] > 0
    assert reports[0]["signature"] == reports[1]["signature"]

    required = [
        ROOT / "submit_all.sh",
        ROOT / "matcha_exp/slurm/train_one.slurm",
        ROOT / "third_party/Matcha-TTS/configs/experiment/augmented_baseline.yaml",
        ROOT / "third_party/Matcha-TTS/configs/experiment/augmented_jepa_a0.yaml",
        ROOT / "third_party/Matcha-TTS/configs/experiment/augmented_canopy_a2.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing experiment files: {missing}")
    print(json.dumps({"status": "ok", "compiled": len(python_files), "checkpoints": reports}, indent=2))


if __name__ == "__main__":
    main()
