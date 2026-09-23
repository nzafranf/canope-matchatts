#!/usr/bin/env python3
"""Create a 12-row Modal fixture without changing approved experiment data."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matcha_exp.scripts.hpc_bundle import APPROVED, digest, rows, verify_source_assets


def prepare(output: Path) -> dict:
    verify_source_assets(ROOT)
    approved = ROOT / "artifacts/ljspeech_augmented_medium"
    wav_root = ROOT / "ljspeech/LJSpeech-1.1/wavs"
    filelists = output / "filelists"
    filelists.mkdir(parents=True, exist_ok=True)
    manifest = {
        "approved_digest": APPROVED,
        "wav_files": {},
        "filelists": {},
        "rows": {},
        "checkpoints": {},
    }
    for split, count in (("train", 8), ("val", 4)):
        source = approved / "filelist_templates" / f"{split}_augmented.txt"
        selected = rows(source)[:count]
        if len(selected) != count:
            raise ValueError(f"Too few approved {split} rows")
        manifest["rows"][split] = selected
        rendered = []
        for name, text in selected:
            wav = wav_root / name
            with wave.open(str(wav), "rb") as audio:
                if audio.getframerate() != 22050 or audio.getnchannels() != 1 or audio.getnframes() < 1:
                    raise ValueError(f"Unexpected WAV format: {wav}")
            manifest["wav_files"][name] = {"sha256": digest(wav), "size": wav.stat().st_size}
            rendered.append(f"/mnt/smoke/LJSpeech-1.1/wavs/{name}|{text}\n")
        target = filelists / f"{split}_augmented.txt"
        target.write_text("".join(rendered), encoding="utf-8")
        manifest["filelists"][target.name] = digest(target)
    if len(manifest["wav_files"]) != 12:
        raise ValueError("Expected 12 distinct approved WAVs")
    for name, path in (
        ("a0_jepa_s42/latest.pt", ROOT / "a0_jepa_s42/latest.pt"),
        ("a2_visreg_s42/latest.pt", ROOT / "a2_visreg_s42/latest.pt"),
        ("matcha_ljspeech.ckpt", ROOT / "checkpoints/matcha_ljspeech.ckpt"),
    ):
        manifest["checkpoints"][name] = {"sha256": digest(path), "size": path.stat().st_size}
    manifest["audit_sha256"] = digest(approved / "augmentation_audit.json")
    target = output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(output), "wav_count": 12, "wav_bytes": sum(item["size"] for item in manifest["wav_files"].values()), "manifest_sha256": digest(target)}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/modal_smoke_fixture")
    arguments = parser.parse_args()
    prepare(arguments.output.resolve())
