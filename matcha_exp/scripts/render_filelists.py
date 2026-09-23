#!/usr/bin/env python3
"""Render portable filelist templates after the dataset is retrieved on HPC."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing LJSpeech-1.1")
    parser.add_argument("--augmentation-root", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    augmentation_root = Path(args.augmentation_root).expanduser().resolve()
    ljspeech = data_root / "LJSpeech-1.1"
    if not (ljspeech / "metadata.csv").is_file():
        raise SystemExit(f"LJSpeech not found below {data_root}")
    if not (augmentation_root / "APPROVED").is_file():
        raise SystemExit(f"Approved augmentation marker missing below {augmentation_root}")

    output = augmentation_root / "filelists"
    output.mkdir(parents=True, exist_ok=True)
    prefix = data_root.as_posix()
    for template in sorted((augmentation_root / "filelist_templates").glob("*.txt")):
        rendered = template.read_text(encoding="utf-8").replace("__DATA_ROOT__", prefix)
        target = output / template.name
        target.write_text(rendered, encoding="utf-8")
        for line_number, line in enumerate(rendered.splitlines(), 1):
            audio = Path(line.split("|", 1)[0])
            if not audio.is_file():
                raise SystemExit(f"Missing WAV referenced by {target}:{line_number}: {audio}")
        print(f"Rendered and validated: {target}")

    checksum_lines = []
    for path in sorted(
        path
        for path in augmentation_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    ):
        relative = path.relative_to(augmentation_root).as_posix()
        checksum_lines.append(f"{sha256_file(path)}  {relative}")
    (augmentation_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Refreshed checksums: {augmentation_root / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
