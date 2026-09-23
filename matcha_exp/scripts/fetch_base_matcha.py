#!/usr/bin/env python3
"""Fetch the official LJSpeech Matcha checkpoint to an explicit path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import urllib.request
from pathlib import Path

URL = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt"
EXPECTED_SIZE = 218_852_329
EXPECTED_SHA256 = "55af8c7f2d3090c22de79311239e0d07bd8effd953661293dd54dc2f56b3f5ce"


def fetch_range(offset: int) -> bytes:
    end = min(offset + 1024 * 1024, EXPECTED_SIZE) - 1
    headers = {"User-Agent": "matcha-hpc-checkpoint-fetch", "Range": f"bytes={offset}-{end}"}
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=20) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{EXPECTED_SIZE}":
                    raise RuntimeError("Server did not honor checkpoint byte range")
                chunk = response.read()
                if len(chunk) != end - offset + 1:
                    raise OSError("Incomplete checkpoint byte range")
                return chunk
        except (OSError, TimeoutError) as error:
            if attempt == 5:
                raise RuntimeError(f"Official checkpoint range failed at offset {offset}") from error
            time.sleep(attempt)
    raise AssertionError("unreachable")


def fetch(output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > EXPECTED_SIZE:
        raise ValueError(f"Partial checkpoint exceeds official release size: {offset}")
    offsets = range(offset, EXPECTED_SIZE, 1024 * 1024)
    with ThreadPoolExecutor(max_workers=6) as pool, temporary.open("ab") as target:
        for start, chunk in zip(offsets, pool.map(fetch_range, offsets)):
            target.write(chunk)
            current = target.tell()
            if current != start + len(chunk):
                raise RuntimeError("Checkpoint assembly offset mismatch")
            if current // (32 * 1024 * 1024) != start // (32 * 1024 * 1024):
                print(f"Downloaded {current}/{EXPECTED_SIZE} bytes", flush=True)
    if temporary.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError("Official checkpoint download ended at the wrong size")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        fetch(output)
    if output.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Checkpoint size mismatch: expected {EXPECTED_SIZE}, got {output.stat().st_size}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Official checkpoint SHA-256 mismatch: {digest}")
    print(f"{digest}  {output}")


if __name__ == "__main__":
    main()
