#!/usr/bin/env python3
"""Print and validate the embedded specs in both Canon latest.pt files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matcha_exp.frozen_canon import inspect_canon_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [inspect_canon_checkpoint(path) for path in args.paths]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

