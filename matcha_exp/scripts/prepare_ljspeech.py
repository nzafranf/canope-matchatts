#!/usr/bin/env python3
"""Preview, approve, and materialize conservative LJSpeech augmentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matcha_exp.augmentation import AugmentationPolicy, ConservativeOfflineAugmentor


def dataset_dir(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    candidate = root / "LJSpeech-1.1"
    return candidate if candidate.is_dir() else root


def read_metadata(root: Path) -> list[dict]:
    path = root / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"LJSpeech metadata not found: {path}")
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for values in csv.reader(handle, delimiter="|"):
            if len(values) < 2:
                raise ValueError(f"Malformed metadata row in {path}: {values!r}")
            sample_id = values[0]
            normalized = values[-1]
            raw = values[-2] if len(values) >= 3 else normalized
            wav = root / "wavs" / f"{sample_id}.wav"
            if not wav.is_file():
                raise FileNotFoundError(f"Missing audio for {sample_id}: {wav}")
            rows.append({"id": sample_id, "raw": raw, "text": normalized})
    if not rows:
        raise ValueError("LJSpeech metadata is empty")
    return rows


def make_policy(args) -> AugmentationPolicy:
    return bind_policy(AugmentationPolicy(
        seed=args.seed,
        augmentation_rate=args.augmentation_rate,
        max_corruption=args.max_corruption,
        max_attempts=args.max_attempts,
    ), args.lexicon_path, dataset_dir(args.dataset_root))


def bind_policy(policy: AugmentationPolicy, lexicon_path: str, dataset: Path) -> AugmentationPolicy:
    # Normalize source newlines so a Windows-to-Linux checkout preserves approval.
    implementation = hashlib.sha256()
    for relative in ("matcha_exp/augmentation.py", "data/augmenter.py",
                     "matcha_exp/scripts/prepare_ljspeech.py"):
        implementation.update(relative.encode("utf-8"))
        implementation.update((ROOT / relative).read_text(encoding="utf-8").encode("utf-8"))
    return replace(
        policy,
        lexicon_sha256=sha256_file(Path(lexicon_path)),
        implementation_sha256=implementation.hexdigest(),
        metadata_sha256=sha256_file(dataset / "metadata.csv"),
    )


def write_policy(output: Path, policy: AugmentationPolicy, dataset: Path, lexicon_path: str) -> None:
    payload = {
        "policy": asdict(policy),
        "policy_digest": policy.digest,
        "dataset": str(dataset),
        "lexicon_path": str(Path(lexicon_path).resolve()),
        "approval_required": True,
    }
    (output / "policy.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def preview(args) -> None:
    root = dataset_dir(args.dataset_root)
    rows = read_metadata(root)
    policy = make_policy(args)
    augmentor = ConservativeOfflineAugmentor(args.lexicon_path, policy)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_policy(output, policy, root, args.lexicon_path)

    rng = random.Random(policy.seed)
    chosen = list(rows)
    rng.shuffle(chosen)
    results = []
    for row in chosen:
        augmented, ratio = augmentor.augment(row["id"], row["text"])
        if args.changed_only and augmented == row["text"]:
            continue
        results.append(
            {**row, "augmented": augmented, "corruption_ratio": ratio}
        )
        if len(results) >= args.samples:
            break
    if len(results) < args.samples:
        qualifier = "changed " if args.changed_only else ""
        raise RuntimeError(
            f"Only {len(results)} {qualifier}examples could be generated from "
            f"{len(rows)} rows; requested {args.samples}."
        )

    policy_file = output / "policy.json"
    policy_payload = json.loads(policy_file.read_text(encoding="utf-8"))
    policy_payload["preview"] = {
        "changed_only": bool(args.changed_only),
        "examples": len(results),
    }
    policy_file.write_text(json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8")

    (output / "augmentation_preview.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# LJSpeech augmentation preview",
        "",
        f"Policy digest: `{policy.digest}`",
        "",
        f"Per-row augmentation selection probability: {policy.augmentation_rate:.0%}; "
        f"maximum per-row corruption: {policy.max_corruption:.0%}.",
        (
            "This review gallery contains changed rows only; clean and clean-fallback "
            "rows are omitted."
            if args.changed_only
            else "This preview includes unchanged rows exactly as the full build does."
        ),
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item['id']} ({item['corruption_ratio']:.3f})",
                "",
                f"- Original: {item['text']}",
                f"- Augmented: {item['augmented']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Approval",
            "",
            "After human review, create the approval marker with:",
            "",
            "```bash",
            f"python matcha_exp/scripts/prepare_ljspeech.py approve --preview-dir \"{output}\" --digest {policy.digest}",
            "```",
            "",
            "Do not approve if the corruption changes meaning or pronunciation too aggressively.",
        ]
    )
    (output / "augmentation_preview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Preview: {output / 'augmentation_preview.md'}")
    print(f"Policy digest requiring approval: {policy.digest}")


def approve(args) -> None:
    preview_dir = Path(args.preview_dir).resolve()
    payload = json.loads((preview_dir / "policy.json").read_text(encoding="utf-8"))
    expected = payload["policy_digest"]
    policy = AugmentationPolicy(**payload["policy"])
    if "lexicon_path" not in payload or bind_policy(
        policy, payload["lexicon_path"], Path(payload["dataset"])
    ).digest != expected:
        raise SystemExit("Preview inputs changed or preview is outdated; generate a new preview.")
    if args.digest != expected:
        raise SystemExit(f"Digest mismatch: supplied={args.digest}, expected={expected}")
    marker = preview_dir / "APPROVED"
    marker.write_text(expected + "\n", encoding="utf-8")
    print(f"Approval marker written: {marker}")


def stable_split(rows: list[dict], seed: int, train_ratio: float, val_ratio: float):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply(args) -> None:
    root = dataset_dir(args.dataset_root)
    rows = read_metadata(root)
    preview_dir = Path(args.preview_dir).resolve()
    policy_payload = json.loads((preview_dir / "policy.json").read_text(encoding="utf-8"))
    policy = AugmentationPolicy(**policy_payload["policy"])
    marker = preview_dir / "APPROVED"
    approved = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    current_policy = bind_policy(policy, args.lexicon_path, root)
    if approved != current_policy.digest or policy_payload["policy_digest"] != current_policy.digest:
        raise SystemExit(
            "Augmentation is not approved. Review augmentation_preview.md and run the approve command first."
        )

    output = Path(args.output_dir).resolve()
    manifests = output / "manifests"
    templates = output / "filelist_templates"
    manifests.mkdir(parents=True, exist_ok=True)
    templates.mkdir(parents=True, exist_ok=True)
    augmentor = ConservativeOfflineAugmentor(args.lexicon_path, policy)
    splits = stable_split(rows, policy.seed, args.train_ratio, args.val_ratio)
    counts = Counter()
    all_ratios = []

    for split, split_rows in splits.items():
        manifest_path = manifests / f"{split}.jsonl"
        template_path = templates / f"{split}_augmented.txt"
        clean_path = templates / f"{split}_clean.txt"
        with (
            manifest_path.open("w", encoding="utf-8", newline="\n") as manifest,
            template_path.open("w", encoding="utf-8", newline="\n") as augmented_file,
            clean_path.open("w", encoding="utf-8", newline="\n") as clean_file,
        ):
            for row in split_rows:
                augmented, ratio = augmentor.augment(row["id"], row["text"])
                record = {
                    "id": row["id"],
                    "wav": f"LJSpeech-1.1/wavs/{row['id']}.wav",
                    "original": row["text"],
                    "augmented": augmented,
                    "corruption_ratio": ratio,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                prefix = f"__DATA_ROOT__/LJSpeech-1.1/wavs/{row['id']}.wav"
                augmented_file.write(f"{prefix}|{augmented}\n")
                clean_file.write(f"{prefix}|{row['text']}\n")
                # counts[f"{split}/{operation}"] += 1
                all_ratios.append(ratio)

    policy_out = {
        **policy_payload,
        "approved_digest": approved,
        "rows": len(rows),
        "splits": {name: len(values) for name, values in splits.items()},
        "mean_corruption": sum(all_ratios) / max(1, len(all_ratios)),
        "max_observed_corruption": max(all_ratios, default=0.0),
    }
    (output / "augmentation_audit.json").write_text(
        json.dumps(policy_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "APPROVED").write_text(approved + "\n", encoding="utf-8")
    checksum_lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(policy_out, indent=2))
    print(f"Portable augmentation bundle written to: {output}")


def common(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-rate", type=float, default=0.35)
    parser.add_argument("--max-corruption", type=float, default=0.10)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--lexicon-path", default=str(ROOT / "lexicon" / "abbrev-lexicon.json"))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_preview = sub.add_parser("preview")
    p_preview.add_argument("--dataset-root", required=True)
    p_preview.add_argument("--output-dir", required=True)
    p_preview.add_argument("--samples", type=int, default=12)
    p_preview.add_argument(
        "--changed-only",
        action="store_true",
        help="show only rows whose full-build augmentation actually changes the text",
    )
    common(p_preview)

    p_approve = sub.add_parser("approve")
    p_approve.add_argument("--preview-dir", required=True)
    p_approve.add_argument("--digest", required=True)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--dataset-root", required=True)
    p_apply.add_argument("--preview-dir", required=True)
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--train-ratio", type=float, default=0.90)
    p_apply.add_argument("--val-ratio", type=float, default=0.05)
    common(p_apply)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "preview":
        preview(arguments)
    elif arguments.command == "approve":
        approve(arguments)
    else:
        apply(arguments)
