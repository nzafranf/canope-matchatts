#!/usr/bin/env python3
"""Package code and verify non-code assets retrieved separately from Drive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPROVED = "3f3c6a85b71d3732"
UPSTREAM = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
DRIVE_FOLDER_ID = "1eFzRECP1BWAvc1OZHUkRRk-kdDrp8qBZ"
MANIFEST = "matcha_hpc_manifest.json"
CHECKPOINTS = {
    "a0_jepa_s42/latest.pt": "f955605701c6ea41ef1c08e97134368999e2a60c1e7b2e201e96635e9fa0cf8d",
    "a2_visreg_s42/latest.pt": "3c30009cdedc986512cf4a4d028cbc259518061ae035692872b280d92c72e980",
    "checkpoints/matcha_ljspeech.ckpt": "55af8c7f2d3090c22de79311239e0d07bd8effd953661293dd54dc2f56b3f5ce",
}
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "outputs", "runs", "data_hpc"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".ipynb"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {name!r}")
    return path


def asset_sources(root: Path) -> dict[str, Path]:
    source = root / "artifacts/ljspeech_augmented_medium"
    paths = {
        "checkpoints/a0_jepa_s42/latest.pt": root / "a0_jepa_s42/latest.pt",
        "checkpoints/a2_visreg_s42/latest.pt": root / "a2_visreg_s42/latest.pt",
        "checkpoints/matcha_ljspeech.ckpt": root / "checkpoints/matcha_ljspeech.ckpt",
        "ljspeech/metadata.csv": root / "ljspeech/LJSpeech-1.1/metadata.csv",
        "lexicon/abbrev-lexicon.json": root / "lexicon/abbrev-lexicon.json",
    }
    paths.update({f"augmentation/{path.relative_to(source).as_posix()}": path for path in source.rglob("*") if path.is_file()})
    return paths


def verify_approved(source: Path, lexicon: Path, metadata: Path, implementation_root: Path) -> None:
    marker = (source / "APPROVED").read_text(encoding="utf-8").strip()
    if marker != APPROVED:
        raise ValueError(f"Approval digest mismatch: {marker}")
    entries = (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(entries) != 17:
        raise ValueError(f"Expected 17 approved checksums, got {len(entries)}")
    for entry in entries:
        expected, name = entry.split("  ", 1)
        actual = digest(source / safe_relative(name))
        if actual != expected:
            raise ValueError(f"Approved checksum mismatch: {name}")
    audit = json.loads((source / "augmentation_audit.json").read_text(encoding="utf-8"))
    if audit.get("policy_digest") != APPROVED or audit.get("approved_digest") != APPROVED:
        raise ValueError("Augmentation audit digest mismatch")
    policy = audit["policy"]
    if digest(lexicon) != policy["lexicon_sha256"]:
        raise ValueError("Approved lexicon checksum mismatch")
    if digest(metadata) != policy["metadata_sha256"]:
        raise ValueError("Approved metadata checksum mismatch")
    implementation = hashlib.sha256()
    for name in ("matcha_exp/augmentation.py", "data/augmenter.py", "matcha_exp/scripts/prepare_ljspeech.py"):
        implementation.update(name.encode("utf-8"))
        implementation.update((implementation_root / name).read_text(encoding="utf-8").encode("utf-8"))
    if implementation.hexdigest() != policy["implementation_sha256"]:
        raise ValueError("Approved augmentation implementation checksum mismatch")


def verify_checkpoints(paths: dict[str, Path]) -> None:
    for name, expected in CHECKPOINTS.items():
        relative = f"checkpoints/{name}" if name != "checkpoints/matcha_ljspeech.ckpt" else name
        actual = digest(paths[relative])
        if actual != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {name}: {actual}")


def verify_source_assets(root: Path) -> dict[str, Path]:
    paths = asset_sources(root)
    verify_approved(root / "artifacts/ljspeech_augmented_medium", paths["lexicon/abbrev-lexicon.json"], paths["ljspeech/metadata.csv"], root)
    verify_checkpoints(paths)
    return paths


def source_files(root: Path) -> list[Path]:
    files = [
        root / name
        for name in (
            "submit_all.sh",
            "HPC_DEPLOY.md",
            "MODAL_GPU_SMOKE_TEST.md",
            "matcha_exp/UPSTREAM_MATCHA_COMMIT",
            "data/augmenter.py",
        )
    ]
    for directory in ("matcha_exp", "third_party/Matcha-TTS", "modules"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDE_DIRS) and path.suffix not in EXCLUDE_SUFFIXES:
                files.append(path)
    result = sorted(set(files))
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")
    return result


def build(output: Path) -> None:
    assets = verify_source_assets(ROOT)
    upstream_root = ROOT / "third_party/Matcha-TTS"
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={upstream_root.as_posix()}", "-C", str(upstream_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != UPSTREAM or (ROOT / "matcha_exp/UPSTREAM_MATCHA_COMMIT").read_text().strip() != UPSTREAM:
        raise ValueError(f"Pinned Matcha revision mismatch: {revision}")
    files = source_files(ROOT)
    manifest = {
        "format": 2,
        "approved_digest": APPROVED,
        "matcha_upstream_commit": UPSTREAM,
        "drive_folder_id": DRIVE_FOLDER_ID,
        "code_files": {path.relative_to(ROOT).as_posix(): {"sha256": digest(path), "size": path.stat().st_size} for path in files},
        "asset_files": {name: {"sha256": digest(path), "size": path.stat().st_size} for name, path in sorted(assets.items())},
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with tarfile.open(temporary, "w") as archive:
        for path in files:
            archive.add(path, arcname=f"matcha-hpc/{path.relative_to(ROOT).as_posix()}", recursive=False)
        info = tarfile.TarInfo(f"matcha-hpc/{MANIFEST}")
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, output)
    archive_hash = digest(output)
    checksum_file = output.with_name(output.name + ".sha256")
    checksum_file.write_text(f"{archive_hash}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "checksum_file": str(checksum_file), "files": len(files), "bytes": output.stat().st_size, "sha256": archive_hash}, indent=2))


def verify(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != 2 or manifest.get("approved_digest") != APPROVED or manifest.get("matcha_upstream_commit") != UPSTREAM:
        raise ValueError("HPC package manifest identity mismatch")
    for name, expected in manifest["code_files"].items():
        path = root / safe_relative(name)
        if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
            raise ValueError(f"HPC package file mismatch: {name}")
    print(f"Verified {len(manifest['code_files'])} packaged code files; {len(manifest['asset_files'])} Drive assets expected")
    return manifest


def verify_assets(root: Path, data_root: Path) -> dict:
    manifest = verify(root)
    asset_root = data_root / "assets"
    for name, expected in manifest["asset_files"].items():
        path = asset_root / safe_relative(name)
        if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Drive asset mismatch: {name}")
    paths = {name: asset_root / name for name in manifest["asset_files"]}
    verify_approved(asset_root / "augmentation", paths["lexicon/abbrev-lexicon.json"], paths["ljspeech/metadata.csv"], root)
    verify_checkpoints(paths)
    print(f"Verified {len(paths)} Drive assets, approved policy, and three checkpoints")
    return manifest


def verify_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r") as archive:
        manifest_member = archive.getmember(f"matcha-hpc/{MANIFEST}")
        manifest = json.load(archive.extractfile(manifest_member))
        if manifest.get("approved_digest") != APPROVED or manifest.get("matcha_upstream_commit") != UPSTREAM:
            raise ValueError("Archive manifest identity mismatch")
        expected = {f"matcha-hpc/{name}": value for name, value in manifest["code_files"].items()}
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != len(expected) + 1 or {member.name for member in members} != set(expected) | {f"matcha-hpc/{MANIFEST}"}:
            raise ValueError("Archive has missing or unexpected files")
        for member in members:
            if member.name == f"matcha-hpc/{MANIFEST}":
                continue
            relative = member.name[len("matcha-hpc/"):]
            safe_relative(relative)
            if member.size != expected[member.name]["size"]:
                raise ValueError(f"Archive size mismatch: {member.name}")
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {member.name}")
    print(f"Verified archive: {len(expected)} files, SHA-256 {digest(archive_path)}")


def rows(path: Path) -> list[tuple[str, str]]:
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "|" not in line:
            raise ValueError(f"Invalid filelist row at {path}:{number}")
        audio, text = line.split("|", 1)
        name = audio.replace("\\", "/").rsplit("/", 1)[-1]
        if not name.endswith(".wav") or "/" in name or not text:
            raise ValueError(f"Invalid WAV or text at {path}:{number}")
        result.append((name, text))
    return result


def stage(root: Path, data_root: Path, drive_filelists: Path) -> None:
    verify_assets(root, data_root)
    source = data_root / "assets/augmentation"
    template_dir = source / "filelist_templates"
    audio_dir = data_root / "LJSpeech-1.1/wavs"
    output_dir = data_root / "augmentation/filelists"
    approved_counts = {"train": 9170, "val": 3930}
    all_audio: set[str] = set()
    rendered: dict[str, str] = {}
    for split in ("train", "val"):
        for kind in ("augmented", "clean"):
            name = f"{split}_{kind}.txt"
            approved_rows = rows(template_dir / name)
            remote_rows = rows(drive_filelists / name)
            if len(approved_rows) != approved_counts[split] or remote_rows != approved_rows:
                raise ValueError(f"Drive filelist differs from approved template: {name}")
            if kind == "augmented":
                all_audio.update(audio for audio, _ in approved_rows)
            rendered[name] = "".join(f"{(audio_dir / audio).resolve().as_posix()}|{text}\n" for audio, text in approved_rows)
    if len(all_audio) != 13100:
        raise ValueError(f"Expected 13,100 unique approved WAVs, found {len(all_audio)}")
    for name in sorted(all_audio):
        path = audio_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Approved WAV missing: {path}")
        with wave.open(str(path), "rb") as stream:
            if stream.getframerate() != 22050 or stream.getnchannels() != 1:
                raise ValueError(f"Unexpected WAV format: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        temporary = output_dir / (name + ".partial")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output_dir / name)
    for kind in ("augmented", "clean"):
        (output_dir / f"test_{kind}.txt").write_text("", encoding="utf-8")
    (data_root / "augmentation/APPROVED").write_text(APPROVED + "\n", encoding="utf-8")
    metadata = data_root / "assets/ljspeech/metadata.csv"
    target_metadata = data_root / "LJSpeech-1.1/metadata.csv"
    if target_metadata.exists() and digest(target_metadata) != digest(metadata):
        raise ValueError("Existing HPC LJSpeech metadata differs from approved source")
    if not target_metadata.exists():
        target_metadata.write_bytes(metadata.read_bytes())
    print(f"Staged 13,100 approved WAVs and absolute filelists under {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--output", default="dist/matcha_hpc_bundle.tar")
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--project-root", default=str(ROOT))
    source_command = commands.add_parser("verify-source-assets")
    source_command.add_argument("--project-root", default=str(ROOT))
    assets_command = commands.add_parser("verify-assets")
    assets_command.add_argument("--project-root", default=str(ROOT))
    assets_command.add_argument("--data-root", required=True)
    archive_command = commands.add_parser("verify-archive")
    archive_command.add_argument("archive")
    stage_command = commands.add_parser("stage")
    stage_command.add_argument("--project-root", default=str(ROOT))
    stage_command.add_argument("--data-root", required=True)
    stage_command.add_argument("--drive-filelists", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(Path(args.output).expanduser().resolve())
    elif args.command == "verify":
        verify(Path(args.project_root).expanduser().resolve())
    elif args.command == "verify-source-assets":
        paths = verify_source_assets(Path(args.project_root).expanduser().resolve())
        print(f"Verified {len(paths)} local non-code assets")
    elif args.command == "verify-assets":
        verify_assets(Path(args.project_root).expanduser().resolve(), Path(args.data_root).expanduser().resolve())
    elif args.command == "verify-archive":
        verify_archive(Path(args.archive).expanduser().resolve())
    else:
        stage(Path(args.project_root).expanduser().resolve(), Path(args.data_root).expanduser().resolve(), Path(args.drive_filelists).expanduser().resolve())


if __name__ == "__main__":
    main()
