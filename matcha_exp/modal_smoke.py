"""Bounded real-data Modal smoke test for the three Matcha experiment variants."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("MATCHA_MODAL_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
APP_NAME = f"matcha-portability-smoke-{RUN_ID}"
VOLUME_NAME = f"matcha-smoke-{RUN_ID}"
MOUNT = "/mnt/smoke"
PROJECT = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "ffmpeg", "espeak-ng", "libsndfile1", "git")
    .pip_install(
        "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4", "Cython==0.29.35", "wheel", "setuptools", "packaging",
        "lightning==2.2.5", "torchmetrics==1.3.2", "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0", "hydra-optuna-sweeper==1.2.0",
        "rootutils==1.0.7", "rich==13.7.1", "phonemizer==3.3.0",
        "tensorboard==2.16.2", "librosa==0.10.2.post1", "einops==0.7.0",
        "inflect==7.2.1", "Unidecode==1.3.8", "scipy==1.13.1",
        "matplotlib==3.8.4", "pandas==2.2.2", "conformer==0.3.2",
        "diffusers==0.25.0", "huggingface-hub==0.25.2",
        "gdown", "wget", "soundfile", "modal==1.4.2",
    )
    .add_local_dir(
        str(ROOT / "third_party/Matcha-TTS"),
        remote_path="/root/project/third_party/Matcha-TTS",
        copy=True,
        ignore=[".git", "**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .run_commands("cd /root/project/third_party/Matcha-TTS && python -m pip install --no-deps --no-build-isolation -e .")
    .add_local_dir(str(ROOT / "matcha_exp"), remote_path="/root/project/matcha_exp", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(ROOT / "modules"), remote_path="/root/project/modules", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({
        "PYTHONPATH": "/root/project:/root/project/third_party/Matcha-TTS",
        "PROJECT_ROOT": "/root/project/third_party/Matcha-TTS",
        "MATCHA_AUG_ROOT": "/mnt/smoke/augmentation",
        "MATCHA_BASE_CKPT": "/mnt/smoke/checkpoints/matcha_ljspeech.ckpt",
        "CANON_A0_CKPT": "/mnt/smoke/checkpoints/a0_jepa_s42/latest.pt",
        "CANON_A2_CKPT": "/mnt/smoke/checkpoints/a2_visreg_s42/latest.pt",
        "MATCHA_MODAL_RUN_ID": RUN_ID,
    })
)

app = modal.App(APP_NAME, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
mounts = {"/mnt/smoke": volume}


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify_remote_fixture() -> dict:
    import wave

    mount = Path(MOUNT)
    manifest = json.loads((mount / "fixture/manifest.json").read_text(encoding="utf-8"))
    assert manifest["approved_digest"] == "3f3c6a85b71d3732"
    assert (mount / "fixture/APPROVED").read_text(encoding="utf-8").strip() == manifest["approved_digest"]
    assert _sha256(mount / "fixture/augmentation_audit.json") == manifest["audit_sha256"]
    assert len((mount / "fixture/SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 17
    for name, expected in manifest["wav_files"].items():
        path = mount / "LJSpeech-1.1/wavs" / name
        assert path.stat().st_size == expected["size"] and _sha256(path) == expected["sha256"], name
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == 22050 and audio.getnchannels() == 1 and audio.getnframes() > 0
    assert len(manifest["wav_files"]) == 12
    for name, expected in manifest["filelists"].items():
        path = mount / "augmentation/filelists" / name
        assert _sha256(path) == expected, name
        actual = [line.split("|", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        split = name.split("_", 1)[0]
        assert [(Path(audio).name, text) for audio, text in actual] == [tuple(row) for row in manifest["rows"][split]]
        assert all(Path(audio).is_file() for audio, _ in actual)
    for name, expected in manifest["checkpoints"].items():
        path = mount / "checkpoints" / name
        assert path.stat().st_size == expected["size"] and _sha256(path) == expected["sha256"], name
    return manifest


def _config_for(variant: str):
    import hydra

    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(Path(PROJECT) / "third_party/Matcha-TTS/configs")):
        return hydra.compose(
            config_name="train",
            overrides=[
                f"experiment={variant}", "seed=42", "data.batch_size=2",
                "data.num_workers=0", "data.pin_memory=false", "test=false",
            ],
        )


@app.function(image=image, volumes=mounts, cpu=2, memory=8192, timeout=600, retries=0)
def cpu_preflight() -> dict:
    import hydra
    import lightning
    import torch
    import torchaudio
    import matcha
    import matcha_exp
    from matcha_exp.frozen_canon import inspect_canon_checkpoint

    manifest = _verify_remote_fixture()
    assert (Path(PROJECT) / "matcha_exp/UPSTREAM_MATCHA_COMMIT").read_text().strip() == "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
    for name in ("a0_jepa_s42/latest.pt", "a2_visreg_s42/latest.pt"):
        inspect_canon_checkpoint(Path(MOUNT) / "checkpoints" / name)
    cfg = _config_for("augmented_baseline")
    model = hydra.utils.instantiate(cfg.model)
    assert model.encoder_variant == "standard"
    assert model.n_vocab == 178 and model.n_feats == 80
    del model
    gc.collect()
    return {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "lightning": lightning.__version__, "cuda": torch.version.cuda,
        "matcha_source": str(Path(matcha.__file__).resolve()),
        "matcha_exp_source": str(Path(matcha_exp.__file__).resolve()),
        "wav_count": len(manifest["wav_files"]),
        "checkpoint_count": len(manifest["checkpoints"]),
        "strict_base_load": True,
    }


def _run_variant(variant: str, gpu_type: str) -> dict:
    import hydra
    import lightning
    import torch
    from lightning.pytorch.callbacks import Callback

    began = time.monotonic()
    _verify_remote_fixture()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_type in gpu_name, (gpu_type, gpu_name)
    if gpu_type == "A100":
        assert torch.cuda.get_device_properties(0).total_memory >= 70 * 1024**3
    torch.cuda.reset_peak_memory_stats()
    cfg = _config_for(variant)
    data = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    assert model.encoder_variant == {"augmented_baseline": "standard", "augmented_jepa_a0": "jepa_a0", "augmented_canopy_a2": "canopy_a2"}[variant]

    class Capture(Callback):
        def __init__(self):
            self.train = []
            self.val = []
            self.paths = {"train": [], "val": []}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.train.append(float(outputs["loss"].detach().float().cpu()))
            self.paths["train"].extend(batch["filepaths"])

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.val.append(float(outputs.detach().float().cpu()))
            self.paths["val"].extend(batch["filepaths"])

    capture = Capture()
    frozen = None
    grad_norms = []
    modes = []
    handles = []
    if variant != "augmented_baseline":
        encoder = model.encoder
        assert encoder.checkpoint_info["signature"]["d_model"] == 512
        assert encoder.phoneme_adapter.weight.shape == (178, 512)
        assert encoder.proj_m.out_channels == 80 and encoder.proj_w.proj.out_channels == 1
        frozen = {name: value.detach().cpu().clone() for group, module in (("embed_norm", encoder.embed_norm), ("encoder", encoder.encoder)) for name, value in module.named_parameters(prefix=group)}
        assert frozen and all(not p.requires_grad for module in (encoder.embed_norm, encoder.encoder) for p in module.parameters())
        handles.append(encoder.phoneme_adapter.weight.register_hook(lambda grad: grad_norms.append(float(grad.detach().float().norm().cpu()))))
        for module in (encoder.embed_norm, encoder.encoder):
            handles.append(module.register_forward_pre_hook(lambda module, inputs: modes.append(module.training)))

    trainer = lightning.Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed", fast_dev_run=True,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        gradient_clip_val=5.0, callbacks=[capture],
        default_root_dir="/tmp/matcha-smoke",
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 1 and len(capture.train) == 1 and len(capture.val) == 1
    assert all(__import__("math").isfinite(value) for value in capture.train + capture.val)
    assert len(capture.paths["train"]) == 2 and len(capture.paths["val"]) == 2
    assert all(Path(path).is_file() for paths in capture.paths.values() for path in paths)

    frozen_result = {}
    if frozen is not None:
        encoder = model.encoder
        for group, module in (("embed_norm", encoder.embed_norm), ("encoder", encoder.encoder)):
            assert not module.training
            for name, value in module.named_parameters(prefix=group):
                assert not value.requires_grad and value.grad is None and torch.equal(value.detach().cpu(), frozen[name]), name
        assert modes and not any(modes)
        assert grad_norms and all(__import__("math").isfinite(value) and value > 0 for value in grad_norms)
        sample = next(iter(data.val_dataloader()))
        with torch.no_grad():
            x = sample["x"].to(model.device)
            x_lengths = sample["x_lengths"].to(model.device)
            adapted = encoder.phoneme_adapter(x)
            mu, logw, mask = encoder(x, x_lengths)
        assert adapted.shape == (2, x.shape[1], 512)
        assert mu.shape == (2, 80, x.shape[1]) and logw.shape == (2, 1, x.shape[1]) and mask.shape == (2, 1, x.shape[1])
        frozen_result = {
            "checkpoint_objective": encoder.checkpoint_info["objective"],
            "adapter_shape": list(adapted.shape), "mel_shape": list(mu.shape),
            "duration_shape": list(logw.shape), "adapter_grad_norms": grad_norms,
            "frozen_parameter_count": len(frozen), "frozen_unchanged": True,
            "frozen_eval_during_forward": True,
        }

    checkpoint = Path("/tmp") / f"{variant}.ckpt"
    trainer.save_checkpoint(str(checkpoint))
    assert checkpoint.stat().st_size > 0
    fresh = hydra.utils.instantiate(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh.load_state_dict(payload["state_dict"], strict=True)
    keys = list(model.state_dict())
    assert keys and all(torch.equal(model.state_dict()[key].detach().cpu(), fresh.state_dict()[key].detach().cpu()) for key in keys)
    result = {
        "variant": variant, "gpu_name": gpu_name,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "train_loss": capture.train[0], "val_loss": capture.val[0],
        "train_wavs": [Path(path).name for path in capture.paths["train"]],
        "val_wavs": [Path(path).name for path in capture.paths["val"]],
        "global_step": trainer.global_step, "checkpoint_save_reload": True,
        "checkpoint_size": checkpoint.stat().st_size,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - began, 2),
        **frozen_result,
    }
    for handle in handles:
        handle.remove()
    del fresh, model, data, trainer, capture
    gc.collect()
    torch.cuda.empty_cache()
    return result


@app.function(image=image, volumes=mounts, gpu="A100-80GB", timeout=600, retries=0, max_containers=1)
def smoke_a100() -> dict:
    return _run_variant("augmented_baseline", "A100")


@app.function(image=image, volumes=mounts, gpu="H100!", timeout=1200, retries=0, max_containers=1)
def smoke_h100() -> dict:
    first = _run_variant("augmented_jepa_a0", "H100")
    second = _run_variant("augmented_canopy_a2", "H100")
    return {"jepa_a0": first, "canopy_a2": second}


def _upload_fixture(local_fixture: Path) -> None:
    manifest = json.loads((local_fixture / "manifest.json").read_text(encoding="utf-8"))
    approved = ROOT / "artifacts/ljspeech_augmented_medium"
    small_files = [
        (local_fixture / "manifest.json", "/fixture/manifest.json"),
        (approved / "APPROVED", "/fixture/APPROVED"),
        (approved / "SHA256SUMS", "/fixture/SHA256SUMS"),
        (approved / "augmentation_audit.json", "/fixture/augmentation_audit.json"),
    ]
    small_files.extend((local_fixture / "filelists" / name, f"/augmentation/filelists/{name}") for name in manifest["filelists"])
    small_files.extend((ROOT / "ljspeech/LJSpeech-1.1/wavs" / name, f"/LJSpeech-1.1/wavs/{name}") for name in manifest["wav_files"])
    checkpoints = [
        (ROOT / "a0_jepa_s42/latest.pt", "/checkpoints/a0_jepa_s42/latest.pt"),
        (ROOT / "a2_visreg_s42/latest.pt", "/checkpoints/a2_visreg_s42/latest.pt"),
        (ROOT / "checkpoints/matcha_ljspeech.ckpt", "/checkpoints/matcha_ljspeech.ckpt"),
    ]
    for group in (small_files, *((item,) for item in checkpoints)):
        for attempt in range(2):
            try:
                with volume.batch_upload(force=True) as batch:
                    for local, remote in group:
                        batch.put_file(local, remote)
                break
            except (OSError, TimeoutError):
                if attempt == 1:
                    raise
                time.sleep(5)


def _write_report(report: dict) -> None:
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "modal_smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Modal portability smoke test", "",
        f"Result: **{report['result']}**", "",
        f"App: `{report['app_id']}`", f"Retained Volume: `{report['volume_name']}`", "",
        "Command: `modal run -m matcha_exp.modal_smoke --expected-profile <confirmed-profile>`", "",
        "| Run | GPU | Train loss | Validation loss | Peak allocated GiB |", "|---|---|---:|---:|---:|",
    ]
    for key in ("baseline", "jepa_a0", "canopy_a2"):
        item = report["runs"].get(key, {})
        lines.append(f"| {key} | {item.get('gpu_name', 'not run')} | {item.get('train_loss', '')} | {item.get('val_loss', '')} | {item.get('peak_allocated_gib', '')} |")
    lines += ["", f"Wall minutes: {report.get('wall_minutes')}", f"Aggregate measured GPU minutes: {report.get('aggregate_gpu_minutes')}", "", "Errors:"]
    lines += [f"- {error}" for error in report["errors"]] or ["- None"]
    lines += ["", "The Volume is retained. Modal GPU and image behavior was checked only to the extent shown in the JSON report."]
    (output / "modal_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(expected_profile: str) -> None:
    from matcha_exp.scripts.prepare_modal_fixture import prepare

    profile = subprocess.check_output(["modal", "profile", "current"], text=True).strip()
    if profile != expected_profile:
        raise RuntimeError(f"Modal profile mismatch: active={profile!r}, expected={expected_profile!r}")
    started = datetime.now(timezone.utc)
    began = time.monotonic()
    report = {
        "result": "INCONCLUSIVE", "modal_profile": profile,
        "app_id": app.app_id or APP_NAME, "volume_name": VOLUME_NAME,
        "started_at_utc": started.isoformat(), "approved_digest": "3f3c6a85b71d3732",
        "git_or_source_revision": "local-worktree", "matcha_upstream_commit": "bd4d90d93214b37f7a159cf205ae85762c2c10aa",
        "assets": {"sha256": {}}, "image": {}, "runs": {},
        "cleanup": {"app_stopped": False, "volume_retained": True}, "errors": [],
        "gpu_client_minutes_upper_bound": 0.0,
    }
    gpu_attempted = False
    gpu_assertions_passed = False
    deadline_hit = threading.Event()

    def emergency_stop():
        deadline_hit.set()
        subprocess.run(["modal", "app", "stop", app.app_id or APP_NAME, "-y"], check=False, capture_output=True, text=True)

    def call_gpu(function):
        nonlocal gpu_attempted
        gpu_attempted = True
        called = time.monotonic()
        try:
            return function.remote()
        finally:
            report["gpu_client_minutes_upper_bound"] += (time.monotonic() - called) / 60

    deadline = threading.Timer(45 * 60, emergency_stop)
    deadline.daemon = True
    deadline.start()
    try:
        fixture_path = ROOT / "artifacts/modal_smoke_fixture"
        manifest = prepare(fixture_path)
        report["assets"]["sha256"] = {**{name: item["sha256"] for name, item in manifest["checkpoints"].items()}, **{name: item["sha256"] for name, item in manifest["wav_files"].items()}}
        _upload_fixture(fixture_path)
        report["image"] = cpu_preflight.remote()
        if time.monotonic() - began >= 45 * 60:
            raise TimeoutError("Wall deadline reached before GPU allocation")
        report["runs"]["baseline"] = call_gpu(smoke_a100)
        if time.monotonic() - began >= 45 * 60:
            raise TimeoutError("Wall deadline reached before H100 allocation")
        report["runs"].update(call_gpu(smoke_h100))
        gpu_assertions_passed = True
    except Exception as error:
        report["result"] = "FAIL" if gpu_attempted else "INCONCLUSIVE"
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["errors"].append(traceback.format_exc(limit=6))
        raise
    finally:
        deadline.cancel()
        try:
            stopped = subprocess.run(["modal", "app", "stop", app.app_id or APP_NAME, "-y"], capture_output=True, text=True, timeout=45)
            for _ in range(10):
                listed = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=30)
                if listed.returncode == 0:
                    apps = json.loads(listed.stdout)
                    matching = [item for item in apps if APP_NAME in str(item.get("Description", ""))]
                    report["cleanup"]["app_stopped"] = all(
                        str(item.get("State", "")).lower() in ("stopped", "disabled")
                        and str(item.get("Tasks", "0")) == "0"
                        for item in matching
                    )
                    if report["cleanup"]["app_stopped"]:
                        break
                time.sleep(2)
            if not report["cleanup"]["app_stopped"]:
                report["errors"].append("Could not verify Modal app stopped with zero tasks")
            if stopped.returncode != 0 and not report["cleanup"]["app_stopped"]:
                report["errors"].append("Modal app stop command failed")
        except Exception as cleanup_error:
            report["errors"].append(f"Modal cleanup verification failed: {cleanup_error}")
        if deadline_hit.is_set():
            report["errors"].append("45-minute wall deadline reached")
        report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_minutes"] = round((time.monotonic() - began) / 60, 3)
        report["aggregate_gpu_minutes"] = round(sum(item.get("elapsed_seconds", 0) for item in report["runs"].values()) / 60, 3)
        report["gpu_client_minutes_upper_bound"] = round(report["gpu_client_minutes_upper_bound"], 3)
        if gpu_assertions_passed:
            if (report["cleanup"]["app_stopped"] and not deadline_hit.is_set()
                    and report["wall_minutes"] < 45 and report["gpu_client_minutes_upper_bound"] < 30):
                report["result"] = "PASS"
            else:
                report["result"] = "INCONCLUSIVE"
        _write_report(report)
