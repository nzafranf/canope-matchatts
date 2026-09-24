# Modal A100/H100 portability smoke-test runbook

## Objective and non-goals

This is a bounded integration test on a clean Linux/CUDA machine. It proves that the repository can be packaged, dependencies imported, approved augmented filelists read, real LJSpeech audio decoded, the official Matcha checkpoint strictly loaded, all three variants run one training and one validation batch, frozen encoders remain frozen, and output checkpoints can be loaded.

It is not a training run, benchmark, hyperparameter search, Drive upload, or SLURM submission. Do not run the complete 13,100-row dataset on Modal.

## Mandatory safety gates

1. Run `modal profile current` before creating any cloud resource.
2. The requester confirmed account/profile `ganeshataqwa01`. Check that `modal profile current` returns this exact string before creating cloud resources.
3. Never print, copy, or commit Modal tokens.
4. Never use `--detach`. The launching process must retain control and stop the ephemeral app when it exits.
5. Use the exact GPU selectors `A100-80GB` and `H100!`. The exclamation mark prevents the H100 request from being fulfilled by a different GPU type.
6. Execute the GPU stages sequentially: A100 first, then H100. Fail fast; do not launch the H100 stage if the common build or A100 stage fails.
7. Configure zero user-code retries. Function timeout must be 600 seconds for A100 and 1,200 seconds for H100. This bounds nominal accelerator allocation to 30 GPU-minutes. Target under 15 minutes and enforce a 45-minute wall-clock kill deadline for the entire test, including queue time.
8. Monitor the Modal app and usage while it runs. On deadline, exception, lost controller connection, or unexpected retry, stop it with `modal app stop <APP_ID_OR_NAME> -y` and verify it is no longer running.
9. Do not start full training, upload to Google Drive, run `sbatch`, or mutate the approved augmentation artifacts.

The platform can retry infrastructure-preempted work independently of user-code retry settings, so the human/agent must still monitor the app and stop it if the bounded plan is exceeded.

## Inputs

Required local inputs:

- repository source;
- `artifacts/ljspeech_augmented_medium/APPROVED`, whose contents must be `3f3c6a85b71d3732`;
- `artifacts/ljspeech_augmented_medium/SHA256SUMS`, which must verify before upload;
- A0 checkpoint: `a0_jepa_s42/latest.pt`;
- A2 checkpoint: `a2_visreg_s42/latest.pt`;
- official Matcha checkpoint: `checkpoints/matcha_ljspeech.ckpt`;
- original WAV files referenced by a deliberately small subset of the approved manifests.

The official Matcha checkpoint is now present locally. Its SHA-256 is `55af8c7f2d3090c22de79311239e0d07bd8effd953661293dd54dc2f56b3f5ce`. If it is absent in another checkout, fetch it before any GPU allocation:

```bash
python matcha_exp/scripts/fetch_base_matcha.py \
  --output checkpoints/matcha_ljspeech.ckpt
```

Do not silently substitute a random or synthetic base checkpoint for the GPU integration test.

## Test fixture and cloud storage

Create a temporary fixture locally or in a CPU-only Modal function:

- first 8 entries from the approved training manifest;
- first 4 entries from the approved validation manifest;
- the corresponding 12 WAV files only;
- rewritten absolute filelists inside the remote container;
- approval marker, audit JSON, and checksum evidence;
- A0, A2, and Matcha checkpoints.

Preserve the exact approved text strings. The subset is only for reducing compute and transfer; it must not regenerate or modify augmentations.

Use a uniquely named Modal Volume such as `matcha-smoke-<UTC timestamp>` and a uniquely named app. Upload only this fixture and the three checkpoints. Do not upload all LJSpeech audio. Retain the Volume until the report has been downloaded and inspected; deletion requires the requester's explicit approval.

## Required Modal entrypoint

The testing agent should implement a small, auditable script at `matcha_exp/modal_smoke.py`. It should use a Python 3.10 Debian-slim image, install the repository and pinned Matcha source, and add Linux audio/text dependencies such as `ffmpeg`, `espeak-ng`, and `libsndfile1`.

The structure should be equivalent to:

```python
import modal

app = modal.App("matcha-portability-smoke-<unique-suffix>")

@app.function(
    gpu="A100-80GB",
    timeout=600,
    retries=0,
    # image=..., volumes=..., secrets are not required
)
def smoke_a100():
    # Environment diagnostics, baseline fast-dev run, save/load check.
    ...

@app.function(
    gpu="H100!",
    timeout=1200,
    retries=0,
    # image=..., volumes=..., secrets are not required
)
def smoke_h100():
    # Run JEPA A0 then CANOPE A2 sequentially in this one allocation.
    ...

@app.local_entrypoint()
def main():
    # Preflight first; call A100; only on PASS call H100; collect report.
    ...
```

Do not embed the entire LJSpeech directory in the image. Use a Volume for the small runtime fixture, and exclude local outputs, `.git`, caches, and unrelated experiment artifacts from the image context.

## Preflight with no GPU

Complete all of these before calling a GPU function:

1. Confirm the Modal profile/account gate.
2. Confirm the pinned upstream commit from `matcha_exp/UPSTREAM_MATCHA_COMMIT`.
3. Verify the approved digest and all 17 entries in the augmentation `SHA256SUMS` file.
4. Verify that the subset WAVs exist and can be decoded at 22,050 Hz.
5. Verify all three checkpoints exist and record their SHA-256 hashes and sizes.
6. Run Python compilation and the local tests that do not require CUDA.
7. Build the Modal image in a CPU-only phase and run import checks for `torch`, `torchaudio`, `lightning`, `matcha`, and `matcha_exp`.
8. Run `python matcha_exp/scripts/validate_experiment.py` against the packaged layout.
9. Confirm no Modal Secret is attached and no credentials will enter logs.

If any preflight fails, stop without allocating a GPU.

## A100 stage: baseline

Inside the `A100-80GB` function:

1. Record Python, PyTorch, CUDA runtime, cuDNN, GPU name, and total memory.
2. Assert exactly one visible CUDA device and GPU name contains `A100`.
3. Render remote absolute filelists for the 8/4 fixture.
4. Run the `augmented_baseline` experiment with:
   - `trainer.fast_dev_run=true`;
   - one device;
   - `bf16-mixed` precision;
   - batch size 2;
   - zero dataloader workers;
   - TensorBoard or external loggers disabled;
   - test disabled.
5. Confirm one real training batch and one real validation batch complete with finite losses and no CUDA OOM.
6. Explicitly save a small Lightning checkpoint after initialization/step, instantiate a fresh model, load it, and verify key parameters match. `fast_dev_run` may suppress normal checkpoint callbacks, so this must be a deliberate save/load check.
7. Write structured results to the report Volume, then release CUDA memory.

Representative Hydra overrides (adjust only to match the installed pinned Matcha API):

```bash
python -m matcha.train \
  experiment=augmented_baseline \
  seed=42 \
  trainer.fast_dev_run=true \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  data.batch_size=2 \
  data.num_workers=0 \
  data.pin_memory=false \
  test=false
```

Do not proceed to H100 unless every baseline acceptance check passes.

## H100 stage: both frozen encoders

Inside one exact `H100!` allocation, run `augmented_jepa_a0` and then `augmented_canopy_a2`, never concurrently. Use the same fixture and overrides as baseline.

For each variant, additionally assert:

- checkpoint specification is recognized and tensor shapes match;
- input adaptor output shape matches the expected 512-channel encoder input;
- the imported Canon encoder and embedding-normalization parameters have `requires_grad=False`;
- those modules remain in evaluation mode during training;
- frozen parameters receive no gradients and do not change after the optimizer step;
- the phoneme adaptor receives a finite, nonzero gradient;
- prior/mel and duration projections have the expected shapes;
- one training and one validation batch finish with finite losses and no OOM.

Record peak allocated and reserved CUDA memory after each variant. Destroy the first model and call Python garbage collection plus `torch.cuda.empty_cache()` before constructing the second.

## Static SLURM portability checks

These require no SLURM cluster and must not call `sbatch`:

- `submit_all.sh` maps baseline to A100 and both frozen runs to H100;
- all runs share the same augmented train/validation paths;
- the watchdog caps retries and skips resubmission after a failed `squeue` query;
- TensorBoard and watchdog are CPU jobs;
- checkpoint resume uses `last.ckpt` and success markers prevent resubmission.

## Acceptance criteria

The overall result is `PASS` only if:

- profile/account was explicitly correct;
- approved digest and checksums matched;
- clean Modal image built without relying on workstation-installed packages;
- exact requested GPU types were observed;
- all three variants consumed real WAVs and approved augmented text;
- one train and one validation batch completed per variant;
- all losses and gradients required above were finite;
- frozen encoder assertions passed for A0 and A2;
- strict base checkpoint load and explicit checkpoint save/reload passed;
- aggregate accelerator time remained under 30 minutes and overall wall time under 45 minutes;
- no live Modal app/function remained after completion;
- no secrets appeared in logs.

Any workaround, skipped assertion, GPU substitution, or timeout produces `FAIL` or `INCONCLUSIVE`, never `PASS`.

## Required report

Download and commit neither large checkpoints nor WAVs. Save these local, git-ignored artifacts:

- `artifacts/modal_smoke_report.json`
- `artifacts/modal_smoke_report.md`

The JSON report must include:

```json
{
  "result": "PASS|FAIL|INCONCLUSIVE",
  "modal_profile": "redacted-or-profile-name",
  "app_id": "...",
  "started_at_utc": "...",
  "ended_at_utc": "...",
  "wall_minutes": 0,
  "aggregate_gpu_minutes": 0,
  "approved_digest": "3f3c6a85b71d3732",
  "git_or_source_revision": "...",
  "matcha_upstream_commit": "bd4d90d93214b37f7a159cf205ae85762c2c10aa",
  "assets": {"sha256": {}},
  "image": {"python": "...", "torch": "...", "cuda": "..."},
  "runs": {
    "baseline": {},
    "jepa_a0": {},
    "canopy_a2": {}
  },
  "cleanup": {"app_stopped": true, "volume_retained": true},
  "errors": []
}
```

The Markdown report should give the exact commands, image definition, relevant short logs, observed GPU models, peak memory, elapsed times, acceptance table, any source changes, and a final recommendation for or against NUS submission.

## Launch and emergency stop

After the script has passed CPU-only review:

```bash
modal profile current
modal run -m matcha_exp.modal_smoke --expected-profile ganeshataqwa01
```

Do not use `--detach`. In another terminal, watch the app in the Modal dashboard or CLI. Emergency stop:

```bash
modal app stop <APP_ID_OR_NAME> -y
```

Finally verify that the app is stopped and copy the reports out before requesting permission to remove the temporary Volume.

## Modal references

- [GPU requests and exact accelerator selection](https://modal.com/docs/guide/gpu)
- [Function execution and configuration](https://modal.com/docs/guide/functions)
- [Function timeouts](https://modal.com/docs/guide/timeouts)
- [Volumes](https://modal.com/docs/guide/volumes)
- [Profiles](https://modal.com/docs/cli/latest/profile)
- [`modal run`](https://modal.com/docs/cli/latest/run)
- [`modal app stop`](https://modal.com/docs/cli/latest/app)
