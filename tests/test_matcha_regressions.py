"""Regression checks without GPU training, scheduler access, or Drive access."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from matcha_exp.augmentation import AugmentationPolicy, ConservativeOfflineAugmentor
from matcha_exp.checkpoints import load_complete_state
from matcha_exp.scripts import prepare_ljspeech as prepare

ROOT = Path(__file__).resolve().parents[1]


def test_preview_corruption_matches_selected_rows():
    aug = ConservativeOfflineAugmentor(ROOT / "lexicon/abbrev-lexicon.json", AugmentationPolicy())
    selected = 0
    for index in range(50):
        text = "The careful librarian couldn't locate the requested volume."
        actual = aug.augment(str(index), text)
        if actual[0] != text:
            selected += 1
            assert actual == aug.augment(str(index), text)
    assert selected > 0


@pytest.fixture
def preview_args(tmp_path):
    dataset = tmp_path / "LJSpeech-1.1"
    (dataset / "wavs").mkdir(parents=True)
    for i in range(20):
        (dataset / "wavs" / f"LJ{i}.wav").write_bytes(b"RIFF")
    (dataset / "metadata.csv").write_text("".join(
        f"LJ{i}|Raw|The careful librarian couldn't locate the requested volume.\n"
        for i in range(20)
    ), encoding="utf-8")
    lexicon = tmp_path / "lexicon.json"
    shutil.copyfile(ROOT / "lexicon/abbrev-lexicon.json", lexicon)
    return SimpleNamespace(
        dataset_root=str(dataset), output_dir=str(tmp_path / "preview"),
        preview_dir=str(tmp_path / "preview"), lexicon_path=str(lexicon),
        seed=42, augmentation_rate=0.35, max_corruption=0.10, max_attempts=8,
        samples=20, changed_only=False, train_ratio=0.9, val_ratio=0.05,
    )


def approve_preview(args):
    prepare.preview(args)
    payload = json.loads((Path(args.preview_dir) / "policy.json").read_text())
    args.digest = payload["policy_digest"]
    prepare.approve(args)
    args.output_dir = str(Path(args.preview_dir).parent / "bundle")


def test_preview_equals_generated_manifest(preview_args):
    args = preview_args
    approve_preview(args)
    prepare.apply(args)
    preview = json.loads((Path(args.preview_dir) / "augmentation_preview.json").read_text())
    generated = {}
    for path in (Path(args.output_dir) / "manifests").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            generated[row["id"]] = row
    for row in preview:
        assert row["augmented"] == generated[row["id"]]["augmented"]
        assert row["corruption_ratio"] == generated[row["id"]]["corruption_ratio"]


@pytest.mark.parametrize("changed", ["lexicon", "metadata", "implementation"])
def test_approval_rejects_changed_inputs(preview_args, monkeypatch, changed):
    args = preview_args
    approve_preview(args)
    if changed == "lexicon":
        Path(args.lexicon_path).write_text('{}', encoding="utf-8")
    elif changed == "metadata":
        path = Path(args.dataset_root) / "metadata.csv"
        path.write_text(path.read_text().replace("careful", "helpful"), encoding="utf-8")
    else:
        fake_root = Path(args.preview_dir).parent / "source"
        for relative in ("matcha_exp/augmentation.py", "data/augmenter.py",
                         "matcha_exp/scripts/prepare_ljspeech.py"):
            target = fake_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / relative).read_text(encoding="utf-8") + "\n# changed\n",
                              encoding="utf-8")
        monkeypatch.setattr(prepare, "ROOT", fake_root)
    with pytest.raises(SystemExit, match="not approved"):
        prepare.apply(args)
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("defect", ["missing", "shape", "unexpected", None])
def test_complete_checkpoint_required_before_loading(defect):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: torch.ones_like(v) for k, v in original.items()}
    if defect == "missing":
        state.pop("1.weight")
    elif defect == "shape":
        state["1.weight"] = torch.ones(1)
    elif defect == "unexpected":
        state["extra"] = torch.ones(1)
    if defect:
        with pytest.raises(ValueError, match="Incompatible base Matcha"):
            load_complete_state(model, state)
        assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    else:
        load_complete_state(model, state)
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def shell_path(path):
    value = Path(path).resolve().as_posix()
    if os.name == "nt":
        value = "/" + value[0].lower() + value[2:]
    return shlex.quote(value)


@pytest.fixture
def shell_project(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash required for orchestration regression tests")
    project = tmp_path / "project"
    shutil.copytree(ROOT / "matcha_exp/slurm", project / "matcha_exp/slurm")
    def run(script):
        prefix = f'''
export PATH="/usr/bin:/bin:$PATH"
export USER=review
cd {shell_path(project)}
export MATCHA_PROJECT_ROOT=.
export MATCHA_OUTPUT_ROOT=outputs
export MATCHA_AUG_ROOT=augmentation
export MATCHA_DATA_ROOT=data_hpc
export MATCHA_PARTITION=review-partition
export MATCHA_SEED=42
export MATCHA_WATCHDOG_INTERVAL=0
sbatch() {{ echo "SUBMIT $*" >&2; echo 123; }}
sacct() {{ return 0; }}
sleep() {{ exit 0; }}
'''
        return subprocess.run([str(bash), "-s"], input=prefix + script, text=True,
                              capture_output=True, timeout=20)
    return project, run


@pytest.mark.parametrize("queue", ["return 1", "printf '%s\\n' mt-baseline-s42 mt-jepa_a0-s42 mt-canopy_a2-s42"])
def test_watchdog_does_not_resubmit_unknown_or_active_jobs(shell_project, queue):
    project, run = shell_project
    result = run('squeue() { ' + queue + '; }\nsource matcha_exp/slurm/watchdog.slurm\n')
    assert result.returncode == 0, result.stderr
    assert "SUBMIT" not in result.stderr
    assert not list((project / "outputs/watchdog").glob("*.retries"))


@pytest.mark.parametrize("accepted", [True, False])
def test_watchdog_retries_only_accepted_submissions(shell_project, accepted):
    project, run = shell_project
    failure = "" if accepted else "sbatch() { return 1; }\n"
    result = run('squeue() { return 0; }\n' + failure + 'source matcha_exp/slurm/watchdog.slurm\n')
    assert result.returncode == 0, result.stderr
    counters = list((project / "outputs/watchdog").glob("*.retries"))
    assert len(counters) == (3 if accepted else 0)
    if accepted:
        assert result.stderr.count("--partition=review-partition") == 3
        assert result.stderr.count("--gres=gpu:h100-96:1") == 2
        assert result.stderr.count("--gres=gpu:a100-80:1") == 1


def test_submit_propagates_partition_to_all_children(shell_project):
    project, run = shell_project
    for relative in ("augmentation/APPROVED", "augmentation/filelists/train_augmented.txt",
                     "augmentation/filelists/val_augmented.txt",
                     "data_hpc/assets/checkpoints/matcha_ljspeech.ckpt",
                     "data_hpc/assets/checkpoints/a0_jepa_s42/latest.pt",
                     "data_hpc/assets/checkpoints/a2_visreg_s42/latest.pt"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (project / "augmentation/APPROVED").write_text("3f3c6a85b71d3732\n", encoding="utf-8")
    result = run('squeue() { return 0; }\nsource matcha_exp/slurm/submit_all.sh\n')
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("SUBMIT") == 5
    assert result.stderr.count("--partition=review-partition") == 5


def test_watchdog_replacement_inherits_partition(shell_project):
    _, run = shell_project
    result = run('''
squeue() { printf '%s\n' mt-baseline-s42 mt-jepa_a0-s42 mt-canopy_a2-s42; }
sleep() { resubmit_watchdog; }
source matcha_exp/slurm/watchdog.slurm
''')
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("SUBMIT") == 1
    assert "--partition=review-partition" in result.stderr
    assert "watchdog.slurm" in result.stderr


def test_retrieve_creates_checkpoint_destinations(shell_project):
    project, run = shell_project
    result = run(f'''
rclone() {{ return 0; }}
python() {{ return 0; }}
source {shell_path(ROOT / 'matcha_exp/scripts/retrieve_from_drive.sh')} remote: "$MATCHA_PROJECT_ROOT/data" "$MATCHA_PROJECT_ROOT"
''')
    assert result.returncode == 0, result.stderr
    for relative in ("data/assets", "data/LJSpeech-1.1/wavs", "data/drive_filelists"):
        assert (project / relative).is_dir()
