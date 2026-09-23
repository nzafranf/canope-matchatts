# MatchaTTS HPC deployment

This handoff separates **code** from **non-code assets**. `dist/matcha_hpc_bundle.tar` contains the experiment code, pinned Matcha source, configs, SLURM scripts, and a checksum manifest. Google Drive supplies the WAVs, approved augmentation files, LJSpeech metadata, lexicon, and three checkpoints. The code archive contains none of those assets.

The [shared Drive folder](https://drive.google.com/drive/folders/1eFzRECP1BWAvc1OZHUkRRk-kdDrp8qBZ?usp=sharing) currently has `dataset/wavs/` and four train/validation filelists. Those filelists match the approved text and row order, but contain Windows paths. The folder **does not yet have `assets/`**; upload those files using step 2 before attempting HPC retrieval.

## 1. Build and transfer the small code archive

On the workstation, from this repository root:

```bash
python matcha_exp/scripts/hpc_bundle.py build --output dist/matcha_hpc_bundle.tar
python matcha_exp/scripts/hpc_bundle.py verify-archive dist/matcha_hpc_bundle.tar
scp dist/matcha_hpc_bundle.tar dist/matcha_hpc_bundle.tar.sha256 USER@LOGIN_HOST:REMOTE_STAGING_DIRECTORY/
```

The builder verifies the approved digest `3f3c6a85b71d3732`, all 17 augmentation checksums, the policy source/lexicon/metadata hashes, the pinned Matcha revision, and the A0, A2, and official Matcha checkpoint hashes. It records expected Drive asset hashes in the code archive manifest. The root-level 30 MB `matcha_ljspeech.ckpt` is excluded; the official release asset is `checkpoints/matcha_ljspeech.ckpt`.

On the HPC login node:

```bash
cd REMOTE_STAGING_DIRECTORY
sha256sum -c matcha_hpc_bundle.tar.sha256
mkdir -p "$SCRATCH/matcha-project"
tar -xf matcha_hpc_bundle.tar -C "$SCRATCH/matcha-project"
cd "$SCRATCH/matcha-project/matcha-hpc"
python matcha_exp/scripts/hpc_bundle.py verify --project-root "$PWD"
```

Choose a writable location if your site does not define `$SCRATCH`. Keep training outputs and the approximately 3.8 GB WAV set on storage with enough quota.

## 2. Put the missing non-code assets in Drive

From the workstation, with a **writable** rclone Google Drive remote named `gdrive`:

```bash
bash matcha_exp/scripts/upload_to_drive.sh gdrive: "$PWD"
```

This uploads the approved bundle and the five other non-code files to `assets/`, checks the transfers, and leaves `dataset/` intact. Expected Drive layout:

```text
canope-matchatts/
├── dataset/
│   ├── wavs/                  # 13,100 WAVs
│   ├── train_augmented.txt
│   ├── train_clean.txt
│   ├── val_augmented.txt
│   └── val_clean.txt
└── assets/
    ├── augmentation/          # APPROVED, SHA256SUMS, templates, manifests, audit
    ├── checkpoints/
    │   ├── a0_jepa_s42/latest.pt
    │   ├── a2_visreg_s42/latest.pt
    │   └── matcha_ljspeech.ckpt
    ├── ljspeech/metadata.csv
    └── lexicon/abbrev-lexicon.json
```

The script needs edit access to the shared folder. If you upload manually through Drive instead, preserve this layout and exact filenames; the HPC retrieval rejects missing or changed assets.

## 3. Retrieve from Drive on the HPC login node

Use **rclone** for this 13,100-file folder. Set up a Google Drive remote with `drive.readonly` scope and an [own OAuth client ID](https://rclone.org/drive/#making-your-own-client-id), or a service account to which the folder is shared. For a headless login node, follow [rclone's remote setup](https://rclone.org/remote_setup/) on a browser-equipped machine. Keep its config and token outside the project and protect them with user-only permissions. Rclone's shared client ID is being retired during 2026.

In practice, run `rclone config` on the login node, create a remote named `gdrive` of type `drive`, select `drive.readonly`, and answer **No** when asked whether that machine can open a browser. Rclone then gives an `rclone authorize` instruction to run on your laptop. Paste its result into the interactive HPC prompt, never into a command file or this repository. Alternatively, configure a separate read-only remote on your laptop and transfer its rclone config to the HPC account with `scp`, then set its permissions to `600`.

Confirm the shared folder ID is accessible:

```bash
rclone lsf gdrive: --drive-root-folder-id 1eFzRECP1BWAvc1OZHUkRRk-kdDrp8qBZ --max-depth 1
```

It should list `dataset/` and `assets/`. From the extracted project root:

```bash
bash matcha_exp/scripts/retrieve_from_drive.sh gdrive: "$SCRATCH/matcha-data" "$PWD"
export MATCHA_DATA_ROOT="$SCRATCH/matcha-data"
export MATCHA_AUG_ROOT="$MATCHA_DATA_ROOT/augmentation"
```

The script downloads and verifies `assets/` first. It then copies the Drive filelists and WAVs, uses `rclone check` on both transfers, compares every filelist row and exact text to the approved templates, checks 13,100 unique mono 22,050 Hz WAVs, and renders Linux absolute paths in a separate runtime directory. It never rewrites the approved augmentation bundle. Re-running it resumes/skips completed rclone transfers.

**gdown is suitable for a small number of public files or archives**, but not this Drive layout. On 2026-09-24, Google's public embedded view of `dataset/wavs/` exposed about 5,500 of the expected 13,100 files. [Gdown's folder implementation](https://github.com/wkentaro/gdown/blob/main/gdown/download_folder.py) reads that view, so a folder download could silently omit WAVs. If OAuth is unavailable, a practical fallback is to upload a few checksum-verified audio archives and use `gdown --continue` on each archive; that layout is not currently in Drive.

## 4. Install and submit

Load your site's Conda or Mamba module, then:

```bash
bash matcha_exp/scripts/setup_hpc.sh
conda activate matcha-canon
python matcha_exp/scripts/validate_experiment.py
```

The environment pins compatible PyTorch, torchvision, and torchaudio CUDA 12.1 builds, NumPy 1.26, and the diffusers/Hugging Face Hub APIs used by the pinned Matcha source. Set the actual SLURM partition and GPU resource names for your cluster; these are examples:

```bash
export MATCHA_PROJECT_ROOT="$PWD"
export MATCHA_DATA_ROOT="$SCRATCH/matcha-data"
export MATCHA_AUG_ROOT="$MATCHA_DATA_ROOT/augmentation"
export MATCHA_OUTPUT_ROOT="$SCRATCH/matcha-outputs"
export MATCHA_PARTITION=gpu-long
export MATCHA_GPU_A100=a100-80
export MATCHA_GPU_H100=h100-96
mkdir -p logs
sbatch --partition="$MATCHA_PARTITION" submit_all.sh
```

Set `SBATCH_ACCOUNT` and `SBATCH_QOS` if required. Complete the bounded GPU smoke test in `MODAL_GPU_SMOKE_TEST.md` or an equivalent cluster allocation before long training; packaging and transfer checks alone cannot verify CUDA training.
