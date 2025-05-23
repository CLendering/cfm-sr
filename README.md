# CFM-SR: Conditional Flow Matching for Super-Resolution

This project implements a Conditional Flow Matching (CFM) model using a U-Net architecture for face image super-resolution. It includes scripts for data preparation, training, and evaluation.

## Core Features

* **Conditional Flow Matching (CFM):** For high-resolution image generation.
* **U-Net Architecture:** With time embeddings and FiLM conditioning.
* **Comprehensive Scripts:** For dataset processing (resizing, splitting, normalization), model training (with LPIPS/ID loss options, resumable), and evaluation (PSNR, SSIM, LPIPS, FID, ID Cosine Similarity).
* **Z-Normalization:** Applied to image data.

## Project Structure

* `datasets/`: Data preparation scripts and `CelebASuperRes` Dataset.
* `models/`: `UNetSR` model definition.
* `train_sr.py`: Training script.
* `eval_sr.py`: Evaluation script.

## Quick Start

### 1. Setup Dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `torchvision`, `lpips`, `torchmetrics`, `pytorch-fid`, `facenet-pytorch`, `einops`, `pandas`, `seaborn`, `matplotlib`, `tqdm`, `tensorboard`.

### 2. Dataset Preparation

The scripts in the `datasets/` directory are designed to work with a dataset like CelebA-HQ.

* **Place initial data:** e.g., `celeba_hq_256/`
* **Resize images:** Run `python datasets/prepare_data.py` (adjust paths in script if needed). This creates e.g., `data/celeba_hq_128/`.
* **Create splits:** Run `python datasets/make_split.py` (adjust `ROOT` and `SIZES` in script). This creates `data/train/`, `data/val/`, `data/test/`.
* **Calculate mean/std:** Run `python datasets/get_mean_std.py` (adjust `train_data_path`). Update `CALCULATED_MEAN` and `CALCULATED_STD` in relevant scripts if values differ.

### 3. Training

Use `train_sr.py`. Example:

```bash
python train_sr.py \
    --data_dir ./data \
    --out_dir checkpoints/cfm_sr_run \
    --run_name cfm_sr_tb \
    --bs 32 \
    --lr 1e-4 \
    --epochs 300 \
    --lambda_lpips 0.1 \
    --best_metric lpips
```

See `python train_sr.py --help` for all options.

### 4. Evaluation

Use `eval_sr.py` with a trained checkpoint. Example:

```bash
python eval_sr.py \
    --data_dir ./data \
    --ckpt checkpoints/cfm_sr_run/best_model.pt \
    --out_dir evaluation_results/cfm_sr_run \
    --steps 50
```

See `python eval_sr.py --help` for all options.

## Model

The `UNetSR` in `models/u_net_flow.py` uses time embeddings, FiLM-conditioned residual blocks, and optional self-attention. It predicts velocity for an ODE solver.