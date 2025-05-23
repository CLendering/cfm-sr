import os
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import logging
from datasets.celeba_sr import CelebASuperRes

try:
    import lpips
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ImportError:
    logging.error("Please install required libraries: pip install lpips torchmetrics")
    exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    from models.u_net_flow import UNetSR
except ImportError:
    logging.error(
        "Failed to import UNetSR from models.u_net_flow. Please ensure the file and class exist."
    )
    exit(1)
try:
    import lpips
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from facenet_pytorch import InceptionResnetV1
    from torchvision import transforms
except ImportError:
    logging.error(
        "Please install required libraries: pip install lpips torchmetrics facenet-pytorch torchvision"
    )
    exit(1)

CALCULATED_MEAN = torch.tensor(
    [0.5174540281295776, 0.4169234037399292, 0.36359208822250366]
)
CALCULATED_STD = torch.tensor(
    [0.3002505898475647, 0.27149978280067444, 0.2665232717990875]
)


def preprocess_for_id_loss(img_tensor_01, target_size=160, device="cuda"):
    """
    Resizes and normalizes images in [0, 1] range for FaceNet input [-1, 1].
    """
    img_resized = F.interpolate(
        img_tensor_01,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    img_normalized = (img_resized * 2.0) - 1.0
    return img_normalized.to(device)


def sample_timesteps(batch_size, device, t_min=1e-5, t_max=1.0):
    """
    Sample random timesteps t for the batch, uniformly in [t_min, t_max].
    """
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    return t


def solve_ode(
    model,
    lr16,
    lr_up,
    solver="euler",
    steps=10,
    t_min=1e-6,
    t_max=1.0 - 1e-6,
    device="cuda",
):
    """
    Solves the ODE dx/dt = model(x, t, lr_up) from t_min to t_max
    using specified solver.

    Args:
        model: The neural network model predicting the velocity v(x, t, condition).
               Expected signature: model(x, t_tensor, lr_up_condition) -> velocity
        lr16: Low-resolution 16x16 condition.
        lr_up: Upscaled low-resolution condition passed to the model.
        solver: 'euler', 'heun', or 'rk4'.
        steps: Number of integration steps.
        t_min: Starting time.
        t_max: Ending time.
        device: Torch device.

    Returns:
        The final state x at t=t_max.
    """
    model.eval()
    batch_size = lr_up.size(0)
    lr_up = lr_up.to(device)

    # 1) Sample pure Gaussian noise
    x = torch.randn_like(lr_up, device=device)

    # 2) Integrate dx/dt = v(x,t, condition) from t_min to t_max
    time_steps = torch.linspace(t_min, t_max, steps + 1, device=device)
    dt = (t_max - t_min) / steps

    logging.debug(
        f"Starting ODE solve with solver={solver}, steps={steps}, dt={dt:.4f}"
    )

    iterable = (
        tqdm(range(steps), desc=f"ODE Solver ({solver})", leave=False)
        if steps > 20
        else range(steps)
    )

    with torch.no_grad():
        for i in iterable:
            t0 = time_steps[i]
            t1 = time_steps[i + 1]
            t_eval = torch.full((batch_size,), t0, device=device)

            if solver == "euler":
                # Simple Euler uses velocity at the start of the interval
                v0 = model(x, t_eval, lr_up)
                x = x + v0 * dt
            elif solver == "heun":
                # Heun's method (Predictor-Corrector)
                v0 = model(x, t_eval, lr_up)
                x_pred = x + v0 * dt
                t1_eval = torch.full((batch_size,), t1, device=device)
                v1 = model(x_pred, t1_eval, lr_up)
                x = x + 0.5 * (v0 + v1) * dt
            elif solver == "rk4":
                # Runge-Kutta 4th order method
                t_mid = torch.full((batch_size,), t0 + 0.5 * dt, device=device)
                t1_eval = torch.full((batch_size,), t1, device=device)

                k1 = model(x, t_eval, lr_up)
                k2 = model(x + 0.5 * dt * k1, t_mid, lr_up)
                k3 = model(x + 0.5 * dt * k2, t_mid, lr_up)
                k4 = model(x + dt * k3, t1_eval, lr_up)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown ODE solver: {solver}")

    logging.debug(
        f"Finished ODE solve. Final state norm: {torch.linalg.norm(x).item():.4f}"
    )
    return x


def _prepare_mean_std(tensor_like, device):
    """Prepares mean and std tensors for broadcasting."""
    mean = CALCULATED_MEAN.to(device).view(1, -1, 1, 1)
    std = CALCULATED_STD.to(device).view(1, -1, 1, 1)
    if tensor_like.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    elif tensor_like.ndim != 4 and tensor_like.ndim != 3:
        raise ValueError(
            f"Unsupported tensor ndim for denormalization: {tensor_like.ndim}"
        )
    return mean, std


def denormalize_znorm(tensor):
    """Denormalizes a Z-normalized tensor back to the [0, 1] range."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch Tensor, got {type(tensor)}")
    try:
        mean, std = _prepare_mean_std(tensor, tensor.device)
        std = torch.where(std == 0, torch.tensor(1e-6, device=std.device), std)
        tensor_denorm = tensor * std + mean
        return torch.clamp(tensor_denorm, 0.0, 1.0)
    except Exception as e:
        logging.error(f"Error during denormalization: {e}")
        logging.error(
            f"Tensor shape: {tensor.shape}, device: {tensor.device}, mean: {mean.flatten()}, std: {std.flatten()}"
        )
        return tensor


@torch.no_grad()
def calculate_psnr(pred_znorm, target_znorm, data_range=1.0):
    """
    Calculates PSNR between two Z-normalized tensors after denormalizing them.
    """
    if pred_znorm is None or target_znorm is None:
        logging.warning("PSNR calculation skipped due to None input.")
        return torch.tensor(0.0)
    pred_01 = denormalize_znorm(pred_znorm)
    target_01 = denormalize_znorm(target_znorm)

    if pred_01 is None or target_01 is None:
        logging.warning("PSNR calculation skipped due to denormalization failure.")
        return torch.tensor(0.0)

    mse = F.mse_loss(pred_01, target_01, reduction="mean")
    if mse == 0:
        return torch.tensor(float("inf"))
    mse = torch.clamp(mse, min=1e-10)
    psnr_val = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)
    return psnr_val


def save_image_grid(x_znorm, hr_znorm, filename, nrow=8):
    """
    Saves a comparison grid of generated (x) and target (hr) images.
    Handles denormalization.
    """
    if x_znorm is None or hr_znorm is None:
        logging.warning(f"Skipping grid save for {filename} due to None tensor.")
        return

    gen_01 = denormalize_znorm(x_znorm)
    tgt_01 = denormalize_znorm(hr_znorm)

    if gen_01 is None or tgt_01 is None:
        logging.warning(
            f"Skipping grid save for {filename} due to denormalization failure."
        )
        return

    # Take the first min(batch_size, nrow) samples for the grid
    num_samples = min(gen_01.size(0), hr_znorm.size(0), nrow)
    if num_samples == 0:
        logging.warning(f"Skipping grid save for {filename}, no samples available.")
        return

    gen_batch = gen_01[:num_samples].cpu()
    tgt_batch = tgt_01[:num_samples].cpu()

    # Create the grid (generate | target)
    comparison_grid = torch.cat([gen_batch, tgt_batch], 0)

    try:
        grid = vutils.make_grid(
            comparison_grid, nrow=num_samples, padding=2, normalize=False
        )
        vutils.save_image(grid, filename)
        logging.debug(f"Saved image grid to {filename}")
    except Exception as e:
        logging.error(f"Failed to save image grid {filename}: {e}")


def train(cfg):
    """Main training and validation loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using Z-Normalization with Mean: {CALCULATED_MEAN.tolist()}, Std: {CALCULATED_STD.tolist()}"
    )
    logging.info(f"Run configuration: {cfg}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    writer_path = Path("runs") / cfg.run_name
    writer_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_path))
    logging.info(f"Tensorboard logs will be saved to: {writer_path}")
    logging.info(f"Checkpoints and grids will be saved to: {cfg.out_dir}")

    try:
        ds_tr = CelebASuperRes(
            cfg.data_dir / "train", mean=CALCULATED_MEAN, std=CALCULATED_STD
        )
        ds_va = CelebASuperRes(
            cfg.data_dir / "val", mean=CALCULATED_MEAN, std=CALCULATED_STD
        )
        logging.info(f"Training dataset size: {len(ds_tr)}")
        logging.info(f"Validation dataset size: {len(ds_va)}")
    except Exception as e:
        logging.error(f"Failed to load datasets from {cfg.data_dir}: {e}")
        return

    dl = DataLoader(
        ds_tr,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )
    vdl = DataLoader(
        ds_va,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )
    model = UNetSR(depth=cfg.model_depth)
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    loss_fn_mse = F.mse_loss
    loss_fn_lpips = None
    if cfg.lambda_lpips > 0:
        logging.info(
            f"Using LPIPS perceptual loss with lambda = {cfg.lambda_lpips} (net: {cfg.lpips_net})"
        )
        loss_fn_lpips = lpips.LPIPS(net=cfg.lpips_net, verbose=False).to(device)

    id_model = None
    if cfg.lambda_id > 0:
        logging.info(
            f"Using Identity loss with lambda = {cfg.lambda_id} (model: InceptionResnetV1-VGGFace2)"
        )
        # Load pre-trained FaceNet model (InceptionResnetV1 trained on VGGFace2)
        id_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        # Freeze the parameters of the identity model
        for param in id_model.parameters():
            param.requires_grad = False
        logging.info("Identity model loaded and frozen.")

    val_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    val_lpips_metric = lpips.LPIPS(net=cfg.lpips_net, verbose=False).to(device)

    initial_best_metric = 0.0
    if cfg.best_metric == "lpips":
        initial_best_metric = float("inf")
    elif cfg.best_metric == "id":
        initial_best_metric = 0.0

    best_val_metric = initial_best_metric
    start_epoch = 0
    gstep = 0

    if cfg.resume_path and Path(cfg.resume_path).exists():
        logging.info(f"Resuming training from checkpoint: {cfg.resume_path}")
        checkpoint = torch.load(cfg.resume_path, map_location=device)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_to_load.load_state_dict(model_state_dict, strict=False)

        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        saved_best_metric_name = checkpoint.get("best_metric_name", "psnr")
        if saved_best_metric_name == cfg.best_metric:
            best_val_metric = checkpoint.get("best_val_metric", initial_best_metric)
        else:
            logging.warning(
                f"Checkpoint best metric ({saved_best_metric_name}) differs from config ({cfg.best_metric}). Resetting best_val_metric."
            )
            best_val_metric = initial_best_metric
        if "gstep" in checkpoint:
            gstep = checkpoint["gstep"]
        logging.info(
            f"Resumed from epoch {start_epoch-1}, stored best validation {saved_best_metric_name}: {checkpoint.get('best_val_metric', 'N/A'):.4f}"
        )

    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    logging.info(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_lpips = 0.0
        epoch_loss_id = 0.0

        pbar = tqdm(
            dl,
            desc=f"Train Epoch {epoch+1}/{cfg.epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for lr_up, lr16, hr in pbar:
            opt.zero_grad()

            lr_up = lr_up.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            current_bs = hr.size(0)

            # Core CFM Training Logic
            t = sample_timesteps(current_bs, device, cfg.t_min, cfg.t_max)
            noise = torch.randn_like(hr)
            t_view = t.view(-1, 1, 1, 1)
            one_minus_t_view = 1.0 - t_view
            x_t = t_view * hr + one_minus_t_view * noise
            v_target = hr - noise

            v_pred = model(x_t, t, lr_up)

            # 1. MSE Loss (on velocity)
            loss_mse = loss_fn_mse(v_pred, v_target)
            total_loss = loss_mse

            # Estimate the high-resolution prediction for perceptual and ID losses
            hr_pred_znorm = x_t + one_minus_t_view * v_pred

            # Denormalize for perceptual/ID losses
            hr_pred_01 = denormalize_znorm(hr_pred_znorm)
            hr_01 = denormalize_znorm(hr)

            # 2. LPIPS Loss
            loss_lpips_val = torch.tensor(0.0, device=device)
            if (
                loss_fn_lpips is not None
                and cfg.lambda_lpips > 0
                and hr_pred_01 is not None
                and hr_01 is not None
            ):
                loss_perceptual = loss_fn_lpips(
                    hr_pred_01.contiguous(), hr_01.contiguous()
                ).mean()
                loss_lpips_val = loss_perceptual.detach()
                total_loss = total_loss + cfg.lambda_lpips * loss_perceptual
            elif cfg.lambda_lpips > 0:
                logging.warning("Skipping LPIPS loss due to denormalization failure.")

            # 3. Identity Loss
            loss_id_val = torch.tensor(0.0, device=device)
            if (
                id_model is not None
                and cfg.lambda_id > 0
                and hr_pred_01 is not None
                and hr_01 is not None
            ):
                hr_pred_id_input = preprocess_for_id_loss(hr_pred_01, device=device)
                hr_id_input = preprocess_for_id_loss(hr_01, device=device)

                with torch.no_grad():
                    feat_gt = id_model(hr_id_input)

                feat_pred = id_model(hr_pred_id_input)

                # Calculate cosine similarity loss (1 - cosine_sim)
                loss_id = (1.0 - F.cosine_similarity(feat_pred, feat_gt, dim=1)).mean()
                loss_id_val = loss_id.detach()
                total_loss = total_loss + cfg.lambda_id * loss_id
            elif cfg.lambda_id > 0:
                logging.warning("Skipping ID loss due to denormalization failure.")

            total_loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # Log losses
            step_loss = total_loss.item()
            epoch_loss_total += step_loss * current_bs
            epoch_loss_mse += loss_mse.item() * current_bs
            epoch_loss_lpips += loss_lpips_val.item() * current_bs
            epoch_loss_id += loss_id_val.item() * current_bs

            if gstep % cfg.log_freq == 0:
                writer.add_scalar("train/step_loss_total", step_loss, gstep)
                writer.add_scalar("train/step_loss_mse", loss_mse.item(), gstep)
                if cfg.lambda_lpips > 0:
                    writer.add_scalar(
                        "train/step_loss_lpips", loss_lpips_val.item(), gstep
                    )
                if cfg.lambda_id > 0:
                    writer.add_scalar("train/step_loss_id", loss_id_val.item(), gstep)
                writer.add_scalar(
                    "train/learning_rate", opt.param_groups[0]["lr"], gstep
                )

            pbar.set_postfix(
                loss=f"{step_loss:.4f}",
                mse=f"{loss_mse.item():.4f}",
                lpips=f"{loss_lpips_val.item():.4f}",
                id=f"{loss_id_val.item():.4f}",
            )
            gstep += 1

        # End of epoch logging
        avg_epoch_loss_total = epoch_loss_total / len(ds_tr)
        avg_epoch_loss_mse = epoch_loss_mse / len(ds_tr)
        avg_epoch_loss_lpips = epoch_loss_lpips / len(ds_tr)
        avg_epoch_loss_id = epoch_loss_id / len(ds_tr)

        writer.add_scalar("train/epoch_loss_total", avg_epoch_loss_total, epoch)
        writer.add_scalar("train/epoch_loss_mse", avg_epoch_loss_mse, epoch)
        if cfg.lambda_lpips > 0:
            writer.add_scalar("train/epoch_loss_lpips", avg_epoch_loss_lpips, epoch)
        if cfg.lambda_id > 0:
            writer.add_scalar("train/epoch_loss_id", avg_epoch_loss_id, epoch)

        logging.info(
            f"Epoch {epoch+1} Train Loss: {avg_epoch_loss_total:.5f} (MSE: {avg_epoch_loss_mse:.5f}, LPIPS: {avg_epoch_loss_lpips:.5f}, ID: {avg_epoch_loss_id:.5f})"
        )

        scheduler.step()

        # Validation every cfg.val_freq epochs
        if (epoch + 1) % cfg.val_freq == 0 or epoch == cfg.epochs - 1:
            model.eval()
            val_psnr_sum = 0.0
            val_ssim_sum = 0.0
            val_lpips_sum = 0.0
            val_id_sim_sum = 0.0
            val_count = 0
            last_val_batch = None

            val_pbar = tqdm(
                vdl,
                desc=f"Validation Epoch {epoch+1}/{cfg.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            with torch.no_grad():
                for lr_up_val, lr16_val, hr_val in val_pbar:
                    lr_up_val = lr_up_val.to(device, non_blocking=True)
                    lr16_val = lr16_val.to(device, non_blocking=True)
                    hr_val = hr_val.to(device, non_blocking=True)

                    # Generate prediction
                    x_gen_znorm = solve_ode(
                        model.module if isinstance(model, nn.DataParallel) else model,
                        lr16_val,
                        lr_up_val,
                        solver=cfg.val_solver,
                        steps=cfg.val_steps,
                        t_min=cfg.t_min,
                        t_max=cfg.t_max,
                        device=device,
                    )

                    # Calculate Metrics
                    batch_psnr = calculate_psnr(x_gen_znorm, hr_val).item()

                    x_gen_01 = denormalize_znorm(x_gen_znorm)
                    hr_01 = denormalize_znorm(hr_val)

                    batch_ssim = 0.0
                    batch_lpips = 0.0
                    batch_id_sim = 0.0

                    if x_gen_01 is not None and hr_01 is not None:
                        batch_ssim = val_ssim_metric(
                            x_gen_01.contiguous(), hr_01.contiguous()
                        ).item()
                        batch_lpips = (
                            val_lpips_metric(x_gen_01.contiguous(), hr_01.contiguous())
                            .mean()
                            .item()
                        )

                        # Calculate ID Similarity
                        if id_model is not None:
                            gen_id_input = preprocess_for_id_loss(
                                x_gen_01, device=device
                            )
                            hr_id_input = preprocess_for_id_loss(hr_01, device=device)
                            feat_gen = id_model(gen_id_input)
                            feat_hr = id_model(hr_id_input)
                            batch_id_sim = (
                                F.cosine_similarity(feat_gen, feat_hr, dim=1)
                                .mean()
                                .item()
                            )

                    else:
                        logging.warning(
                            "Skipping SSIM/LPIPS/ID calculation due to denormalization failure."
                        )

                    # Accumulate sums safely
                    if math.isfinite(batch_psnr):
                        val_psnr_sum += batch_psnr * hr_val.size(0)
                    if math.isfinite(batch_ssim):
                        val_ssim_sum += batch_ssim * hr_val.size(0)
                    if math.isfinite(batch_lpips):
                        val_lpips_sum += batch_lpips * hr_val.size(0)
                    if math.isfinite(batch_id_sim):
                        val_id_sim_sum += batch_id_sim * hr_val.size(0)
                    val_count += hr_val.size(0)

                    val_pbar.set_postfix(
                        psnr=f"{batch_psnr:.2f}",
                        ssim=f"{batch_ssim:.3f}",
                        lpips=f"{batch_lpips:.3f}",
                        id=f"{batch_id_sim:.3f}",
                    )

                    if cfg.save_val_grid or (epoch + 1) % cfg.save_hires_freq == 0:
                        last_val_batch = {
                            "lr16": lr16_val.cpu(),
                            "lr_up": lr_up_val.cpu(),
                            "hr": hr_val.cpu(),
                            "gen": x_gen_znorm.cpu(),
                        }

            # Log Validation Metrics
            if val_count > 0:
                avg_val_psnr = val_psnr_sum / val_count
                avg_val_ssim = val_ssim_sum / val_count
                avg_val_lpips = val_lpips_sum / val_count
                avg_val_id_sim = val_id_sim_sum / val_count

                writer.add_scalar("val/psnr", avg_val_psnr, epoch)
                writer.add_scalar("val/ssim", avg_val_ssim, epoch)
                writer.add_scalar("val/lpips", avg_val_lpips, epoch)
                writer.add_scalar("val/id_sim", avg_val_id_sim, epoch)
                logging.info(
                    f"Epoch {epoch+1} Validation PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}, LPIPS: {avg_val_lpips:.4f}, ID Sim: {avg_val_id_sim:.4f}"
                )

                current_metric = 0.0
                is_better = False
                metric_name = cfg.best_metric
                if metric_name == "psnr":
                    current_metric = avg_val_psnr
                    is_better = current_metric > best_val_metric
                elif metric_name == "ssim":
                    current_metric = avg_val_ssim
                    is_better = current_metric > best_val_metric
                elif metric_name == "lpips":
                    current_metric = avg_val_lpips
                    is_better = current_metric < best_val_metric
                elif metric_name == "id":
                    current_metric = avg_val_id_sim
                    is_better = current_metric > best_val_metric
                else:
                    logging.warning(
                        f"Invalid best_metric '{metric_name}', defaulting to PSNR."
                    )
                    metric_name = "psnr"
                    current_metric = avg_val_psnr
                    is_better = current_metric > best_val_metric

                # Save best model if current metric is better
                if is_better:
                    best_val_metric = current_metric
                    logging.info(
                        f"New best model found with {metric_name}: {best_val_metric:.4f}"
                    )
                    save_path_best = cfg.out_dir / "best_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": (
                                model.module.state_dict()
                                if isinstance(model, nn.DataParallel)
                                else model.state_dict()
                            ),
                            "optimizer_state_dict": opt.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_metric": best_val_metric,
                            "best_metric_name": metric_name,
                            "gstep": gstep,
                            "cfg": cfg,
                        },
                        save_path_best,
                    )
                    logging.info(f"Best model saved to {save_path_best}")

                if (epoch + 1) % cfg.save_latest_freq == 0 or epoch == cfg.epochs - 1:
                    save_path_latest = cfg.out_dir / "latest_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": (
                                model.module.state_dict()
                                if isinstance(model, nn.DataParallel)
                                else model.state_dict()
                            ),
                            "optimizer_state_dict": opt.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_metric": best_val_metric,
                            "best_metric_name": metric_name,
                            "current_val_psnr": avg_val_psnr,
                            "current_val_ssim": avg_val_ssim,
                            "current_val_lpips": avg_val_lpips,
                            "current_val_id_sim": avg_val_id_sim,
                            "gstep": gstep,
                            "cfg": cfg,
                        },
                        save_path_latest,
                    )
                    logging.info(f"Latest model checkpoint saved to {save_path_latest}")

            else:
                logging.warning(
                    f"Epoch {epoch+1} Validation: No valid metrics calculated (val_count=0)."
                )
            if last_val_batch is not None and cfg.save_val_grid:
                grid_filename = cfg.out_dir / f"validation_grid_ep{epoch+1:04d}.png"
                save_image_grid(
                    last_val_batch["gen"],
                    last_val_batch["hr"],
                    grid_filename,
                    nrow=cfg.grid_rows,
                )

            # High-resolution sample generation
            if last_val_batch is not None and (epoch + 1) % cfg.save_hires_freq == 0:
                logging.info(
                    f"Generating high-resolution sample for epoch {epoch+1}..."
                )
                x_hires_znorm = solve_ode(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    last_val_batch["lr16"].to(device),
                    last_val_batch["lr_up"].to(device),
                    solver=cfg.hires_solver,
                    steps=cfg.hires_steps,
                    t_min=cfg.t_min,
                    t_max=cfg.t_max,
                    device=device,
                )
                hires_filename = cfg.out_dir / f"high_res_ep{epoch+1:04d}.png"
                save_image_grid(
                    x_hires_znorm.cpu(),
                    last_val_batch["hr"],
                    hires_filename,
                    nrow=cfg.grid_rows,
                )
                logging.info(f"High-resolution sample saved to {hires_filename}")

    writer.close()
    logging.info("Training finished.")


# Argument Parser
def parse():
    p = argparse.ArgumentParser(
        description="Train Conditional Flow Matching (CFM) U-Net for Face SR with Perceptual and ID Loss"
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory of dataset (containing train/val subdirs)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("checkpoints_cfm_sr_fixed_no_id_no_lpips"),
        help="Directory to save checkpoints and generated images",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default="cfm_face_sr_lpips_id_aug_btlnek",
        help="Name for TensorBoard run directory",
    )
    p.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Training Parameters
    p.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size per GPU (Reduced further due to ID model memory usage)",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    p.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW"
    )
    p.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )
    p.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader workers"
    )

    # Loss Parameters
    p.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.0,
        help="Weight for LPIPS perceptual loss (Set > 0 to enable)",
    )
    p.add_argument(
        "--lpips_net",
        type=str,
        default="vgg",
        choices=["alex", "vgg"],
        help="Network backbone for LPIPS",
    )
    p.add_argument(
        "--lambda_id",
        type=float,
        default=0.0,
        help="Weight for Identity loss (Set > 0 to enable)",
    )

    # CFM Parameters
    p.add_argument(
        "--t_min",
        type=float,
        default=1e-6,
        help="Minimum time t for sampling/training ODE path",
    )
    p.add_argument(
        "--t_max",
        type=float,
        default=1.0 - 1e-6,
        help="Maximum time t for sampling/training ODE path",
    )

    # Sampling Parameters
    p.add_argument(
        "--val_steps",
        type=int,
        default=20,
        help="Number of steps for validation sampling (Adjusted default)",
    )
    p.add_argument(
        "--val_solver",
        type=str,
        default="euler",
        choices=["euler", "heun", "rk4"],
        help="ODE solver for validation sampling",
    )
    p.add_argument(
        "--val_freq",
        type=int,
        default=100,
        help="Frequency (in epochs) to run validation",
    )
    p.add_argument(
        "--best_metric",
        type=str,
        default="lpips",
        choices=["psnr", "ssim", "lpips", "id"],
        help="Metric for saving best model",
    )

    # High-Resolution Sampling Parameters
    p.add_argument(
        "--save_hires_freq",
        type=int,
        default=25,
        help="Frequency (in epochs) to save high-resolution samples",
    )
    p.add_argument(
        "--hires_steps",
        type=int,
        default=100,
        help="Number of steps for high-resolution sampling",
    )
    p.add_argument(
        "--hires_solver",
        type=str,
        default="euler",
        choices=["euler", "heun", "rk4"],
        help="ODE solver for high-resolution sampling",
    )

    # Logging and Checkpointing Parameters
    p.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Frequency (in steps) to log training loss",
    )
    p.add_argument(
        "--save_latest_freq",
        type=int,
        default=5,
        help="Frequency (in epochs) to save the latest checkpoint",
    )
    p.add_argument(
        "--save_val_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save image grid during validation",
    )
    p.add_argument(
        "--grid_rows",
        type=int,
        default=8,
        help="Number of samples per row in image grids",
    )

    # Model Parameters
    p.add_argument(
        "--model_depth", type=int, default=2, help="Depth of the U-Net model (Example)"
    )

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse()
    train(cfg)
