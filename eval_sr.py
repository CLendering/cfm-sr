import argparse, math, csv, random, os
from pathlib import Path
from typing import List

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from pytorch_fid import fid_score
from facenet_pytorch import InceptionResnetV1

from models.u_net_flow import UNetSR
from datasets.celeba_sr import CelebASuperRes
from train_sr import solve_ode

CALCULATED_MEAN = torch.tensor(
    [0.5174540281295776, 0.4169234037399292, 0.36359208822250366]
)
CALCULATED_STD = torch.tensor(
    [0.3002505898475647, 0.27149978280067444, 0.2665232717990875]
)


def _prepare_mean_std(tensor_like, device):
    """Prepares mean/std tensors for broadcasting."""
    mean = CALCULATED_MEAN.to(device).view(1, -1, 1, 1)
    std = CALCULATED_STD.to(device).view(1, -1, 1, 1)
    if tensor_like.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    elif tensor_like.ndim == 0:
        pass
    elif tensor_like.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Unsupported tensor ndim for _prepare_mean_std: {tensor_like.ndim}"
        )

    return mean, std


def denormalize_znorm(tensor_znorm):
    """Denormalizes a Z-normalized tensor back to the [0, 1] range."""
    if tensor_znorm is None:
        return None
    device = tensor_znorm.device
    mean, std = _prepare_mean_std(tensor_znorm, device)
    std = torch.where(std == 0, torch.tensor(1e-6, device=device), std)
    tensor_01 = tensor_znorm * std + mean
    return tensor_01.clamp(0.0, 1.0)


def renormalize_for_lpips(tensor_01):
    """renormalizes a tensor from [0, 1] range to [-1, 1] range"""
    return tensor_01 * 2.0 - 1.0


def psnr(pred_znorm, target_znorm):
    """Calculates PSNR between two Z-normalized tensors."""
    pred_01 = denormalize_znorm(pred_znorm)
    target_01 = denormalize_znorm(target_znorm)
    mse = F.mse_loss(pred_01, target_01).item()
    if mse == 0:
        return float("inf")
    max_pixel_value = 1.0
    return 20 * math.log10(max_pixel_value) - 10 * math.log10(mse)


@torch.no_grad()
def evaluate(
    model,
    dl: DataLoader,
    device,
    steps: int,
    out_dir: Path,
    max_batches: int = None,
    use_heun: bool = False,
):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {out_dir}")
    print(f"Using {'Heun' if use_heun else 'Euler'} sampler with {steps} steps.")

    lpips_alex = LPIPS(net="alex").to(device).eval()
    lpips_vgg = LPIPS(net="vgg").to(device).eval()
    ssim_metric = SSIM(data_range=1.0).to(device)
    id_net = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    per_img = []
    sampler_fn = solve_ode

    print("Calculating per-image metrics (PSNR, SSIM, LPIPS, ID Cos)...")
    for b, batch_data in enumerate(tqdm(dl, desc="Eval Metrics")):
        if max_batches and b >= max_batches:
            print(f"Stopping after {max_batches} batches.")
            break

        # Data from DataLoader is already Z-normalized
        lr_up, lr16, hr = batch_data
        lr_up, lr16, hr = lr_up.to(device), lr16.to(device), hr.to(device)

        sr = sampler_fn(model, lr16, lr_up, steps=steps)
        for i in range(sr.size(0)):
            pr_znorm = sr[i : i + 1]
            gt_znorm = hr[i : i + 1]
            pr_01 = denormalize_znorm(pr_znorm)
            gt_01 = denormalize_znorm(gt_znorm)
            p = psnr(pr_znorm, gt_znorm)
            s = ssim_metric(pr_01, gt_01).item()
            pr_lpips = renormalize_for_lpips(pr_01)
            gt_lpips = renormalize_for_lpips(gt_01)
            l_a = lpips_alex(pr_lpips, gt_lpips).item()
            l_v = lpips_vgg(pr_lpips, gt_lpips).item()
            id_gt_embed = id_net(gt_01)
            id_pr_embed = id_net(pr_01)
            id_sim = F.cosine_similarity(id_gt_embed, id_pr_embed).mean().item()

            per_img.append(
                dict(
                    file_idx=b * dl.batch_size + i,
                    psnr=p,
                    ssim=s,
                    lpips_alex=l_a,
                    lpips_vgg=l_v,
                    id_cos=id_sim,
                )
            )
        if b == 0:
            print("Saving visual comparison grid for the first batch...")
            grid = vutils.make_grid(
                torch.cat([denormalize_znorm(sr), denormalize_znorm(hr)], 0)[:16],
                nrow=8,
                normalize=False,
            )
            vutils.save_image(grid, out_dir / "grid_first_batch_comparison.png")
            print(f"Saved grid to {out_dir / 'grid_first_batch_comparison.png'}")
    if not per_img:
        print("Warning: No images processed. Skipping metric aggregation and FID.")
        return

    print("Aggregating metrics...")
    df = pd.DataFrame(per_img)
    df.to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Saved per-image metrics to {out_dir / 'per_image_metrics.csv'}")
    means = df.drop(columns=["file_idx"]).mean().to_dict()
    print("\n===> Aggregate Metrics (Mean)")
    for k, v in means.items():
        print(f"{k:<12s}: {v:>8.4f}")

    # FID calculation
    print("\nPreparing images for FID calculation...")
    gen_dir, gt_dir = out_dir / "fid_fake", out_dir / "fid_real"
    gen_dir.mkdir(exist_ok=True), gt_dir.mkdir(exist_ok=True)

    def dump_for_fid(imgs_znorm: torch.Tensor, root: Path, offset: int):
        imgs_01 = denormalize_znorm(imgs_znorm).clamp(0.0, 1.0)
        for j, im_01 in enumerate(imgs_01):  # Iterate through batch
            vutils.save_image(im_01, root / f"{offset + j:06d}.png")

    fid_image_limit = 5000
    idx = 0
    print(f"Dumping up to {fid_image_limit} images for FID...")
    fid_pbar = tqdm(
        dl,
        desc="Dump FID Images",
        total=min(len(dl), (fid_image_limit + dl.batch_size - 1) // dl.batch_size),
    )
    for batch_data in fid_pbar:
        if idx >= fid_image_limit:
            break
        lr_up, lr16, hr_znorm = batch_data
        lr_up, lr16 = lr_up.to(device), lr16.to(device)
        hr_znorm = hr_znorm.to(device)
        sr_znorm = sampler_fn(model, lr16, lr_up, steps=steps)
        num_to_save = min(sr_znorm.size(0), fid_image_limit - idx)

        # Dump generated (fake) and ground truth (real) images
        dump_for_fid(sr_znorm[:num_to_save], gen_dir, idx)
        dump_for_fid(hr_znorm[:num_to_save], gt_dir, idx)

        idx += num_to_save
        fid_pbar.set_postfix(images_dumped=idx)

    print(f"\nFinished dumping {idx} images.")
    print("Calculating FID score...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [str(gt_dir), str(gen_dir)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        print(f"FID: {fid_value:.2f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_value = float("nan")

    # Summary markdown file
    print("Writing summary markdown file...")
    md = out_dir / "summary.md"
    with md.open("w") as f:
        f.write(
            f"# Evaluation Summary ({'Heun' if use_heun else 'Euler'}, {steps} steps)\n\n"
        )
        f.write("## Mean Metrics\n")
        for k, v in means.items():
            f.write(f"- **{k}**: `{v:.4f}`\n")
        f.write("\n## FID Score\n")
        f.write(f"- **FID**: `{fid_value:.2f}`\n")
    print("Generating metric pairplot...")
    try:
        plot_cols = [
            col for col in ["psnr", "ssim", "lpips_alex", "id_cos"] if col in df.columns
        ]
        if plot_cols:
            sns.pairplot(df[plot_cols])
            plt.suptitle("Metric Correlations", y=1.02)
            plt.savefig(out_dir / "metric_pairplot.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved pairplot to {out_dir / 'metric_pairplot.png'}")
        else:
            print("Skipping pairplot: No suitable metric columns found.")
    except Exception as e:
        print(f"Error generating pairplot: {e}")

    # Worst/Best examples
    print("Generating worst/best LPIPS examples...")
    metric_to_sort = "lpips_vgg"
    if metric_to_sort not in df.columns:
        print(f"Skipping worst/best examples: '{metric_to_sort}' not found in metrics.")
        return

    def save_extremes(metric: str, n: int = 4):
        if df.empty:
            print(f"Skipping extremes for {metric}: DataFrame is empty.")
            return
        if metric not in df.columns:
            print(f"Skipping extremes: Metric '{metric}' not found.")
            return
        print(f"Finding {n} worst and best examples based on {metric}...")
        if "file_idx" not in df.columns:
            print(
                "Error: 'file_idx' column missing in DataFrame. Cannot retrieve examples."
            )
            return

        try:
            n_samples = len(df)
            if n > n_samples:
                print(
                    f"Warning: Requested {n} examples, but only {n_samples} available. Using {n_samples}."
                )
                n = n_samples
            if n == 0:
                print("Skipping extremes: n=0.")
                return

            low_df = df.nsmallest(n, metric)
            high_df = df.nlargest(n, metric)

            low_indices = low_df.file_idx.tolist()
            high_indices = high_df.file_idx.tolist()

            print(f"Worst {metric} indices: {low_indices}")
            print(f"Best {metric} indices: {high_indices}")

            cases = {"worst": low_indices, "best": high_indices}

            for tag, idxs in cases.items():
                if not idxs:
                    continue
                batch_lr_up, batch_lr16, batch_hr = [], [], []
                print(f"Retrieving data for {tag} {metric} examples...")
                for file_idx in idxs:
                    try:
                        lr_up_i, lr16_i, hr_i = dl.dataset[file_idx]
                        batch_lr_up.append(lr_up_i)
                        batch_lr16.append(lr16_i)
                        batch_hr.append(hr_i)
                    except IndexError:
                        print(
                            f"Warning: Index {file_idx} out of bounds for dataset. Skipping."
                        )
                    except Exception as e:
                        print(
                            f"Error retrieving data for index {file_idx}: {e}. Skipping."
                        )

                if not batch_hr:
                    print(
                        f"Could not retrieve data for {tag} {metric}. Skipping grid save."
                    )
                    continue
                # Stack the batches
                hrs_znorm = torch.stack(batch_hr).to(device)
                lru_znorm = torch.stack(batch_lr_up).to(device)
                lr16_znorm = torch.stack(batch_lr16).to(device)
                print(f"Generating SR for {tag} {metric} examples...")
                srs_znorm = sampler_fn(model, lr16_znorm, lru_znorm, steps=steps)

                # Denormalize for visualization
                srs_01 = denormalize_znorm(srs_znorm)
                hrs_01 = denormalize_znorm(hrs_znorm)
                lru_01 = denormalize_znorm(lru_znorm)

                # Save grid (Generated SRs on top, Ground Truth HRs below)
                grid = vutils.make_grid(
                    torch.cat([srs_01, hrs_01, lru_01], 0),
                    normalize=False,
                )
                filename = out_dir / f"{tag}_{metric}_examples.png"
                vutils.save_image(grid, filename)
                print(f"Saved {tag} {metric} examples grid to {filename}")

        except KeyError as e:
            print(f"Error accessing DataFrame column for sorting extremes: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in save_extremes: {e}")

    save_extremes(metric_to_sort)
    print("Finished generating worst/best examples.")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate CFM U-Net SR with Z-Normalization"
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory of CelebA data (containing train/val)",
    )
    p.add_argument(
        "--ckpt", type=Path, required=True, help="Path to the model checkpoint (.pt)"
    )
    p.add_argument(
        "--steps", type=int, default=25, help="Number of steps for reverse sampling"
    )
    p.add_argument(
        "--bs",
        type=int,
        default=256,
        help="Batch size for evaluation (adjust based on VRAM)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("eval_out_znorm_plain"),
        help="Directory to save evaluation results",
    )
    p.add_argument("--cuda", action="store_true", help="Force use CUDA if available")
    p.add_argument("--no-cuda", dest="cuda", action="store_false", help="Force use CPU")
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit evaluation to first N batches (for debugging)",
    )
    p.add_argument(
        "--use_heun", action="store_true", help="Use Heun sampler instead of Euler"
    )
    p.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader workers"
    )

    p.set_defaults(cuda=torch.cuda.is_available())
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading validation dataset from: {args.data_dir / 'val'}")
    print(f"Using Z-Normalization Mean: {CALCULATED_MEAN.tolist()}")
    print(f"Using Z-Normalization Std:  {CALCULATED_STD.tolist()}")
    try:
        val_ds = CelebASuperRes(
            args.data_dir / "test", mean=CALCULATED_MEAN, std=CALCULATED_STD
        )
        if len(val_ds) == 0:
            print(
                f"Error: No data found in validation dataset at {args.data_dir / 'val'}. Please check the path."
            )
            return
        print(f"Validation dataset size: {len(val_ds)}")
    except FileNotFoundError:
        print(f"Error: Validation data directory not found at {args.data_dir / 'val'}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dl = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Loading model checkpoint from: {args.ckpt}")
    try:
        model = UNetSR(depth=2)
        model = model.to(device)

        state = torch.load(args.ckpt, map_location=device)["model_state_dict"]
        if isinstance(state, dict) and next(iter(state)).startswith("module."):
            print("Adjusting keys from DataParallel checkpoint...")
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state.items():
                name = k[7:]  # remove `module.` prefix
                new_state_dict[name] = v
            state = new_state_dict
        if isinstance(state, dict):
            model.load_state_dict(state, strict=True)
        else:
            print(
                "Warning: Checkpoint appears to be the full model object, not just state_dict."
            )
            if hasattr(state, "state_dict"):
                model.load_state_dict(state.state_dict(), strict=True)
            else:
                print("Error: Cannot load state_dict from the loaded checkpoint file.")
                return

        model.eval()
        model = torch.nn.DataParallel(model)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.ckpt}")
        return
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print(
            "This often means the model architecture in eval.py doesn't match the saved checkpoint."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return
    print("\nStarting evaluation...")
    evaluate(
        model=model,
        dl=dl,
        device=device,
        steps=args.steps,
        out_dir=args.out_dir,
        max_batches=args.max_batches,
        use_heun=args.use_heun,
    )
    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
