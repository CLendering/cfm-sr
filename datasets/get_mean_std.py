import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import time


class CalculationDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(list(Path(root).glob("*.png")))
        if not self.paths:
            raise FileNotFoundError(f"No PNG files found in directory: {root}")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = default_loader(self.paths[idx])
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {self.paths[idx]}: {e}")
            try:
                sample_img = self.transform(default_loader(self.paths[0]))
                return torch.zeros_like(sample_img)
            except:
                return torch.zeros(3, 128, 128)


def calculate_mean_std(loader: DataLoader):
    """
    Calculates the mean and standard deviation of a dataset represented by a DataLoader.
    Assumes images are tensors in [0, 1] range with shape [B, C, H, W].
    """
    n_channels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    n_total_pixels = 0

    start_time = time.time()
    num_batches = len(loader)

    for i, images in enumerate(
        tqdm(loader, desc="Calculating Mean/Std", total=num_batches)
    ):
        if i == 0:
            n_channels = images.shape[1]
            sum_pixels = torch.zeros(n_channels, device=images.device)
            sum_sq_pixels = torch.zeros(n_channels, device=images.device)
        images = images.to(sum_pixels.device)
        batch_size, _, height, width = images.shape
        num_pixels_in_batch_per_channel = batch_size * height * width
        sum_pixels += torch.sum(images, dim=[0, 2, 3])
        sum_sq_pixels += torch.sum(images**2, dim=[0, 2, 3])
        n_total_pixels += num_pixels_in_batch_per_channel
    if n_total_pixels == 0:
        print("Warning: No pixels processed. Returning default values.")
        return torch.zeros(n_channels), torch.ones(n_channels)
    mean = sum_pixels / n_total_pixels
    mean_sq = sum_sq_pixels / n_total_pixels
    variance = mean_sq - mean**2
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance + 1e-7)
    end_time = time.time()
    print(f"Calculation took {end_time - start_time:.2f} seconds.")
    return mean.cpu(), std.cpu()


if __name__ == "__main__":
    train_data_path = "data/splits/train"

    if not Path(train_data_path).exists() or not any(
        Path(train_data_path).glob("*.png")
    ):
        print(f"Error: Path '{train_data_path}' not found or contains no PNG files.")
        celeba_hq_mean = torch.tensor([0.5, 0.5, 0.5])
        celeba_hq_std = torch.tensor([0.5, 0.5, 0.5])
        print("\nWARNING: Using placeholder values for mean and std.")
    else:
        print(f"Calculating mean and std for dataset at: {train_data_path}")
        try:
            calc_dataset = CalculationDataset(train_data_path)
            print(f"Found {len(calc_dataset)} images.")
            calc_loader = DataLoader(
                calc_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
            celeba_hq_mean, celeba_hq_std = calculate_mean_std(calc_loader)
        except Exception as e:
            print(f"\nAn error occurred during calculation: {e}")
            celeba_hq_mean = torch.tensor([0.5, 0.5, 0.5])
            celeba_hq_std = torch.tensor([0.5, 0.5, 0.5])

    print(f"\nCalculated Mean: {celeba_hq_mean.tolist()}")
    print(f"Calculated Std:  {celeba_hq_std.tolist()}")
