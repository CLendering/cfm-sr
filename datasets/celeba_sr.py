import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import (
    resize,
    InterpolationMode,
)
from torchvision.datasets.folder import default_loader
from pathlib import Path


CALCULATED_MEAN = torch.tensor(
    [0.5174540281295776, 0.4169234037399292, 0.36359208822250366]
)
CALCULATED_STD = torch.tensor(
    [0.3002505898475647, 0.27149978280067444, 0.2665232717990875]
)


class CelebASuperRes(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        mean=CALCULATED_MEAN,
        std=CALCULATED_STD,
        use_augmentation=False,
    ):
        self.root = Path(root)
        self.paths = list(self.root.glob("*.png"))

        self.mean = mean
        self.std = std
        self.use_augmentation = use_augmentation
        self.augment_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=(-5, 5), interpolation=InterpolationMode.BILINEAR
                        )
                    ],
                    p=0.3,
                ),
            ]
        )
        self.hr_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.lr_resize = transforms.Resize(
            16, interpolation=InterpolationMode.BICUBIC, antialias=True
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = default_loader(self.paths[idx])
        if self.use_augmentation:
            img = self.augment_tf(img)
        hr = self.hr_tf(img)
        lr16 = self.lr_resize(hr)
        lr_up = F.interpolate(
            lr16.unsqueeze(0),
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        lr_up = lr_up.squeeze(0)
        return lr_up, lr16, hr
