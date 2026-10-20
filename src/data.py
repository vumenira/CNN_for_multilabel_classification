import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image


def channel_stats(paths) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of pixels in [0, 1] over all given images."""
    total, total_sq, n = np.zeros(3), np.zeros(3), 0
    for path in paths:
        px = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255
        total += px.sum(0)
        total_sq += (px ** 2).sum(0)
        n += len(px)
    mean = total / n
    return mean, np.sqrt(total_sq / n - mean ** 2)


def eval_transforms(size: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def train_transforms(size: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(1.0, 1.4)),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_by_max=False, p=0.5),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(50, 95)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.Downscale(scale_range=(0.5, 0.75)),
                    A.GaussNoise(std_range=(0.02, 0.05)),
                ],
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
