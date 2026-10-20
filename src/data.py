from pathlib import Path

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import DATA_DIR, FOLDS_PATH, LABELS


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


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose, img_dir: Path):
        self.paths = [img_dir / f"{i}.jpg" for i in df.image_id]
        self.ids = df.image_id.tolist()
        has_labels = any(c in df.columns for c in LABELS)  
        self.labels = torch.from_numpy(df[LABELS].to_numpy(np.float32)) if has_labels else torch.zeros((len(df), len(LABELS)))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        image = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        return self.transform(image=image)["image"], self.labels[i], self.ids[i]


def seed_worker(_worker_id: int):
    """Workers get pickled copies of the transform with the same RNG state; reseed each one."""
    info = torch.utils.data.get_worker_info()
    assert info is not None and isinstance(info.dataset, ImageDataset)
    info.dataset.transform.set_random_seed(info.seed % 2**32)


class DataModule(L.LightningDataModule):
    def __init__(self, mean, std, fold: int = 0, size: int = 224, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.mean, self.std = mean, std
        self.fold, self.size = fold, size
        self.batch_size, self.num_workers = batch_size, num_workers

    def setup(self, stage: str):
        eval_tf = eval_transforms(self.size, self.mean, self.std)
        if stage in ("fit", "validate"):
            folds = pd.read_csv(FOLDS_PATH)
            is_val = folds.fold == self.fold
            train_tf = train_transforms(self.size, self.mean, self.std)
            self.train_ds = ImageDataset(folds[~is_val], train_tf, DATA_DIR / "train")
            self.val_ds = ImageDataset(folds[is_val], eval_tf, DATA_DIR / "train")
        if stage == "predict":
            self.test_ds = ImageDataset(pd.read_csv(DATA_DIR / "test.csv"), eval_tf, DATA_DIR / "test")

    def _loader(self, ds: Dataset, train: bool = False) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=train, drop_last=train,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          worker_init_fn=seed_worker)

    def train_dataloader(self):
        return self._loader(self.train_ds, train=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def predict_dataloader(self):
        return self._loader(self.test_ds)