import json
import os
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import DATA_DIR, FOLDS_PATH, LABELS

CACHE_DIR = DATA_DIR / "cache"


def channel_stats(paths) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of pixels in [0, 1] over all given images."""
    total, total_sq, n = np.zeros(3), np.zeros(3), 0
    for path in paths:
        px = (
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).reshape(-1, 3)
            / 255
        )
        total += px.sum(0)
        total_sq += (px**2).sum(0)
        n += len(px)
    mean = total / n
    return mean, np.sqrt(total_sq / n - mean**2)


def _load_resized(args) -> np.ndarray:
    path, size = args
    image = np.asarray(Image.open(path).convert("RGB"))
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def cache_path(split: str, size: int) -> Path:
    return CACHE_DIR / f"{split}_{size}.npy"


def split_ids(split: str) -> list[str]:
    return pd.read_csv(DATA_DIR / f"{split}.csv").image_id.tolist()


def build_cache(split: str, size: int, workers: int | None = None) -> Path:
    """Decode and resize every image of a split once into a uint8 .npy of shape (N, size, size, 3).

    Rows follow the order of <split>.csv. The resize is cv2 INTER_LINEAR, the same as A.Resize,
    so cached eval images match the uncached eval pipeline.
    """
    path = cache_path(split, size)
    if path.exists():
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = split_ids(split)
    tmp = path.with_name(path.name + ".tmp")
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(len(ids), size, size, 3)
    )
    jobs = ((DATA_DIR / split / f"{i}.jpg", size) for i in ids)
    with Pool(workers or os.cpu_count()) as pool:
        for row, image in enumerate(
            tqdm(pool.imap(_load_resized, jobs, chunksize=64), total=len(ids), desc=f"cache {split}")
        ):
            out[row] = image
    out.flush()
    del out
    os.replace(tmp, path)
    return path


def fold_stats(fold: int, size: int) -> tuple[list[float], list[float]]:
    """Mean/std of the training part of `fold`, computed from the cache and stored next to it."""
    path = CACHE_DIR / f"stats_{size}_fold{fold}.json"
    if path.exists():
        stats = json.loads(path.read_text())
        return stats["mean"], stats["std"]
    images = np.load(build_cache("train", size), mmap_mode="r")
    row_of = {i: r for r, i in enumerate(split_ids("train"))}
    folds = pd.read_csv(FOLDS_PATH)
    rows = sorted(row_of[i] for i in folds[folds.fold != fold].image_id)
    total, total_sq, n = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(rows), 256):
        chunk = np.asarray(images[rows[start : start + 256]], dtype=np.float32) / 255
        total += chunk.sum(axis=(0, 1, 2), dtype=np.float64)
        total_sq += (chunk**2).sum(axis=(0, 1, 2), dtype=np.float64)
        n += chunk.shape[0] * size * size
    mean = total / n
    std = np.sqrt(total_sq / n - mean**2)
    stats = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    path.write_text(json.dumps(stats))
    return stats["mean"], stats["std"]


def class_pos_weights(fold: int, mode: str | None) -> list[float] | None:
    """BCE pos_weight from the training part of `fold`: None, sqrt(neg/pos) or neg/pos per class."""
    if mode in (None, "none"):
        return None
    folds = pd.read_csv(FOLDS_PATH)
    y = folds[folds.fold != fold][LABELS]
    ratio = (len(y) - y.sum()) / y.sum()
    if mode == "sqrt":
        ratio = np.sqrt(ratio)
    elif mode != "full":
        raise ValueError(f"Unknown pos_weight mode: {mode}")
    return ratio.round(3).tolist()


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
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        img_dir: Path,
        cache: Path | None = None,
    ):
        """With `cache` (a build_cache file of the same split) images come from it instead of JPEGs."""
        self.paths = [img_dir / f"{i}.jpg" for i in df.image_id]
        self.ids = df.image_id.tolist()
        has_labels = any(c in df.columns for c in LABELS)
        self.labels = (
            torch.from_numpy(df[LABELS].to_numpy(np.float32))
            if has_labels
            else torch.zeros((len(df), len(LABELS)))
        )
        self.transform = transform
        self.cache = cache
        self.images = None  # opened lazily so every worker maps the file itself instead of pickling it
        if cache is not None:
            row_of = {i: r for r, i in enumerate(split_ids(img_dir.name))}
            self.rows = [row_of[i] for i in self.ids]

    def __len__(self):
        return len(self.paths)

    def read(self, i: int) -> np.ndarray:
        if self.cache is None:
            return np.asarray(Image.open(self.paths[i]).convert("RGB"))
        if self.images is None:
            self.images = np.load(self.cache, mmap_mode="r")
        return np.array(self.images[self.rows[i]])

    def __getitem__(self, i: int):
        return self.transform(image=self.read(i))["image"], self.labels[i], self.ids[i]


def seed_worker(_worker_id: int):
    """Workers get pickled copies of the transform with the same RNG state; reseed each one."""
    info = torch.utils.data.get_worker_info()
    assert info is not None and isinstance(info.dataset, ImageDataset)
    info.dataset.transform.set_random_seed(info.seed % 2**32)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        mean,
        std,
        fold: int = 0,
        size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        cache: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mean, self.std = mean, std
        self.fold, self.size = fold, size
        self.batch_size, self.num_workers = batch_size, num_workers
        self.cache = cache

    def setup(self, stage: str):
        eval_tf = eval_transforms(self.size, self.mean, self.std)
        if stage in ("fit", "validate"):
            folds = pd.read_csv(FOLDS_PATH)
            is_val = folds.fold == self.fold
            if not is_val.any():
                raise ValueError(f"Fold {self.fold} is absent from {FOLDS_PATH}")
            train_tf = train_transforms(self.size, self.mean, self.std)
            train_tf.set_random_seed(torch.initial_seed() % 2**32)
            cache = build_cache("train", self.size) if self.cache else None
            self.train_ds = ImageDataset(folds[~is_val], train_tf, DATA_DIR / "train", cache)
            self.val_ds = ImageDataset(folds[is_val], eval_tf, DATA_DIR / "train", cache)
        if stage == "predict":
            cache = build_cache("test", self.size) if self.cache else None
            self.test_ds = ImageDataset(
                pd.read_csv(DATA_DIR / "test.csv"), eval_tf, DATA_DIR / "test", cache
            )

    def loader(self, ds: Dataset, train: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
        )

    def train_dataloader(self):
        return self.loader(self.train_ds, train=True)

    def val_dataloader(self):
        return self.loader(self.val_ds)

    def predict_dataloader(self):
        return self.loader(self.test_ds)
