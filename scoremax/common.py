import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "results" / "scoremax"))
CACHE = Path(os.environ.get("CACHE_DIR", DATA / "cache"))
FOLDS = ROOT / "data" / "folds.csv"
H, W = 375, 450


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(dev):
    if dev.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def ids(split):
    name = "folds.csv" if split == "train" else "test.csv"
    path = FOLDS if split == "train" else DATA / name
    return pd.read_csv(path).image_id.tolist()


def _decode(path):
    image = Image.open(path).convert("RGB")
    if image.size != (W, H):
        image = image.resize((W, H), Image.BILINEAR)
    return np.asarray(image)


def build_cache(split, workers=None):
    path = CACHE / f"{split}_native.npy"
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    names = ids(split)
    tmp = path.with_name(path.name + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(names), H, W, 3))
    with Pool(workers or os.cpu_count()) as pool:
        for row, image in enumerate(pool.imap(_decode, [DATA / split / f"{i}.jpg" for i in names], chunksize=64)):
            out[row] = image
    out.flush()
    del out
    os.replace(tmp, path)
    return path


class Images(torch.utils.data.Dataset):
    def __init__(self, split, rows, labels=None):
        self.path = build_cache(split)
        self.rows = np.asarray(rows)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        self.images = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r")
        image = torch.from_numpy(np.array(self.images[self.rows[i]]))
        if self.labels is None:
            return image
        return image, self.labels[i]


def to_float(x):
    return x.permute(0, 3, 1, 2).float().div_(255)


def normalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def resize(x, size):
    return F.interpolate(x, (size, size), mode="bilinear", align_corners=False)


def augment(x, size, scale=(0.6, 1.0), ratio=0.25, rotate=15.0, jitter=0.2):
    b, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(b, device=dev).uniform_(lo, hi)
    s = u(*scale).sqrt()
    r = u(-ratio, ratio)
    sx = (s * torch.exp(r / 2)).clamp(max=1.0)
    sy = (s * torch.exp(-r / 2)).clamp(max=1.0)
    angle = torch.deg2rad(u(-rotate, rotate)) * (torch.rand(b, device=dev) < 0.5)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - sx)
    ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - sy)
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack(
        [
            torch.stack([sx * cos * flip, -sy * sin, tx], 1),
            torch.stack([sx * sin * flip, sy * cos, ty], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, (b, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    mean = x.mean((1, 2, 3), keepdim=True)
    contrast = 1 + u(-jitter, jitter).view(b, 1, 1, 1)
    brightness = u(-jitter, jitter).view(b, 1, 1, 1)
    return ((x - mean) * contrast + mean * (1 + brightness)).clamp_(0, 1)


def create(model_name, size, **kwargs):
    import timm

    try:
        model = timm.create_model(model_name, pretrained=True, img_size=size, **kwargs)
    except TypeError:
        model = timm.create_model(model_name, pretrained=True, **kwargs)
    cfg = model.pretrained_cfg
    return model, list(cfg["mean"]), list(cfg["std"])
