import copy

import albumentations as A
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.classification import multilabel_f1_score

from src.config import LABELS


def hflip(ds):
    """Same dataset with every image flipped horizontally, for TTA."""
    flipped = copy.copy(ds)
    flipped.transform = A.Compose([A.HorizontalFlip(p=1), *ds.transform.transforms])
    return flipped


def phash(paths) -> np.ndarray:
    return np.array(
        [int(str(imagehash.phash(Image.open(p))), 16) for p in paths], dtype=np.uint64
    )


def blend_duplicates(
    probs: pd.DataFrame, hashes, train_hashes, train_labels, max_distance=2, weight=0.5
):
    """Photos with train photos within max_distance: p = (1 − λ)·p + λ·mean label vector of those photos."""
    out = probs.astype(float)
    for i, h in enumerate(hashes):
        match = np.bitwise_count(train_hashes ^ h) <= max_distance
        if match.any():
            out.iloc[i] = (1 - weight) * out.iloc[i] + weight * train_labels[
                match
            ].mean(0)
    return out


def best_thresholds(probs: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Per-class threshold with the best F1 on a 0.05…0.95 grid."""
    grid = np.linspace(0.05, 0.95, 19)
    p, y = torch.tensor(probs.to_numpy()), torch.tensor(targets.to_numpy())
    f1 = torch.stack(
        [
            multilabel_f1_score(p, y, len(LABELS), threshold=t, average=None)
            for t in grid
        ]
    )
    return pd.Series(grid[f1.argmax(0).numpy()], index=LABELS)


def best_combination(
    probs: pd.DataFrame, thresholds: pd.Series, combos: np.ndarray
) -> pd.DataFrame:
    """Best of the label combinations seen in train (rows of combos).

    A combination scores Σ logit(p) − logit(threshold) over its positive classes,
    so whatever the thresholds give stays whenever it is one of combos.
    """
    scores = torch.logit(torch.tensor(probs.to_numpy()), eps=1e-6) - torch.logit(
        torch.tensor(thresholds.to_numpy())
    )
    best = (scores @ torch.tensor(combos.T, dtype=scores.dtype)).argmax(1)
    return pd.DataFrame(combos[best.numpy()], index=probs.index, columns=LABELS)


def add_parents(pred: pd.DataFrame) -> pd.DataFrame:
    """A named animal implies its species."""
    return pred.assign(
        Cat=pred.Cat | pred.Moris | pred.Motya, Dog=pred.Dog | pred.Biatrix
    )
