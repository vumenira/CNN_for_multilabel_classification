import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import DATA_DIR, FOLDS_PATH, LABELS, SEED

N_FOLDS = 5


def assign_folds(labels: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold index per row; iterative stratification keeps each label's share equal across folds (lecture 7, §9.3)."""
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(labels), -1)
    for k, (_, val_idx) in enumerate(mskf.split(labels, labels)):
        fold[val_idx] = k
    return fold


if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv")
    folds = train.assign(fold=assign_folds(train[LABELS]))
    positives = folds.groupby("fold")[LABELS].sum()
    assert (folds.fold >= 0).all() and (positives > 0).all(axis=None)
    folds.to_csv(FOLDS_PATH, index=False)
    print(positives)
