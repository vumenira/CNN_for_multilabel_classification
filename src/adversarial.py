"""Adversarial validation: the final CNN's architecture, retrained to tell test images from train images.

A test image is the positive class. Out-of-fold probabilities over train + test give one AUC:
~0.5 means the two sets look alike, so validation scores on train folds should carry over to the test.
Folds keep pHash near-duplicates together, so a photo and its copy never sit on both sides of a split.

Run with `python -m src.adversarial`; folds that already have predictions in OUT_DIR are skipped.
"""

import json

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from tqdm.auto import tqdm

from src import colab, tracking
from src.best_model import ARCHITECTURE
from src.cnn import CNN
from src.config import DATA_DIR, LABELS, RESULTS_DIR, ROOT, SEED
from src.data import ImageDataset, build_cache, eval_transforms, seed_worker, split_ids, train_transforms
from src.postprocess import phash

OUT_DIR = RESULTS_DIR / "adversarial"
# MLflow runs of the folds (the curves in src.figures), copied next to the predictions after every fold
DB_COPY = OUT_DIR / "mlflow.db"
FIGURES_DIR = ROOT / "docs" / "figures"
SIZE = 224
# final1-fold0: its normalisation (training part of fold 0) and optimiser, from final_model/config.json
MEAN, STD = [0.4889, 0.4623, 0.42], [0.265, 0.2573, 0.2631]
OPTIMIZER = dict(initial_lr=0.00034698579534598526, final_lr=1e-5, weight_decay=0.001176187489328776)


def duplicate_groups(hashes: np.ndarray, max_distance: int = 6) -> np.ndarray:
    """Connected components of images linked by pHash distance <= max_distance (the EDA's groups)."""
    a, b = [], []
    for start in range(0, len(hashes), 512):
        i, j = np.nonzero(np.bitwise_count(hashes[start : start + 512, None] ^ hashes) <= max_distance)
        a.append(i + start)
        b.append(j)
    a, b = np.concatenate(a), np.concatenate(b)
    group = np.arange(len(hashes))
    while True:  # push the smallest index through every edge until nothing changes
        before = group.copy()
        np.minimum.at(group, a, group[b])
        np.minimum.at(group, b, group[a])
        if (group == before).all():
            return group


def av_folds(n_folds: int = 5) -> pd.DataFrame:
    """One row per train and test image: split, is_test, pHash group and AV fold."""
    path = OUT_DIR / "folds.csv"
    if path.exists():
        return pd.read_csv(path)
    df = pd.concat(
        [pd.DataFrame({"image_id": split_ids(s), "split": s}) for s in ("train", "test")],
        ignore_index=True,
    )
    df["is_test"] = (df.split == "test").astype(int)
    paths = [DATA_DIR / s / f"{i}.jpg" for s, i in zip(df.split, df.image_id)]
    df["group"] = duplicate_groups(phash(tqdm(paths, desc="pHash")))
    df["fold"] = -1
    cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED)
    for k, (_, rows) in enumerate(cv.split(df, df.is_test, df.group)):
        df.loc[rows, "fold"] = k
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class TrainTestDataset(ImageDataset):
    """Train and test images together; the target is 1 for a test image.

    With `cache` images come from the build_cache files of both splits, as the final model was trained.
    """

    def __init__(self, df: pd.DataFrame, transform, cache: bool = False):
        super().__init__(df[["image_id"]], transform, DATA_DIR)
        self.paths = [DATA_DIR / s / f"{i}.jpg" for s, i in zip(df.split, df.image_id)]
        self.labels = torch.tensor(df.is_test.to_numpy(np.float32)).unsqueeze(1)
        self.splits = df.split.tolist()
        if cache:
            self.cache = {s: build_cache(s, SIZE) for s in ("train", "test")}
            row_of = {s: {i: r for r, i in enumerate(split_ids(s))} for s in self.cache}
            self.rows = [row_of[s][i] for s, i in zip(self.splits, self.ids)]

    def read(self, i: int) -> np.ndarray:
        if self.cache is None:
            return super().read(i)
        if self.images is None:
            self.images = {s: np.load(p, mmap_mode="r") for s, p in self.cache.items()}
        return np.array(self.images[self.splits[i]][self.rows[i]])


class AdversarialClassifier(L.LightningModule):
    """One logit (test vs train), BCE, AdamW with cosine decay as in MultiLabelClassifier."""

    def __init__(self, model, initial_lr, final_lr, weight_decay):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_auc, self.val_auc = BinaryAUROC(), BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.train_auc.update(logits.float().sigmoid().squeeze(1), targets.squeeze(1).int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits = self(images)
        self.val_auc.update(logits.float().sigmoid().squeeze(1), targets.squeeze(1).int())
        self.log("val_loss", self.criterion(logits, targets), prog_bar=True, batch_size=len(images))
        self.log("val_auc", self.val_auc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        images, _, ids = batch
        return ids, self(images).float().sigmoid().squeeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.hparams.final_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def _loader(ds, batch_size, num_workers, train=False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )


def fit_fold(df: pd.DataFrame, fold: int, epochs: int, batch_size: int, num_workers: int, cache: bool) -> pd.DataFrame:
    """Train on the other folds; probabilities of the held-out fold from the last epoch (no selection on it)."""
    L.seed_everything(SEED, workers=True)
    tracking.setup()
    is_val = (df.fold == fold).to_numpy()
    train_tf = train_transforms(SIZE, MEAN, STD)
    train_tf.set_random_seed(SEED + fold)
    train_ds = TrainTestDataset(df[~is_val], train_tf, cache)
    val_ds = TrainTestDataset(df[is_val], eval_transforms(SIZE, MEAN, STD), cache)
    train_dl = _loader(train_ds, batch_size, num_workers, train=True)
    val_dl = _loader(val_ds, batch_size, num_workers)
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    logger = MLFlowLogger(
        experiment_name=tracking.EXPERIMENT_NAME,
        tracking_uri=tracking.TRACKING_URI,
        run_name=f"av-fold{fold}",
        tags={"model": "CNN", "stage": "adversarial", "fold": str(fold)},
    )
    logger.log_hyperparams(
        {"seed": SEED, "max_epochs": epochs, "batch_size": batch_size, "size": SIZE, "precision": precision,
         "cache": cache, **ARCHITECTURE, **OPTIMIZER}  # fmt: skip
    )
    model = AdversarialClassifier(CNN(num_classes=1, **ARCHITECTURE), **OPTIMIZER)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        enable_checkpointing=False,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dl, val_dl)
    batches = trainer.predict(model, val_dl)
    out = pd.DataFrame(
        {
            "image_id": [i for ids, _ in batches for i in ids],
            "p_test": torch.cat([p for _, p in batches]).numpy(),
        }
    )
    auc = roc_auc_score(df.is_test[is_val], out.p_test)
    logger.experiment.log_metric(logger.run_id, "final_auc", auc)
    print(f"[done] av-fold{fold}: AUC {auc:.4f}")
    return out


def summarize(oof: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Out-of-fold AUC with a bootstrap 95% interval, and the AUC of every fold."""
    rng = np.random.default_rng(SEED)
    y, p = oof.is_test.to_numpy(), oof.p_test.to_numpy()
    boot = [roc_auc_score(y[rows], p[rows]) for rows in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return {
        "auc": roc_auc_score(y, p),
        "auc_ci95": np.percentile(boot, [2.5, 97.5]).round(4).tolist(),
        "auc_per_fold": {int(k): roc_auc_score(d.is_test, d.p_test) for k, d in oof.groupby("fold")},
        "n_train": int((y == 0).sum()),
        "n_test": int(y.sum()),
    }


def run(epochs: int = 10, batch_size: int = 64, num_workers: int = 2, cache: bool = True) -> pd.DataFrame:
    """All folds, then oof.csv and summary.json in OUT_DIR. Batch size, workers and cache as for the final model."""
    torch.backends.cudnn.benchmark = True
    df = av_folds()
    colab.restore_sqlite(tracking.MLRUNS_DIR / "mlflow.db", DB_COPY)  # continue the runs of earlier folds
    parts = []
    for fold in sorted(map(int, df.fold.unique())):
        path = OUT_DIR / f"oof_fold{fold}.csv"
        if not path.exists():
            fit_fold(df, fold, epochs, batch_size, num_workers, cache).to_csv(path, index=False)
            colab.backup_sqlite(tracking.MLRUNS_DIR / "mlflow.db", DB_COPY)
        parts.append(pd.read_csv(path))
    oof = df.merge(pd.concat(parts), on="image_id", validate="one_to_one")
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    summary = summarize(oof)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return oof


def image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-image features: file size, JPEG quality (mean luma quantisation step), mean and std per channel."""
    path = OUT_DIR / "image_stats.csv"
    if path.exists():
        return pd.read_csv(path).set_index("image_id").loc[df.image_id]
    rows = []
    for split, image_id in tqdm(zip(df.split, df.image_id), total=len(df), desc="stats"):
        file = DATA_DIR / split / f"{image_id}.jpg"
        with Image.open(file) as im:
            q_luma = np.mean(im.quantization[0])
            px = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        rows.append([image_id, file.stat().st_size, q_luma, *px.mean((0, 1)), *px.std((0, 1))])
    columns = ["image_id", "bytes", "q_luma", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    stats = pd.DataFrame(rows, columns=columns).set_index("image_id")
    stats.to_csv(path)
    return stats


def metadata_auc(df: pd.DataFrame) -> float:
    """Out-of-fold AUC of gradient boosting on image_stats, same folds: the shift a model can find without pixels."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, p = image_stats(df).to_numpy(), np.zeros(len(df))
    for k in sorted(df.fold.unique()):
        held = (df.fold == k).to_numpy()
        clf = HistGradientBoostingClassifier(random_state=SEED).fit(X[~held], df.is_test[~held])
        p[held] = clf.predict_proba(X[held])[:, 1]
    return roc_auc_score(df.is_test, p)


def test_likeness_by_label(oof: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold p_test of train images by label group: does the model find some kinds of photos test-like?"""
    train = oof[oof.split == "train"].merge(pd.read_csv(DATA_DIR / "train.csv"), on="image_id")
    named = train[["Moris", "Motya", "Biatrix"]].any(axis=1)
    kind = np.select(
        [named, train.Cat.eq(1) & train.Dog.eq(0), train.Dog.eq(1) & train.Cat.eq(0), train[LABELS].sum(axis=1).eq(0)],
        ["named animal", "Cat only", "Dog only", "no label"],
        "other",
    )
    return train.groupby(kind).p_test.describe()[["count", "mean", "50%", "max"]].round(3)


def plot_report(oof: pd.DataFrame, summary: dict) -> None:
    """Histogram of out-of-fold p_test per split, and the most test-like train / most train-like test photos."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 51)
    for split, d in oof.groupby("split"):
        ax.hist(d.p_test, bins=bins, density=True, alpha=0.55, label=f"{split} ({len(d)})")
    ax.axvline(oof.is_test.mean(), color="k", lw=0.8, ls="--", label="test share")
    lo, hi = summary["auc_ci95"]
    ax.set(xlabel="out-of-fold P(test)", ylabel="density", title=f"Adversarial validation: AUC {summary['auc']:.3f} [{lo:.3f}, {hi:.3f}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "av_hist.png", dpi=120)
    plt.close(fig)

    extremes = {
        "Train photos the model finds most test-like": oof[oof.split == "train"].nlargest(8, "p_test"),
        "Test photos the model finds most train-like": oof[oof.split == "test"].nsmallest(8, "p_test"),
    }
    fig = plt.figure(figsize=(16, 5.2), layout="constrained")
    for sub, (title, d) in zip(fig.subfigures(2, 1), extremes.items()):
        sub.suptitle(title, fontsize=10, weight="bold", x=0.01, ha="left")
        for ax, (_, r) in zip(sub.subplots(1, 8), d.iterrows()):
            ax.imshow(Image.open(DATA_DIR / r["split"] / f"{r['image_id']}.jpg"))
            ax.set_title(f"P(test) {r['p_test']:.2f}\n{r['image_id'][:8]}", fontsize=8)
            ax.axis("off")
    fig.savefig(FIGURES_DIR / "av_extremes.png", dpi=100)
    plt.close(fig)


def report(oof: pd.DataFrame) -> dict:
    """Metadata AUC (added to summary.json), test-likeness by label (likeness_by_label.csv) and the figures."""
    summary = json.loads((OUT_DIR / "summary.json").read_text())
    summary["metadata_auc"] = metadata_auc(oof)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    likeness = test_likeness_by_label(oof)
    likeness.to_csv(OUT_DIR / "likeness_by_label.csv")
    print(f"metadata AUC {summary['metadata_auc']:.4f}")
    print(likeness)
    plot_report(oof, summary)
    return summary


if __name__ == "__main__":
    report(run())
