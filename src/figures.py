"""Report figures from the logged runs: loss and metric curves, gradient flow, adversarial validation curves.

Sources:
- mlruns/mac/mlflow.db: the MLflow runs of the baseline CNN (86ea605c) and ConvNeXt-B (b4bddc3f);
- results/final_model/: the Drive export of the final CNN (final1-fold0);
- results/adversarial/mlflow.db: the adversarial-validation runs (stage=adversarial), copied there by src.adversarial.

Run with `python -m src.figures`; PNGs go to docs/figures/.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import tracking
from src.config import LABELS, RESULTS_DIR, ROOT

FIGURES_DIR = ROOT / "docs" / "figures"
MAC_DB = tracking.MLRUNS_DIR / "mac" / "mlflow.db"
FINAL_EXPORT = RESULTS_DIR / "final_model"
AV_DB = RESULTS_DIR / "adversarial" / "mlflow.db"
NAMED = ["Moris", "Motya", "Biatrix"]


def _query(db: Path, sql: str, params=()) -> pd.DataFrame:
    with closing(sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)) as con:
        return pd.read_sql(sql, con, params=params)


def per_epoch(db: Path, run_id: str) -> pd.DataFrame:
    """One row per epoch of an MLflow run (id or its prefix): every per-epoch key, numbered in logging order.

    Steps of epoch-level metrics differ between keys (the lr monitor logs at the start of an epoch, the rest at
    its end), so counting is more reliable than matching steps.
    """
    m = _query(
        db,
        "select key, step, value from metrics where run_uuid like ? order by key, step, timestamp",
        [run_id + "%"],
    )
    m = m[~m.key.str.endswith("_step") & (m.key != "epoch") & ~m.key.str.startswith("final_")].copy()
    m["epoch"] = m.groupby("key").cumcount()
    return m.pivot(index="epoch", columns="key", values="value")


def final_cnn() -> pd.DataFrame:
    """Per-epoch metrics of final1-fold0 from its export, with the learning rate restored.

    The export kept the lr of only two epochs (it was logged at other steps than the rest), so the cosine schedule
    of config.json fills it in; the kept values check the formula.
    """
    h = pd.read_csv(FINAL_EXPORT / "metrics_per_epoch.csv").drop(columns="step")
    h = h.set_index(h.pop("epoch").astype(int))
    cfg = json.loads((FINAL_EXPORT / "config.json").read_text())
    e = h.index.to_numpy()
    lr = cfg["final_lr"] + (cfg["lr"] - cfg["final_lr"]) * (1 + np.cos(np.pi * e / cfg["epochs"])) / 2
    kept = h["lr-AdamW"].notna().to_numpy()
    assert np.allclose(h["lr-AdamW"][kept], lr[kept], rtol=1e-4), "lr export does not match the cosine schedule"
    h["lr-AdamW"] = lr
    return h


def parameter_counts() -> dict[str, dict[str, int]]:
    """Parameters per top-level module, the units of grad_norm/<module>."""
    import timm

    from src.best_model import ARCHITECTURE
    from src.cnn import CNN

    nets = {
        "Baseline CNN": CNN(),
        "Final CNN": CNN(**ARCHITECTURE),
        "ConvNeXt-B": timm.create_model("convnext_base.fb_in22k_ft_in1k_384", pretrained=False, num_classes=5),
    }
    return {
        name: {n: sum(p.numel() for p in m.parameters()) for n, m in net.named_children()}
        for name, net in nets.items()
    }


def histories() -> dict[str, pd.DataFrame]:
    return {
        "Baseline CNN": per_epoch(MAC_DB, "86ea605c"),
        "Final CNN": final_cnn(),
        "ConvNeXt-B": per_epoch(MAC_DB, "b4bddc3f"),
    }


def _best_epoch(h: pd.DataFrame) -> int:
    """The epoch the run's checkpoint kept: best val_ap if logged (final CNN), else best val_f1."""
    return int(h["val_ap" if "val_ap" in h else "val_f1"].idxmax())


def plot_curves(name: str, h: pd.DataFrame, path: Path) -> None:
    """Loss, macro scores, per-class validation score and learning rate of one run, best epoch marked."""
    per_class = "val_ap" if "val_ap" in h else "val_f1"
    panels = {
        "Loss (BCE)": ["train_loss_epoch", "val_loss"],
        "Macro score": [k for k in ("train_f1", "val_f1", "val_ap") if k in h],
        f"Validation {per_class.removeprefix('val_').upper()} per class": [f"{per_class}_{l}" for l in LABELS],
        "Learning rate": ["lr-AdamW"],
    }
    best = _best_epoch(h)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, (title, keys) in zip(axes, panels.items()):
        for key in keys:
            ax.plot(h.index, h[key], marker=".", label=key.removeprefix(f"{per_class}_"))
        ax.axvline(best, color="grey", lw=0.8, ls="--")
        ax.set(title=title, xlabel="epoch")
        if title == "Learning rate":
            ax.set_yscale("log")
        else:
            ax.legend(fontsize=7)
    axes[1].text(best, axes[1].get_ylim()[0], f"best epoch {best} ", fontsize=7, color="grey", ha="right", va="bottom")
    fig.suptitle(name, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_comparison(runs: dict[str, pd.DataFrame], path: Path) -> None:
    """Validation macro F1 at threshold 0.5 and mean F1 of the named animals, per epoch, for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, h in runs.items():
        axes[0].plot(h.index, h.val_f1, marker=".", label=name)
        axes[1].plot(h.index, h[[f"val_f1_{l}" for l in NAMED]].mean(axis=1), marker=".", label=name)
    axes[0].set(title="Validation macro F1 (threshold 0.5)", xlabel="epoch", ylim=(0, 1.02))
    axes[1].set(title="Validation F1, mean of Moris / Motya / Biatrix", xlabel="epoch", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gradient_flow(runs: dict[str, pd.DataFrame], params: dict[str, dict[str, int]], path: Path) -> None:
    """Top: L2 norm of the gradient of each module over epochs (mean over steps).
    Bottom: the same as root mean square per parameter, modules in input → output order, one line per epoch.

    Under fp16 mixed precision an overflowing step has infinite gradients (GradScaler skips it and lowers the
    scale), which makes that epoch's mean infinite; such points are left out and named on the plot.
    """
    fig, axes = plt.subplots(2, len(runs), figsize=(5.4 * len(runs), 7.2))
    for (name, h), (top, bottom) in zip(runs.items(), axes.T):
        modules = [m for m in params[name] if f"grad_norm/{m}_epoch" in h]
        norms = h[[f"grad_norm/{m}_epoch" for m in modules]].set_axis(modules, axis=1)
        overflow = ~np.isfinite(norms) | (norms >= np.finfo(np.float64).max)  # MLflow stores inf as the max float
        norms = norms.mask(overflow)
        for m in modules:
            top.plot(norms.index, norms[m], marker=".", label=m)
        notes = [f"{m} epochs {', '.join(map(str, norms.index[overflow[m]]))}" for m in modules if overflow[m].any()]
        title = name + (f"\nnot shown, fp16 overflow step: {'; '.join(notes)}" if notes else "")
        top.set(title=title, xlabel="epoch", ylabel="gradient L2 norm", yscale="log")
        top.title.set_fontsize(9 if notes else 12)
        top.legend(fontsize=7)
        rms = norms / np.sqrt([params[name][m] for m in modules])
        colors = plt.cm.viridis(np.linspace(0, 1, len(rms)))
        for (epoch, row), color in zip(rms.iterrows(), colors):
            bottom.plot(modules, row, marker="o", ms=3, color=color, lw=1)
        bottom.set(xlabel="module (input → output)", ylabel="gradient RMS per parameter", yscale="log")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(rms.index.min(), rms.index.max()))
        fig.colorbar(sm, ax=bottom, label="epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def adversarial_runs(db: Path = AV_DB) -> dict[str, pd.DataFrame]:
    """Per-epoch metrics of the finished adversarial-validation folds, by run name."""
    if not db.exists():
        return {}
    runs = _query(
        db,
        "select r.run_uuid, r.name from runs r join tags t on t.run_uuid = r.run_uuid "
        "where t.key = 'stage' and t.value = 'adversarial' and r.status = 'FINISHED' order by r.name, r.start_time",
    )
    # a fold that was run more than once keeps its last finished run
    return {name: per_epoch(db, run_id) for run_id, name in runs.values}


def plot_adversarial(runs: dict[str, pd.DataFrame], path: Path) -> None:
    """Loss and AUC of every adversarial-validation fold, train and held-out."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for (name, h), color in zip(runs.items(), plt.cm.tab10.colors):
        axes[0].plot(h.index, h.train_loss_epoch, color=color, ls="--")
        axes[0].plot(h.index, h.val_loss, color=color, marker=".", label=name)
        axes[1].plot(h.index, h.train_auc, color=color, ls="--")
        axes[1].plot(h.index, h.val_auc, color=color, marker=".", label=name)
    axes[1].axhline(0.5, color="k", lw=0.8)
    axes[0].set(title="Loss (dashed: train, solid: held-out fold)", xlabel="epoch")
    axes[1].set(title="AUC test vs train (dashed: train, solid: held-out fold)", xlabel="epoch")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = histories()
    for name, h in runs.items():
        plot_curves(name, h, FIGURES_DIR / f"curves_{name.lower().replace(' ', '_').replace('-', '')}.png")
        print(f"{name}: {len(h)} epochs, best epoch {_best_epoch(h)}")
    plot_comparison(runs, FIGURES_DIR / "curves_comparison.png")
    plot_gradient_flow(runs, parameter_counts(), FIGURES_DIR / "grad_flow.png")
    if av := adversarial_runs():
        plot_adversarial(av, FIGURES_DIR / "av_curves.png")
    print(sorted(p.name for p in FIGURES_DIR.iterdir()))
