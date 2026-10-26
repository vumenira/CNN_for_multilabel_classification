"""Evaluation of a finished run and export of everything the report needs."""

import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from torchmetrics.functional.classification import (
    multilabel_average_precision,
    multilabel_f1_score,
)

from src import tracking
from src.config import LABELS, RESULTS_DIR
from src.postprocess import best_thresholds


def evaluate(probs: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
    """Threshold-free (AP) and thresholded (F1, exact match) validation metrics, flat for logging."""
    n = len(LABELS)
    p = torch.tensor(probs.to_numpy(), dtype=torch.float32)
    t = torch.tensor(y.to_numpy()).int()
    thresholds = best_thresholds(probs, y)
    tuned = torch.tensor((probs >= thresholds).to_numpy(dtype="float32"))
    f1 = multilabel_f1_score(p, t, n, average=None)
    ap = multilabel_average_precision(p, t, n, average=None)
    return {
        "ap_macro": ap.mean().item(),
        "f1_macro_05": f1.mean().item(),
        "f1_macro_tuned": multilabel_f1_score(tuned, t, n).item(),
        "exact_match_05": ((p >= 0.5).int() == t).all(1).float().mean().item(),
        **{f"ap_{l}": v.item() for l, v in zip(LABELS, ap)},
        **{f"f1_{l}": v.item() for l, v in zip(LABELS, f1)},
        **{f"threshold_{l}": float(v) for l, v in thresholds.items()},
    }


def metric_history(run_id: str) -> pd.DataFrame:
    """Every logged metric of an MLflow run as long format (key, step, value)."""
    client = MlflowClient(tracking_uri=tracking.TRACKING_URI)
    keys = client.get_run(run_id).data.metrics
    rows = [
        (key, m.step, m.value)
        for key in keys
        for m in client.get_metric_history(run_id, key)
    ]
    return pd.DataFrame(rows, columns=["key", "step", "value"]).sort_values(["key", "step"])


def log_final_metrics(run_id: str, metrics: dict[str, float]) -> None:
    client = MlflowClient(tracking_uri=tracking.TRACKING_URI)
    for key, value in metrics.items():
        client.log_metric(run_id, f"final_{key}", value)


def plot_curves(history: pd.DataFrame, path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = set(history.key)
    panels = {
        "Loss": ["train_loss_epoch", "val_loss"],
        "Validation quality": ["val_ap", "val_f1"],
        "Val AP per class": [f"val_ap_{l}" for l in LABELS],
        "Gradient norm per module": sorted(
            k for k in keys if k.startswith("grad_norm/") and k.endswith("_epoch")
        ),
        "Learning rate": sorted(k for k in keys if k.startswith("lr-")),
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (name, names) in zip(axes, panels.items()):
        for key in names:
            d = history[history.key == key]
            if len(d):
                ax.plot(d.step, d.value, label=key.removeprefix("grad_norm/").removesuffix("_epoch"))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        if name.startswith("Gradient"):
            ax.set_yscale("log")
        if ax.lines:
            ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def export_run(
    run_dir: Path,
    summary: dict,
    val_probs: pd.DataFrame,
    test_probs: pd.DataFrame | None,
    best_checkpoint: str,
) -> None:
    """Write one run's files into run_dir (a Drive folder in Colab). summary.json goes last: it marks the run done."""
    run_dir.mkdir(parents=True, exist_ok=True)
    history = metric_history(summary["run_id"])
    history.to_csv(run_dir / "metrics.csv", index=False)
    client = MlflowClient(tracking_uri=tracking.TRACKING_URI)
    params = client.get_run(summary["run_id"]).data.params
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    plot_curves(history, run_dir / "curves.png", summary["name"])
    val_probs.to_csv(run_dir / "val_probs.csv")
    if test_probs is not None:
        test_probs.to_csv(run_dir / "test_probs.csv")
    if best_checkpoint:
        shutil.copy2(best_checkpoint, run_dir / "best.ckpt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def summarize(results_dir: Path | str | None = None) -> pd.DataFrame:
    """One row per finished or failed run, best validation AP first."""
    rows = []
    for folder in sorted(Path(results_dir or RESULTS_DIR).iterdir()):
        for marker in ("summary.json", "failed.json"):
            if (folder / marker).exists():
                rows.append(json.loads((folder / marker).read_text()))
                break
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    df = df.drop(columns=[c for c in ("error", "config.trainer_kwargs") if c in df], errors="ignore")
    if "best_score" in df:
        df = df.sort_values("best_score", ascending=False, na_position="last")
    return df.reset_index(drop=True)
