"""Unattended experiment runner: fixed ablation lists, an Optuna study and final longer runs.

Every run is isolated (a crash is recorded and the next run starts), resumable (a finished run
has summary.json in its folder and is skipped) and exported to RESULTS_DIR, which is on Drive in Colab.
"""

import gc
import json
import re
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

from src import colab, results, tracking
from src.classifier import predict
from src.cnn import train
from src.config import FOLDS_PATH, LABELS, RESULTS_DIR, ROOT
from src.data import class_pos_weights

DEFAULTS = dict(
    name=None,
    stage="",
    fold=0,
    epochs=20,
    batch_size=64,
    num_workers=2,
    optimizer="AdamW",
    scheduler="cosine",
    lr=1e-3,
    final_lr=1e-5,
    weight_decay=1e-4,
    widths=[32, 64, 128, 256],
    blocks=[2, 2, 2, 2],
    activation="relu",
    dropout=0.2,
    use_batchnorm=True,
    pos_weight="none",
    size=224,
    monitor="val_ap",
    max_minutes=40,
    predict_test=False,
    trainer_kwargs={},
)

DEPTHS = {"shallow": [1, 1, 1, 1], "base": [2, 2, 2, 2], "deep": [2, 3, 4, 2]}

# The first Optuna trial: the baseline architecture, so every study contains a reference point.
BASELINE_TRIAL = dict(
    base_width=32, depth="base", activation="relu", dropout=0.2,
    lr=1e-3, weight_decay=1e-4, pos_weight="none",
)  # fmt: skip


class PruneCallback(Callback):
    """Reports the monitored metric to Optuna after every validation and stops a hopeless run."""

    def __init__(self, trial, monitor: str):
        self.trial, self.monitor, self.pruned = trial, monitor, False

    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get(self.monitor)
        if trainer.sanity_checking or value is None:
            return
        self.trial.report(float(value), trainer.current_epoch)
        if self.trial.should_prune():
            self.pruned = True
            trainer.should_stop = True


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _backup_mlflow(results_dir: Path):
    """Copy the MLflow database next to the exports; never allowed to break a run."""
    try:
        colab.backup_sqlite(tracking.MLRUNS_DIR / "mlflow.db", results_dir / "mlflow.db")
    except Exception as error:
        print(f"MLflow backup failed: {error}")


def _fit_and_export(cfg: dict, run_dir: Path, trial) -> dict:
    torch.backends.cudnn.benchmark = True
    prune = PruneCallback(trial, cfg["monitor"]) if trial is not None else None
    started = time.time()
    trainer = train(
        fold=cfg["fold"],
        max_epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        use_batchnorm=cfg["use_batchnorm"],
        model_kwargs=dict(
            widths=cfg["widths"],
            blocks=cfg["blocks"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        ),
        cache=True,
        size=cfg["size"],
        run_name=cfg["name"],
        tags={"stage": cfg["stage"]},
        monitor=cfg["monitor"],
        ckpt_dir=ROOT / "models" / cfg["name"],  # local disk; only the best one is copied out
        callbacks=[prune] if prune else [],
        trainer_kwargs={"max_time": {"minutes": cfg["max_minutes"]}, **cfg["trainer_kwargs"]},
        optimizer=cfg["optimizer"],
        scheduler=cfg["scheduler"],
        initial_lr=cfg["lr"],
        final_lr=cfg["final_lr"],
        weight_decay=cfg["weight_decay"],
        pos_weight=class_pos_weights(cfg["fold"], cfg["pos_weight"]),
    )
    train_seconds = time.time() - started
    run_id = trainer.logger.run_id
    checkpoint = trainer.checkpoint_callback
    datamodule = trainer.datamodule

    val = predict(trainer, dataloaders=datamodule.val_dataloader(), ckpt_path="best")
    y = pd.read_csv(FOLDS_PATH).set_index("image_id").loc[val.index, LABELS]
    final = results.evaluate(val, y)
    results.log_final_metrics(run_id, final)
    test = predict(trainer, datamodule=datamodule, ckpt_path="best") if cfg["predict_test"] else None

    epochs_run = trainer.current_epoch
    summary = {
        "name": cfg["name"],
        "status": "pruned" if prune and prune.pruned else "done",
        "run_id": run_id,
        "config": cfg,
        "best_score": float(checkpoint.best_model_score),
        "best_epoch": int(re.search(r"epoch=(\d+)", checkpoint.best_model_path).group(1)),
        "epochs_run": epochs_run,
        "train_seconds": round(train_seconds, 1),
        "sec_per_epoch": round(train_seconds / max(epochs_run, 1), 1),
        "num_parameters": sum(p.numel() for p in trainer.lightning_module.model.parameters()),
        **final,
    }
    results.export_run(run_dir, summary, val, test, checkpoint.best_model_path)
    shutil.rmtree(ROOT / "models" / cfg["name"], ignore_errors=True)
    return summary


def run_experiment(cfg: dict, results_dir: Path | str | None = None, trial=None) -> dict:
    """Train and evaluate one configuration; returns its summary, or a record with status "failed"."""
    cfg = {**DEFAULTS, **cfg}
    results_dir = Path(results_dir or RESULTS_DIR)
    run_dir = results_dir / cfg["name"]
    stored = run_dir / "summary.json"
    if stored.exists():
        summary = json.loads(stored.read_text())
        if summary["config"] == json.loads(json.dumps(cfg)):
            print(f"[skip] {cfg['name']} already done")
            return summary
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)

    print(f"[start] {cfg['name']}")
    summary, error = None, None
    try:
        summary = _fit_and_export(cfg, run_dir, trial)
    except Exception:
        error = traceback.format_exc()
        print(error)
    _cleanup()
    shutil.rmtree(ROOT / "models" / cfg["name"], ignore_errors=True)

    if error is not None:
        summary = {"name": cfg["name"], "status": "failed", "config": cfg, "error": error}
        (run_dir / "failed.json").write_text(json.dumps(summary, indent=2))
    else:
        print(
            f"[{summary['status']}] {cfg['name']}: val AP {summary['best_score']:.4f}, "
            f"F1@0.5 {summary['f1_macro_05']:.4f}, {summary['sec_per_epoch']} s/epoch"
        )
    _backup_mlflow(results_dir)
    return summary


def run_list(cfgs: list[dict], results_dir=None, deadline: float | None = None) -> list[dict]:
    """Run configurations one after another; stops starting new ones once `deadline` (epoch seconds) passes."""
    out = []
    for cfg in cfgs:
        if deadline is not None and time.time() > deadline:
            print(f"[budget] skipping {cfg['name']}")
            out.append({"name": cfg["name"], "status": "skipped"})
            continue
        out.append(run_experiment(cfg, results_dir))
    return out


def suggest_config(trial) -> dict:
    """Search space of the Optuna study: architecture and optimisation knobs for one trial."""
    # TODO(human): adjust the ranges / choices below to what you think is worth searching.
    base_width = trial.suggest_categorical("base_width", [32, 48, 64])
    return dict(
        widths=[base_width * m for m in (1, 2, 4, 8)],
        blocks=DEPTHS[trial.suggest_categorical("depth", list(DEPTHS))],
        activation=trial.suggest_categorical("activation", ["relu", "silu"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        pos_weight=trial.suggest_categorical("pos_weight", ["none", "sqrt"]),
    )


def run_optuna(
    study_name: str,
    n_trials: int,
    results_dir=None,
    overrides: dict | None = None,
    timeout: float | None = None,
    seed: int = 42,
):
    """Maximise validation AP over suggest_config; resumable from the study database backed up in results_dir.

    `overrides` is merged into every trial's config (e.g. epochs); `timeout` (seconds) stops new trials.
    """
    import optuna

    results_dir = Path(results_dir or RESULTS_DIR)
    tracking.MLRUNS_DIR.mkdir(exist_ok=True)
    local_db = tracking.MLRUNS_DIR / "optuna.db"
    if colab.restore_sqlite(local_db, results_dir / "optuna.db"):
        print("restored the Optuna study from the backup")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{local_db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(BASELINE_TRIAL)

    def objective(trial):
        cfg = {
            **suggest_config(trial),
            **(overrides or {}),
            "name": f"{study_name}-{trial.number:03d}",
            "stage": study_name,
        }
        trial.set_user_attr("cfg", cfg)
        summary = run_experiment(cfg, results_dir, trial)
        if summary["status"] == "failed":
            raise RuntimeError(summary["error"].strip().splitlines()[-1])
        if summary["status"] == "pruned":
            raise optuna.TrialPruned()
        return summary["best_score"]

    def backup(study, trial):
        try:
            colab.backup_sqlite(local_db, results_dir / "optuna.db")
            study.trials_dataframe().to_csv(results_dir / f"{study_name}_trials.csv", index=False)
        except Exception as error:
            print(f"Optuna backup failed: {error}")

    finished = sum(t.state.is_finished() for t in study.trials)
    study.optimize(
        objective,
        n_trials=max(n_trials - finished, 0),
        timeout=timeout,
        callbacks=[backup],
        catch=(RuntimeError,),
    )
    return study


def final_configs(study, top_k: int = 2, epochs: int = 40, folds=(0, 1)) -> list[dict]:
    """Longer runs of the best finished trials, on several folds, with test predictions for ensembling."""
    import optuna

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = sorted(done, key=lambda t: t.value, reverse=True)[:top_k]
    return [
        {
            **t.user_attrs["cfg"],
            "name": f"final{rank}-fold{fold}",
            "stage": "final",
            "epochs": epochs,
            "fold": fold,
            "predict_test": True,
            "max_minutes": 90,
        }
        for rank, t in enumerate(best, start=1)
        for fold in folds
    ]
