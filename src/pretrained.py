import hashlib
import os

import lightning as L
import timm
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src import tracking
from src.classifier import MultiLabelClassifier
from src.config import FOLDS_PATH, LABELS, MODELS_DIR, ROOT, SEED
from src.data import DataModule


def train(
    model_name,
    size,
    fold=0,
    max_epochs=10,
    batch_size=16,
    num_workers=None,
    initial_lr=1e-4,
    cache=False,
    callbacks=(),
    trainer_kwargs=None,
    ckpt_path=None,
    run_id=None,
    **classifier_kwargs,
):
    L.seed_everything(SEED, workers=True)
    tracking.setup()
    num_workers = min(4, os.cpu_count()) if num_workers is None else num_workers
    net = timm.create_model(
        model_name, pretrained=True, num_classes=len(LABELS), drop_path_rate=0.1
    )
    data_config = timm.data.resolve_model_data_config(net)
    datamodule = DataModule(
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
        fold=fold,
        size=size,
        batch_size=batch_size,
        num_workers=num_workers,
        cache=cache,
    )
    model = MultiLabelClassifier(net, initial_lr=initial_lr, **classifier_kwargs)
    logger = MLFlowLogger(
        experiment_name=tracking.EXPERIMENT_NAME,
        tracking_uri=tracking.TRACKING_URI,
        run_name=f"{model_name}-fold{fold}",
        tags={"model": model_name, "fold": str(fold)},
        run_id=run_id,
    )
    logger.log_hyperparams(
        {
            "seed": SEED,
            "max_epochs": max_epochs,
            "model_name": model_name,
            "size": size,
            "labels": LABELS,
            "num_parameters": sum(p.numel() for p in net.parameters()),
            "folds_sha256": hashlib.sha256(FOLDS_PATH.read_bytes()).hexdigest(),
        }
    )
    logger.experiment.log_text(logger.run_id, str(net), "architecture.txt")
    for path in (ROOT / "src").glob("*.py"):
        logger.experiment.log_artifact(logger.run_id, str(path), "source")

    checkpoint = ModelCheckpoint(
        dirpath=MODELS_DIR / logger.run_id,
        filename=model_name + "-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch"), *callbacks],
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        **(trainer_kwargs or {}),
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"MLflow experiment: {tracking.EXPERIMENT_NAME}; run: {logger.run_id}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return trainer
