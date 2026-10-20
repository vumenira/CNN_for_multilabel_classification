import hashlib

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src import tracking
from src.classifier import MultiLabelClassifier
from src.config import DATA_DIR, FOLDS_PATH, LABELS, ROOT
from src.data import DataModule, channel_stats

SEED = 42


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, use_batchnorm=True):
    """Conv + BatchNorm; without BN the conv keeps its bias."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=kernel_size // 2, bias=not use_batchnorm),
        nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
    ]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super().__init__()
        self.layers = nn.Sequential(
            *conv_bn(in_channels, out_channels, stride=stride, use_batchnorm=use_batchnorm),
            nn.ReLU(),
            *conv_bn(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                *conv_bn(in_channels, out_channels, kernel_size=1, stride=stride, use_batchnorm=use_batchnorm),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x))


def stage(in_channels, out_channels, stride, use_batchnorm):
    """Two residual blocks; the first one changes width and resolution."""
    return nn.Sequential(
        ResidualBlock(in_channels, out_channels, stride, use_batchnorm),
        ResidualBlock(out_channels, out_channels, 1, use_batchnorm),
    )


class CNN(nn.Module):
    def __init__(self, num_classes=len(LABELS), use_batchnorm=True):
        super().__init__()
        self.stem = nn.Sequential(
            *conv_bn(3, 32, stride=2, use_batchnorm=use_batchnorm),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = stage(32, 32, 1, use_batchnorm)
        self.stage2 = stage(32, 64, 2, use_batchnorm)
        self.stage3 = stage(64, 128, 2, use_batchnorm)
        self.stage4 = stage(128, 256, 2, use_batchnorm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def train(fold=0, max_epochs=10, batch_size=64, num_workers=4, use_batchnorm=True, **classifier_kwargs):
    """classifier_kwargs go to MultiLabelClassifier, e.g. optimizer="Adam", initial_lr=3e-4."""
    L.seed_everything(SEED, workers=True)
    tracking.setup()

    folds = pd.read_csv(FOLDS_PATH)
    mean, std = channel_stats(DATA_DIR / "train" / f"{i}.jpg" for i in folds[folds.fold != fold].image_id)
    datamodule = DataModule(mean=mean.round(4).tolist(), std=std.round(4).tolist(),
                            fold=fold, batch_size=batch_size, num_workers=num_workers)
    model = MultiLabelClassifier(CNN(use_batchnorm=use_batchnorm), **classifier_kwargs)
    logger = MLFlowLogger(
        experiment_name=tracking.EXPERIMENT_NAME, tracking_uri=tracking.TRACKING_URI,
        run_name=f"cnn-fold{fold}", tags={"model": "CNN", "fold": str(fold)},
    )
    logger.log_hyperparams({
        "seed": SEED, "max_epochs": max_epochs, "use_batchnorm": use_batchnorm, "labels": LABELS,
        "num_parameters": sum(p.numel() for p in model.model.parameters()),
        "folds_sha256": hashlib.sha256(FOLDS_PATH.read_bytes()).hexdigest(),
    })
    logger.experiment.log_text(logger.run_id, str(model.model), "architecture.txt")
    for path in (ROOT / "src").glob("*.py"):
        logger.experiment.log_artifact(logger.run_id, str(path), "source")

    checkpoint = ModelCheckpoint(
        dirpath=ROOT / "models" / logger.run_id,
        filename="cnn-{epoch:02d}-{val_f1:.4f}", monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator="auto", devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger, callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=20, num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"MLflow experiment: {tracking.EXPERIMENT_NAME}; run: {logger.run_id}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return trainer


if __name__ == "__main__":
    train()
