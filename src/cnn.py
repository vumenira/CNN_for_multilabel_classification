import hashlib

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src import tracking
from src.classifier import MultiLabelClassifier
from src.config import DATA_DIR, FOLDS_PATH, LABELS, ROOT, SEED
from src.data import DataModule, channel_stats, fold_stats


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, use_batchnorm=True):
    """Conv + BatchNorm; without BN the conv keeps its bias."""
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=not use_batchnorm,
        ),
        nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
    ]


ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU}


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, use_batchnorm=True, activation=nn.ReLU
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *conv_bn(
                in_channels, out_channels, stride=stride, use_batchnorm=use_batchnorm
            ),
            activation(),
            *conv_bn(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                *conv_bn(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    use_batchnorm=use_batchnorm,
                ),
            )
        self.act = activation()

    def forward(self, x):
        return self.act(self.layers(x) + self.shortcut(x))


def stage(in_channels, out_channels, stride, use_batchnorm, blocks=2, activation=nn.ReLU):
    """Residual blocks; the first one changes width and resolution."""
    return nn.Sequential(
        ResidualBlock(in_channels, out_channels, stride, use_batchnorm, activation),
        *(
            ResidualBlock(out_channels, out_channels, 1, use_batchnorm, activation)
            for _ in range(blocks - 1)
        ),
    )


class CNN(nn.Module):
    """ResNet-style network: stem, one stage per entry of `widths`, global pooling, linear head.

    The defaults give the original 2.8M-parameter baseline.
    """

    def __init__(
        self,
        num_classes=len(LABELS),
        use_batchnorm=True,
        widths=(32, 64, 128, 256),
        blocks=(2, 2, 2, 2),
        activation="relu",
        dropout=0.2,
    ):
        super().__init__()
        assert len(widths) == len(blocks), "one block count per stage"
        act = ACTIVATIONS[activation]
        self.stem = nn.Sequential(
            *conv_bn(3, widths[0], stride=2, use_batchnorm=use_batchnorm),
            act(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # one attribute per stage so that per-stage gradient norms stay separate in the logs
        self.stage_names = []
        in_channels = widths[0]
        for i, (width, n_blocks) in enumerate(zip(widths, blocks)):
            name = f"stage{i + 1}"
            setattr(
                self,
                name,
                stage(in_channels, width, 1 if i == 0 else 2, use_batchnorm, n_blocks, act),
            )
            self.stage_names.append(name)
            in_channels = width
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def train(
    fold=0,
    max_epochs=10,
    batch_size=64,
    num_workers=4,
    use_batchnorm=True,
    model_kwargs=None,
    cache=False,
    size=224,
    run_name=None,
    tags=None,
    monitor="val_f1",
    ckpt_dir=None,
    callbacks=(),
    trainer_kwargs=None,
    **classifier_kwargs,
):
    """classifier_kwargs go to MultiLabelClassifier, e.g. optimizer="Adam", initial_lr=3e-4.

    model_kwargs go to CNN (widths, blocks, activation, dropout); cache=True reads images
    from the pre-resized .npy cache (src.data.build_cache) instead of decoding JPEGs;
    monitor is the metric that picks the best checkpoint; trainer_kwargs go to L.Trainer.
    """
    model_kwargs = model_kwargs or {}
    L.seed_everything(SEED, workers=True)
    tracking.setup()

    if cache:
        mean, std = fold_stats(fold, size)
    else:
        folds = pd.read_csv(FOLDS_PATH)
        mean, std = channel_stats(
            DATA_DIR / "train" / f"{i}.jpg" for i in folds[folds.fold != fold].image_id
        )
        mean, std = mean.round(4).tolist(), std.round(4).tolist()
    datamodule = DataModule(
        mean=mean,
        std=std,
        fold=fold,
        size=size,
        batch_size=batch_size,
        num_workers=num_workers,
        cache=cache,
    )
    model = MultiLabelClassifier(
        CNN(use_batchnorm=use_batchnorm, **model_kwargs), **classifier_kwargs
    )
    logger = MLFlowLogger(
        experiment_name=tracking.EXPERIMENT_NAME,
        tracking_uri=tracking.TRACKING_URI,
        run_name=run_name or f"cnn-fold{fold}",
        tags={"model": "CNN", "fold": str(fold), **(tags or {})},
    )
    logger.log_hyperparams(
        {
            "seed": SEED,
            "max_epochs": max_epochs,
            "use_batchnorm": use_batchnorm,
            **model_kwargs,
            "labels": LABELS,
            "num_parameters": sum(p.numel() for p in model.model.parameters()),
            "folds_sha256": hashlib.sha256(FOLDS_PATH.read_bytes()).hexdigest(),
        }
    )
    logger.experiment.log_text(logger.run_id, str(model.model), "architecture.txt")
    for path in (ROOT / "src").glob("*.py"):
        logger.experiment.log_artifact(logger.run_id, str(path), "source")

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir or ROOT / "models" / logger.run_id,
        filename="cnn-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        **{
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "devices": 1,
            "precision": "16-mixed" if torch.cuda.is_available() else "32-true",
            "logger": logger,
            "callbacks": [checkpoint, LearningRateMonitor(logging_interval="epoch"), *callbacks],
            "log_every_n_steps": 20,
            "num_sanity_val_steps": 0,
            **(trainer_kwargs or {}),
        }
    )
    try:
        trainer.fit(model, datamodule=datamodule)
    except BaseException:
        logger.finalize("failed")
        raise
    print(f"MLflow experiment: {tracking.EXPERIMENT_NAME}; run: {logger.run_id}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return trainer
