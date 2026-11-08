import math

import lightning as L
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

from src.config import LABELS


class MultiLabelClassifier(L.LightningModule):
    def __init__(
        self,
        model,
        optimizer="AdamW",
        scheduler="cosine",
        initial_lr=1e-3,
        final_lr=1e-5,
        weight_decay=1e-4,
        pos_weight=None,
        head_epochs=0,
        head_lr=None,
        warmup_steps=0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)
        )
        self.train_f1 = MultilabelF1Score(num_labels=len(LABELS), average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=len(LABELS), average=None)
        self.val_ap = MultilabelAveragePrecision(num_labels=len(LABELS), average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.train_f1.update(logits.float().sigmoid(), targets.int())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        probs = logits.float().sigmoid()
        self.val_f1.update(probs, targets.int())
        self.val_ap.update(probs, targets.int())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        self.log(
            "val_bce",
            F.binary_cross_entropy_with_logits(logits.float(), targets),
            on_step=False,
            on_epoch=True,
            batch_size=len(images),
        )

    def on_validation_epoch_end(self):
        scores = self.val_f1.compute()
        self.log("val_f1", scores.mean(), prog_bar=True, batch_size=1)
        self.log_dict(
            {f"val_f1_{label}": score for label, score in zip(LABELS, scores)},
            batch_size=1,
        )
        self.val_f1.reset()
        ap = self.val_ap.compute()
        self.log("val_ap", ap.mean(), prog_bar=True, batch_size=1)
        self.log_dict(
            {f"val_ap_{label}": score for label, score in zip(LABELS, ap)},
            batch_size=1,
        )
        self.val_ap.reset()

    def predict_step(self, batch, batch_idx):
        images, _, ids = batch
        return {"image_id": ids, "probabilities": self(images).float().sigmoid()}

    def on_before_optimizer_step(self, optimizer):
        for name, module in self.model.named_children():
            norms = [
                p.grad.detach().norm()
                for p in module.parameters()
                if p.grad is not None
            ]
            if norms:
                self.log(
                    f"grad_norm/{name}",
                    torch.stack(norms).norm(),
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                )

    @property
    def staged(self):
        h = self.hparams
        return h.head_epochs > 0 or h.warmup_steps > 0 or h.head_lr is not None

    def split_parameters(self):
        head = list(self.model.get_classifier().parameters())
        head_ids = {id(p) for p in head}
        return head, [p for p in self.model.parameters() if id(p) not in head_ids]

    def on_train_epoch_start(self):
        if self.hparams.head_epochs > 0:
            trainable = self.current_epoch >= self.hparams.head_epochs
            for p in self.split_parameters()[1]:
                p.requires_grad_(trainable)

    def lr_lambdas(self):
        h = self.hparams
        total = self.trainer.estimated_stepping_batches
        head_steps = h.head_epochs * total // self.trainer.max_epochs

        def cosine(progress, base_lr):
            floor = h.final_lr / base_lr
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        def head(step):
            return cosine(step / total, h.head_lr or h.initial_lr)

        def backbone(step):
            if step < head_steps:
                return 0.0
            step -= head_steps
            if step < h.warmup_steps:
                return (step + 1) / h.warmup_steps
            decay_steps = max(1, total - head_steps - h.warmup_steps)
            return cosine((step - h.warmup_steps) / decay_steps, h.initial_lr)

        return [head, backbone]

    def configure_staged_optimizers(self):
        head, backbone = self.split_parameters()
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            [
                {"params": head, "lr": self.hparams.head_lr or self.hparams.initial_lr, "name": "head"},
                {"params": backbone, "lr": self.hparams.initial_lr, "name": "backbone"},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambdas())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_optimizers(self):
        if self.staged:
            return self.configure_staged_optimizers()
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "constant":
            return optimizer
        if self.hparams.scheduler != "cosine":
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def predict(trainer, **kwargs) -> pd.DataFrame:
    """Probabilities of the trained model, one row per image_id."""
    batches = trainer.predict(trainer.lightning_module, **kwargs)
    return pd.DataFrame(
        torch.cat([b["probabilities"] for b in batches]).numpy(),
        index=[i for b in batches for i in b["image_id"]],
        columns=LABELS,
    )
