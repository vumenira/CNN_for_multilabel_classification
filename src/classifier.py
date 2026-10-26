import lightning as L
import pandas as pd
import torch
from torch import nn
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

    def configure_optimizers(self):
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
