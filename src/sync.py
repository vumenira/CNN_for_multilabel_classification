import os
import shutil
from pathlib import Path

from lightning.pytorch.callbacks import Callback

from src import tracking
from src.colab import backup_sqlite


class SyncCheckpoints(Callback):
    def __init__(self, target):
        self.target = Path(target)
        self.synced_best = None

    def on_train_start(self, trainer, pl_module):
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "run_id.txt").write_text(trainer.logger.run_id)

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync(trainer)

    def on_train_end(self, trainer, pl_module):
        self._sync(trainer)

    def _sync(self, trainer):
        checkpoint = trainer.checkpoint_callback
        last = checkpoint.last_model_path
        if not last or not Path(last).exists():
            return
        self._copy(last, "last.ckpt")
        best = checkpoint.best_model_path
        if best and best != self.synced_best and Path(best).exists():
            self._copy(best, "best.ckpt")
            self.synced_best = best
        backup_sqlite(tracking.MLRUNS_DIR / "mlflow.db", self.target / "mlflow.db")
        print(f"[sync] {Path(last).name} and {Path(best).name} copied to {self.target}")

    def _copy(self, source, name):
        if not source or not Path(source).exists():
            return
        tmp = self.target / f"{name}.tmp"
        shutil.copy2(source, tmp)
        os.replace(tmp, self.target / name)
