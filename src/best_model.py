"""The final custom CNN (Optuna trial 28, trained 30 epochs on fold 0 as `final1-fold0`): rebuild, load, embed.

Used for adversarial validation and ensembling. The numbers below are exactly the config the run was trained with.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.classifier import MultiLabelClassifier
from src.cnn import CNN
from src.config import DATA_DIR
from src.data import ImageDataset, build_cache, eval_transforms, fold_stats, split_ids

FOLD, SIZE = 0, 224
ARCHITECTURE = dict(
    use_batchnorm=True, widths=[32, 64, 128, 256], blocks=[1, 1, 1, 1], activation="relu", dropout=0.19102524128543955
)  # fmt: skip


def load(checkpoint: str, device: str | None = None) -> CNN:
    """The trained network (best.ckpt of final1-fold0) in eval mode."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = MultiLabelClassifier.load_from_checkpoint(
        checkpoint, model=CNN(**ARCHITECTURE), map_location=device
    )
    return module.model.to(device).eval()


@torch.no_grad()
def embeddings(model: CNN, split: str, batch_size: int = 128, num_workers: int = 2) -> pd.DataFrame:
    """256-d globally pooled features (input of the classifier head), one row per image of `split` ("train" or "test").

    Images go through the eval pipeline with the mean/std the model was trained with (training part of fold 0).
    """
    mean, std = fold_stats(FOLD, SIZE)
    dataset = ImageDataset(
        pd.DataFrame({"image_id": split_ids(split)}),
        eval_transforms(SIZE, mean, std),
        DATA_DIR / split,
        build_cache(split, SIZE),
    )
    device = next(model.parameters()).device
    rows = []
    for images, _, _ in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        x = model.stem(images.to(device))
        for name in model.stage_names:
            x = getattr(model, name)(x)
        rows.append(model.pool(x).flatten(1).float().cpu())
    return pd.DataFrame(torch.cat(rows).numpy(), index=dataset.ids)
