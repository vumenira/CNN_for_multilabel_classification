"""Grad-CAM of the final CNN and of ConvNeXt-B on fold-0 validation photos, which neither model trained on.

The map is for one label's logit at the last stage of each network: final CNN stage4 (7×7 at 224 px),
ConvNeXt-B stages[3] (12×12 at 384 px). Photos are picked from the final CNN's validation probabilities:
its most confident correct photo per label, and its typical errors.

Run with `python -m src.gradcam`; figures go to docs/figures/.
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.best_model import ARCHITECTURE
from src.cnn import CNN
from src.config import DATA_DIR, FOLDS_PATH, LABELS, RESULTS_DIR, ROOT
from src.data import eval_transforms

FIGURES_DIR = ROOT / "docs" / "figures"
FINAL_CNN = dict(
    checkpoint=RESULTS_DIR / "final_model" / "best.ckpt",
    size=224,
    mean=[0.4889, 0.4623, 0.42],
    std=[0.265, 0.2573, 0.2631],
)
CONVNEXT = dict(
    name="convnext_base.fb_in22k_ft_in1k_384",
    checkpoint=ROOT / "models" / "b4bddc3f" / "convnext_base.fb_in22k_ft_in1k_384-epoch=08-val_f1=0.9990.ckpt",
    size=384,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
NAMED = ["Moris", "Motya", "Biatrix"]


def _weights(checkpoint) -> dict[str, torch.Tensor]:
    """The network's weights from a Lightning checkpoint of MultiLabelClassifier (tensors only, no pickled code)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    return {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}


def load_models() -> dict[str, tuple[torch.nn.Module, torch.nn.Module, dict]]:
    """name -> (network in eval mode, layer to explain, preprocessing)."""
    import timm

    cnn = CNN(**ARCHITECTURE)
    cnn.load_state_dict(_weights(FINAL_CNN["checkpoint"]))
    convnext = timm.create_model(CONVNEXT["name"], pretrained=False, num_classes=len(LABELS))
    convnext.load_state_dict(_weights(CONVNEXT["checkpoint"]))
    return {
        "Final CNN": (cnn.eval(), cnn.stage4, FINAL_CNN),
        "ConvNeXt-B": (convnext.eval(), convnext.stages[-1], CONVNEXT),
    }


class GradCAM:
    """Channel weights = spatial mean of d(logit)/d(activation); map = ReLU of the weighted sum of channels."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.activation = output
        output.register_hook(lambda grad: setattr(self, "gradient", grad))

    def __call__(self, x: torch.Tensor, label: int) -> tuple[np.ndarray, np.ndarray]:
        """Map in [0, 1] at the layer's resolution, and the probabilities of all labels."""
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, label].backward()
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activation).sum(1))[0].detach()
        return (cam / cam.max().clamp_min(1e-8)).numpy(), logits[0].detach().sigmoid().numpy()


def select_images() -> pd.DataFrame:
    """Fold-0 validation photos: final CNN's most confident hit per label, and its typical misses.

    Hits: Cat and Dog among photos with that species only, named animals among their positives.
    Errors: the most confident false positive of each named animal, its least confident named positive,
    and the most confident Cat/Dog on a photo with no labels.
    """
    probs = pd.read_csv(RESULTS_DIR / "final_model" / "val_probs.csv", index_col="image_id")
    folds = pd.read_csv(FOLDS_PATH).set_index("image_id")
    y = folds.loc[probs.index, LABELS]
    rows = []
    for label in LABELS:
        pool = y[label].eq(1)
        if label in ("Cat", "Dog"):
            pool &= y.sum(axis=1).eq(1)
        rows.append((probs[label][pool].idxmax(), label, "hit"))
    for label in NAMED:
        rows.append((probs[label][y[label].eq(0)].idxmax(), label, "false positive"))
    named = y[NAMED].stack()
    image_id, label = probs[NAMED].stack()[named.eq(1)].idxmin()
    rows.append((image_id, label, "false negative"))
    empty = probs[["Cat", "Dog"]][y.sum(axis=1).eq(0)]
    image_id, label = empty.stack().idxmax()
    rows.append((image_id, label, "false positive"))
    out = pd.DataFrame(rows, columns=["image_id", "label", "kind"])
    out["true_labels"] = [", ".join(l for l in LABELS if y.at[i, l]) or "none" for i in out.image_id]
    return out


def explain(images: pd.DataFrame, models) -> list[dict]:
    """Grad-CAM of every model for every (image, label) row."""
    cams = {name: GradCAM(model, layer) for name, (model, layer, _) in models.items()}
    results = []
    for row in images.itertuples():
        image = np.asarray(Image.open(DATA_DIR / "train" / f"{row.image_id}.jpg").convert("RGB"))
        out = {"row": row, "image": image}
        for name, (_, _, pre) in models.items():
            x = eval_transforms(pre["size"], pre["mean"], pre["std"])(image=image)["image"][None]
            cam, p = cams[name](x, LABELS.index(row.label))
            out[name] = (cv2.resize(cam, image.shape[1::-1], interpolation=cv2.INTER_LINEAR), p)
        results.append(out)
    return results


def plot(results: list[dict], model_names: list[str], path, title: str) -> None:
    """Columns are photos; rows: the photo, then each model's map for the row's label with its probability."""
    fig, axes = plt.subplots(1 + len(model_names), len(results), figsize=(2.6 * len(results), 2.3 * (1 + len(model_names))))
    for col, r in zip(axes.T, results):
        row = r["row"]
        col[0].imshow(r["image"])
        col[0].set_title(f"{row.kind}: {row.label}\ntrue: {row.true_labels}", fontsize=8)
        for ax, name in zip(col[1:], model_names):
            cam, p = r[name]
            ax.imshow(r["image"])
            ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"{name}: P({row.label}) = {p[LABELS.index(row.label)]:.2f}", fontsize=8)
        for ax in col:
            ax.axis("off")
    fig.suptitle(title, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    images = select_images()
    print(images.to_string())
    models = load_models()
    results = explain(images, models)
    # same pixels as in training, so the CNN's probabilities should repeat its exported val_probs.csv
    exported = pd.read_csv(RESULTS_DIR / "final_model" / "val_probs.csv", index_col="image_id")
    gap = max(np.abs(r["Final CNN"][1] - exported.loc[r["row"].image_id, LABELS]).max() for r in results)
    print(f"max |p - exported p| of the final CNN: {gap:.4f}")
    hits = [r for r in results if r["row"].kind == "hit"]
    misses = [r for r in results if r["row"].kind != "hit"]
    plot(hits, list(models), FIGURES_DIR / "gradcam_hits.png", "Grad-CAM: the final CNN's most confident correct photo per label")
    plot(misses, list(models), FIGURES_DIR / "gradcam_errors.png", "Grad-CAM: typical errors of the final CNN")
