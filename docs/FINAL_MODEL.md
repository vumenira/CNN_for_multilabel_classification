# Final custom CNN: handover for ensemble and adversarial validation

The model is run `final1-fold0`: the best trial of the Optuna search (trial 28), trained for 30 epochs on **fold 0** of `data/folds.csv` (validation = fold 0, training = folds 1–4). Everything below describes only this model.

## Files (Drive: `dl_lab2/final_model/`)

| File | Content |
|---|---|
| `metrics_overall.csv` | validation macro AP, macro F1 at 0.5, macro F1 with tuned thresholds, exact match, best epoch, parameters, time per epoch |
| `metrics_per_class.csv` | per class: AP, F1 at 0.5, tuned threshold, number of test positives at 0.5 |
| `metrics_per_epoch.csv` | one row per epoch: `train_loss_epoch`, `val_loss`, `val_ap`, `val_f1`, `val_ap_<label>`, learning rate, gradient norm per module (`grad_norm/<module>_epoch`) |
| `curves.png` | the same curves as a picture |
| `config.json` | every hyper-parameter of the run |
| `val_probs.csv` | probabilities on the ~2,450 fold-0 validation images (never trained on) |
| `test_probs.csv` | probabilities on the 18,673 test images (`image_id` + 5 columns: Cat, Dog, Moris, Motya, Biatrix) |
| `submission_05.csv`, `submission_tuned.csv` | 0/1 labels at threshold 0.5 and at the per-class tuned thresholds, in the Kaggle format |
| `normalization_fold0.json` | mean/std the model was trained with |
| `best.ckpt` | weights of the best epoch (25) |

Headline validation numbers (fold 0, best epoch): macro AP 0.784, macro F1 0.584 at threshold 0.5 and 0.735 with per-class thresholds. Kaggle public score: _fill in_.
Validation has very few Moris/Motya/Biatrix positives, so the F1 values (especially the tuned thresholds) are noisy; AP is the more stable number. Selection was by validation macro AP.

## Architecture (1,227,685 parameters)

ResNet-style, implemented in `src/cnn.py` (`class CNN`), input 224×224×3:

```
stem      Conv3x3 s2 (3→32) + BN + ReLU, MaxPool 3x3 s2          224 → 56
stage1    1 residual block,  32 → 32   (stride 1)                  56
stage2    1 residual block,  32 → 64   (stride 2, 1x1 shortcut)    28
stage3    1 residual block,  64 → 128  (stride 2, 1x1 shortcut)    14
stage4    1 residual block, 128 → 256  (stride 2, 1x1 shortcut)     7
pool      global average pooling → 256 features
head      Dropout(0.191) → Linear(256 → 5)  → sigmoid per label
```

A residual block is `Conv3x3 + BN + ReLU + Conv3x3 + BN`, added to the shortcut, then ReLU. The baseline of the project had two blocks per stage (2.8 M parameters); the search found the shallow version better.

## How it was trained

- Optimiser AdamW, learning rate 3.47e-4 with cosine decay to 1e-5 over 30 epochs, weight decay 1.18e-3, batch size 64, fp16 mixed precision, seed 42.
- Loss: BCE with logits and `pos_weight = sqrt(negatives / positives)` per class (computed on the training folds).
- Augmentation: random resized crop (scale 0.9–1), horizontal flip, small affine, brightness/contrast, and one of JPEG / blur / motion blur / downscale / noise.
- Images were resized once to 224×224 and cached as a uint8 `.npy` (`src/data.py`, `build_cache`); evaluation uses exactly these cached pixels.
- Best epoch chosen by validation macro AP (`val_ap`).

## Code that is new or changed in this branch (`param-tuning`)

| File | What |
|---|---|
| `src/cnn.py` | `CNN` takes `widths`, `blocks`, `activation`, `dropout` (defaults = original baseline); `train()` takes `model_kwargs`, `cache`, `monitor`, … |
| `src/data.py` | image cache (`build_cache`), per-fold normalisation (`fold_stats`), `class_pos_weights` |
| `src/classifier.py` | validation AP metrics, `pos_weight` option |
| `src/sweep.py`, `src/results.py`, `src/colab.py` | search runner (ablations, Optuna, finals), evaluation/export, Colab helpers |
| `src/best_model.py` | **for you**: rebuilds exactly this network, loads `best.ckpt`, extracts 256-d embeddings |

## Using it

### Ensemble
Average `test_probs.csv` (and `val_probs.csv` for tuning thresholds) with the probabilities of the other models; all use the same columns and the same `image_id` index. The validation set is fold 0 for this model, so only weights fitted on fold 0 are honest for it.

### Adversarial validation (train vs test)
Run on a GPU/Colab with the repo, the dataset in `data/`, and `best.ckpt` uploaded:

```python
from src.best_model import load, embeddings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

model = load("best.ckpt")
train, test = embeddings(model, "train"), embeddings(model, "test")      # (12260, 256), (18673, 256)
X = pd.concat([train, test])
y = np.r_[np.zeros(len(train)), np.ones(len(test))]                      # 1 = test image
score = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5, method="predict_proba")[:, 1]
print("AUC", roc_auc_score(y, score))                                    # ~0.5 = same distribution, ~1 = shifted
test_likeness = pd.Series(score[: len(train)], index=train.index)        # high = train image that looks like test
```

AUC near 0.5 means train and test look alike, so validation scores should transfer. A high AUC points at a shift; the train images with the highest `test_likeness` are the ones to use as a validation set. Note that the CNN features were trained on folds 1–4, so for a cleaner check compute the features only for the fold-0 train images or use a raw-pixel/pretrained feature extractor instead. `embeddings` reads `data/cache/*.npy` (built automatically on first use, about a minute).
