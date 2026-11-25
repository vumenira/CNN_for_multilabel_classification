import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from scoremax.common import DATA, FOLDS, LABELS, OUT

COMBOS = np.array(
    [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
    ]
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def truth():
    folds = pd.read_csv(FOLDS)
    return folds, folds[LABELS].to_numpy()


def test_ids():
    return pd.read_csv(DATA / "test.csv").image_id.tolist()


def collect(name, kind="ema", root=OUT):
    run = Path(root) / name
    folds, y = truth()
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    tests = []
    for k in sorted(folds.fold.unique()):
        val = run / f"fold{k}_val_{kind}.npy"
        if not val.exists():
            continue
        oof[np.load(run / f"fold{k}_val_rows.npy")] = np.load(val)
        tests.append(np.load(run / f"fold{k}_test_{kind}.npy"))
    for path in sorted(run.glob(f"full*_test_{kind}.npy")):
        tests.append(np.load(path))
    return oof, (np.mean(tests, 0) if tests else None)


def report(oof, y=None, thresholds=None):
    if y is None:
        y = truth()[1]
    done = ~np.isnan(oof).any(1)
    p, t = sigmoid(oof[done]), y[done]
    row = {"n": int(done.sum())}
    for c, label in enumerate(LABELS):
        row[f"ap_{label}"] = average_precision_score(t[:, c], p[:, c])
        thr = 0.5 if thresholds is None else thresholds[c]
        row[f"f1_{label}"] = f1_score(t[:, c], p[:, c] >= thr)
        pred = p[:, c] >= thr
        row[f"fn_{label}"] = int((~pred & (t[:, c] == 1)).sum())
        row[f"fp_{label}"] = int((pred & (t[:, c] == 0)).sum())
    row["f1_macro"] = float(np.mean([row[f"f1_{l}"] for l in LABELS]))
    row["ap_named"] = float(np.mean([row[f"ap_{l}"] for l in LABELS[2:]]))
    return row


def best_thresholds(oof, y=None, grid=np.round(np.arange(0.02, 0.981, 0.01), 2)):
    if y is None:
        y = truth()[1]
    done = ~np.isnan(oof).any(1)
    p, t = sigmoid(oof[done]), y[done]
    out, ranges = [], []
    for c in range(len(LABELS)):
        f1 = np.array([f1_score(t[:, c], p[:, c] >= g) for g in grid])
        tied = np.flatnonzero(f1 >= f1.max() - 1e-9)
        out.append(float(grid[tied[len(tied) // 2]]))
        ranges.append((float(grid[tied[0]]), float(grid[tied[-1]]), round(float(f1.max()), 4)))
    return np.array(out), ranges


def decode(logits, thresholds, combos=True):
    z = logits - logit(np.asarray(thresholds))[None]
    if combos:
        return COMBOS[(z @ COMBOS.T).argmax(1)]
    pred = (z >= 0).astype(int)
    pred[:, 0] |= pred[:, 2] | pred[:, 3]
    pred[:, 1] |= pred[:, 4]
    return pred


def macro_f1(pred, y):
    return float(np.mean([f1_score(y[:, c], pred[:, c]) for c in range(len(LABELS))]))


def write_submission(pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(pred.astype(int), columns=LABELS)
    frame.insert(0, "image_id", test_ids())
    sample = pd.read_csv(DATA / "sample_submission.csv")
    frame = frame.set_index("image_id").loc[sample.image_id].reset_index()
    frame.to_csv(path, index=False)
    return frame


def load_submission(path):
    return pd.read_csv(path).set_index("image_id").loc[test_ids(), LABELS].to_numpy()


def diff(a, b):
    changed = (a != b).any(1)
    delta = b.astype(int) - a.astype(int)
    return {
        "rows": int(changed.sum()),
        "added": dict(zip(LABELS, np.clip(delta, 0, None).sum(0).tolist())),
        "removed": dict(zip(LABELS, np.clip(-delta, 0, None).sum(0).tolist())),
    }


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
