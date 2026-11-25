import argparse

import numpy as np
import torch

from scoremax.blend import report, truth
from scoremax.common import LABELS, OUT


def fit_predict(x_train, y_train, x_eval, wd, epochs=300, lr=0.05, pos_weight=None):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y = torch.tensor(y_train, dtype=torch.float32, device=dev)
    head = torch.nn.Linear(x.shape[1], y.shape[1]).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y, pos_weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        return [head(torch.tensor(e, dtype=torch.float32, device=dev)).cpu().numpy() for e in x_eval]


def prepare(train, test):
    train, test = train.astype(np.float32), test.astype(np.float32)
    mean, std = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-6
    return (train - mean) / std, (test - mean) / std


def probe(tag, wd=0.01, root=OUT):
    folds, y = truth()
    train, test = prepare(np.load(root / "emb" / f"{tag}_train.npy"), np.load(root / "emb" / f"{tag}_test.npy"))
    oof = np.zeros(y.shape, np.float32)
    tests = []
    for k in sorted(folds.fold.unique()):
        val = (folds.fold == k).to_numpy()
        pos = y[~val].sum(0)
        weight = np.sqrt((len(y[~val]) - pos) / pos)
        oof[val], test_logits = fit_predict(train[~val], y[~val], [train[val], test], wd, pos_weight=weight)
        tests.append(test_logits)
    return oof, np.mean(tests, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--wd", type=float, nargs="+", default=[0.01])
    a = ap.parse_args()
    for tag in a.tags:
        for wd in a.wd:
            oof, test = probe(tag, wd)
            r = report(oof)
            run = OUT / f"probe_{tag}_wd{wd}"
            run.mkdir(parents=True, exist_ok=True)
            np.save(run / "oof.npy", oof)
            np.save(run / "test.npy", test)
            print(
                f"{tag} wd={wd} f1_macro@0.5 {r['f1_macro']:.4f} ap_named {r['ap_named']:.4f} "
                + " ".join(f"{l}: ap {r[f'ap_{l}']:.3f} fn {r[f'fn_{l}']} fp {r[f'fp_{l}']}" for l in LABELS),
                flush=True,
            )
