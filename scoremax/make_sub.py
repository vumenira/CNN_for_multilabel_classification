import argparse
import json

import numpy as np
import pandas as pd

from scoremax.blend import COMBOS, best_thresholds, collect, decode, diff, load_submission, logit, macro_f1, report, sigmoid, test_ids, truth, write_submission
from scoremax.common import LABELS, OUT, ROOT

BEST = ROOT / "data" / "pretrained_model_results" / "submission_combos.csv"


def load(spec):
    name, _, weight = spec.partition(":")
    kind = "ema"
    if "@" in name:
        name, kind = name.split("@")
    run = OUT / name
    if (run / "oof.npy").exists():
        oof, test = np.load(run / "oof.npy"), np.load(run / "test.npy")
    else:
        oof, test = collect(name, kind)
    return name, float(weight or 1), oof, test


def ensemble(specs, space="logit"):
    runs = [load(s) for s in specs]
    total = sum(w for _, w, _, _ in runs)
    if space == "logit":
        oof = sum(w * o for _, w, o, _ in runs) / total
        test = sum(w * t for _, w, _, t in runs) / total
    else:
        oof = logit(sum(w * sigmoid(o) for _, w, o, _ in runs) / total)
        test = logit(sum(w * sigmoid(t) for _, w, _, t in runs) / total)
    return runs, oof, test


def line(name, oof, y):
    r = report(oof, y)
    thr, _ = best_thresholds(oof, y)
    done = ~np.isnan(oof).any(1)
    tuned = macro_f1(decode(oof[done], thr), y[done])
    half = macro_f1(decode(oof[done], [0.5] * 5), y[done])
    errors = " ".join(f"{l[:3]} {r[f'fn_{l}']}/{r[f'fp_{l}']}" for l in LABELS)
    return f"{name:34s} n {r['n']:5d} ap_named {r['ap_named']:.4f} f1@.5 {r['f1_macro']:.4f} combos@.5 {half:.4f} combos@tuned {tuned:.4f} | fn/fp@.5 {errors}"


def describe(pred):
    both = int((pred[:, 2] & pred[:, 3]).sum())
    return f"positives {dict(zip(LABELS, pred.sum(0).tolist()))} Moris&Motya {both}"


def pseudo(test, lo, hi, others, share=0.8, seed=0):
    p = sigmoid(test)
    confident = ((p <= lo) | (p >= hi)).all(1)
    pred = decode(test, [0.5] * 5)
    named = np.zeros(len(p), bool)
    for c in (2, 3, 4):
        quota = int(share * p[:, c].sum())
        ranked = np.argsort(-p[:, c])[:quota]
        named[ranked[confident[ranked] & (pred[ranked, c] == 1)]] = True
    rest = np.flatnonzero(confident & ~pred[:, 2:].any(1) & (p[:, 2:].max(1) < 0.01))
    rest = np.random.default_rng(seed).choice(rest, min(others, len(rest)), replace=False)
    rows = np.concatenate([np.flatnonzero(named), rest])
    frame = pd.DataFrame(pred[rows], columns=LABELS)
    frame.insert(0, "image_id", np.array(test_ids())[rows])
    return frame, int(confident.sum()), int(named.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--space", default="logit")
    ap.add_argument("--tag")
    ap.add_argument("--thr", type=float, nargs=5)
    ap.add_argument("--thr-scale", type=float, default=1.0)
    ap.add_argument("--pseudo")
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--others", type=int, default=1500)
    a = ap.parse_args()
    _, y = truth()
    runs, oof, test = ensemble(a.runs, a.space)
    for name, _, o, _ in runs:
        print(line(name, o, y))
    if len(runs) > 1:
        print(line("ENSEMBLE", oof, y))
    thr, ranges = best_thresholds(oof, y)
    print("OOF thresholds", dict(zip(LABELS, thr.tolist())), "ranges", ranges)
    use = np.array(a.thr) if a.thr else np.clip(thr * a.thr_scale, 0.02, 0.98)
    best = load_submission(BEST)
    p = sigmoid(test)
    gray = ((p >= 0.05) & (p < 0.95)).sum(0)
    print("test gray zone 0.05-0.95", dict(zip(LABELS, gray.tolist())))
    for label, t in (("@0.5", [0.5] * 5), ("@used", use)):
        pred = decode(test, t)
        print(f"test {label} {np.round(t, 2).tolist()} {describe(pred)} vs best {json.dumps(diff(best, pred))}")
    if a.tag:
        path = OUT / "subs" / f"{a.tag}.csv"
        write_submission(decode(test, use), path)
        np.save(OUT / "subs" / f"{a.tag}_test_logits.npy", test)
        print("saved", path)
    if a.pseudo:
        frame, confident, named = pseudo(test, a.lo, a.hi, a.others)
        frame.to_csv(a.pseudo, index=False)
        print(f"pseudo: {confident} confident test rows, {named} named; wrote {len(frame)} rows, positives {frame[LABELS].sum().to_dict()}")
