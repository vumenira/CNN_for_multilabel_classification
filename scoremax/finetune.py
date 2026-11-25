import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

from scoremax.common import FOLDS, LABELS, OUT, Images, augment, autocast, create, device, ids, normalize, resize, to_float


def build_optimizer(model, lr, head_lr, weight_decay, layer_decay):
    from timm.optim import create_optimizer_v2

    optimizer = create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)
    head = {id(p) for p in model.get_classifier().parameters()}
    for group in optimizer.param_groups:
        is_head = any(id(p) in head for p in group["params"])
        group["base_lr"] = head_lr if is_head else lr * group.get("lr_scale", 1.0)
    return optimizer


def set_lr(optimizer, step, total, warmup):
    factor = (step + 1) / warmup if step < warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


@torch.no_grad()
def predict(model, dataset, size, mean, std, batch_size, dev):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    out = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = normalize(resize(to_float(x.to(dev, non_blocking=True)), size), mean, std)
        with autocast(dev):
            logits = (model(x).float() + model(x.flip(3)).float()) / 2
        out.append(logits.cpu())
    return torch.cat(out).numpy()


def scores(logits, y):
    p = 1 / (1 + np.exp(-logits))
    ap = [average_precision_score(y[:, c], p[:, c]) for c in range(len(LABELS))]
    f1 = [f1_score(y[:, c], p[:, c] >= 0.5, zero_division=0) for c in range(len(LABELS))]
    return {"ap": float(np.mean(ap)), "f1": float(np.mean(f1)), "ap_named": float(np.mean(ap[2:])), "f1_named": float(np.mean(f1[2:]))}


def run_fold(a, fold, folds, pseudo, run_dir):
    dev = device()
    torch.manual_seed(a.seed + fold)
    np.random.seed(a.seed + fold)
    y_all = folds[LABELS].to_numpy(np.float32)
    is_val = (folds.fold == fold).to_numpy() if fold >= 0 else np.zeros(len(folds), bool)
    train_rows = np.flatnonzero(~is_val)
    val_rows = np.flatnonzero(is_val)
    if a.limit:
        train_rows, val_rows = train_rows[: a.limit], val_rows[: a.limit // 4]
    named = y_all[train_rows, 2:].any(1)
    train_rows = np.concatenate([train_rows, np.repeat(train_rows[named], a.named_repeat - 1)])
    datasets = [Images("train", train_rows, y_all[train_rows])]
    if pseudo is not None:
        rows = np.repeat(pseudo["rows"], np.where(pseudo["labels"][:, 2:].any(1), a.named_repeat, 1))
        labels = np.repeat(pseudo["labels"], np.where(pseudo["labels"][:, 2:].any(1), a.named_repeat, 1), 0)
        datasets.append(Images("test", rows, labels))
    train_ds = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True, persistent_workers=True
    )
    model, mean, std = create(a.model, a.size, num_classes=len(LABELS), drop_path_rate=a.drop_path)
    model = model.to(dev)
    if dev.type == "cuda" and not hasattr(model, "pos_embed"):
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, a.lr, a.head_lr, a.wd, a.layer_decay)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda" and torch.cuda.get_device_capability()[0] < 8)
    ema = None
    if a.ema:
        from timm.utils import ModelEmaV3

        ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
    total = len(loader) * a.epochs
    warmup = int(a.warmup * len(loader))
    log, step = [], 0
    val_ds = Images("train", val_rows)
    for epoch in range(a.epochs):
        model.train()
        start, running = time.time(), 0.0
        for x, y in loader:
            set_lr(optimizer, step, total, warmup)
            x = to_float(x.to(dev, non_blocking=True))
            x = normalize(augment(x, a.size, scale=(a.scale, 1.0)), mean, std)
            y = y.to(dev)
            if a.smooth:
                y = y * (1 - a.smooth) + 0.5 * a.smooth
            with autocast(dev):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model, step=step)
            running += loss.item()
            step += 1
            if a.limit and step % 5 == 0:
                print(f"  step {step} loss {loss.item():.4f}", flush=True)
        row = {"epoch": epoch, "train_loss": running / len(loader), "seconds": round(time.time() - start)}
        if len(val_rows) and (epoch == a.epochs - 1 or a.val_every):
            row["raw"] = scores(predict(model, val_ds, a.size, mean, std, a.bs * 2, dev), y_all[val_rows])
            if ema is not None:
                row["ema"] = scores(predict(ema.module, val_ds, a.size, mean, std, a.bs * 2, dev), y_all[val_rows])
        log.append(row)
        print(f"fold {fold} {json.dumps(row)}", flush=True)
    test_ds = Images("test", np.arange(a.limit or len(ids("test"))))
    tag = f"fold{fold}" if fold >= 0 else f"full{a.seed}"
    for kind, net in (("raw", model), ("ema", ema.module if ema is not None else None)):
        if net is None:
            continue
        if len(val_rows):
            np.save(run_dir / f"{tag}_val_{kind}.npy", predict(net, val_ds, a.size, mean, std, a.bs * 2, dev))
        np.save(run_dir / f"{tag}_test_{kind}.npy", predict(net, test_ds, a.size, mean, std, a.bs * 2, dev))
    if a.save_weights:
        torch.save((ema.module if ema is not None else model).state_dict(), run_dir / f"{tag}.pt")
    (run_dir / f"{tag}_log.json").write_text(json.dumps(log, indent=1))
    np.save(run_dir / f"{tag}_val_rows.npy", val_rows)
    del model, ema, optimizer, loader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.8)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--named-repeat", type=int, default=3)
    ap.add_argument("--pseudo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--save-weights", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run_dir = OUT / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    folds = pd.read_csv(FOLDS)
    pseudo = None
    if a.pseudo:
        table = pd.read_csv(a.pseudo)
        row_of = {i: r for r, i in enumerate(ids("test"))}
        pseudo = {"rows": np.array([row_of[i] for i in table.image_id]), "labels": table[LABELS].to_numpy(np.float32)}
        print(f"pseudo labels: {len(table)} test photos, positives {table[LABELS].sum().to_dict()}", flush=True)
    for fold in [int(f) for f in a.folds.split(",")]:
        tag = f"fold{fold}" if fold >= 0 else f"full{a.seed}"
        if (run_dir / f"{tag}_log.json").exists():
            print("skip", tag, flush=True)
            continue
        start = time.time()
        run_fold(a, fold, folds, pseudo, run_dir)
        print(f"{tag} done in {time.time() - start:.0f}s", flush=True)
    print("ALL DONE", a.name, flush=True)
