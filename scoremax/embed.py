import argparse
import time

import numpy as np
import torch

from scoremax.common import OUT, Images, autocast, create, device, ids, normalize, resize, to_float


@torch.no_grad()
def embed(model_name, size, batch_size, limit=None):
    dev = device()
    model, mean, std = create(model_name, size, num_classes=0)
    model = model.to(dev).eval()
    out = {}
    for split in ("train", "test"):
        n = len(ids(split)) if limit is None else limit
        loader = torch.utils.data.DataLoader(Images(split, np.arange(n)), batch_size=batch_size, num_workers=4)
        chunks = []
        for x in loader:
            x = normalize(resize(to_float(x.to(dev, non_blocking=True)), size), mean, std)
            feats = []
            for view in (x, x.flip(3)):
                with autocast(dev):
                    f = model.forward_features(view)
                    if f.ndim == 3:
                        prefix = getattr(model, "num_prefix_tokens", 1)
                        f = torch.cat([f[:, 0], f[:, prefix:].mean(1)], 1)
                    else:
                        f = model.forward_head(f, pre_logits=True)
                feats.append(f.float())
            chunks.append(((feats[0] + feats[1]) / 2).cpu().numpy().astype(np.float16))
        out[split] = np.concatenate(chunks)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name:size")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    target = OUT / "emb"
    target.mkdir(parents=True, exist_ok=True)
    for spec in a.models:
        name, size = spec.rsplit(":", 1)
        tag = f"{name.replace('/', '_')}_{size}"
        if (target / f"{tag}_test.npy").exists():
            print("skip", tag, flush=True)
            continue
        start = time.time()
        try:
            out = embed(name, int(size), a.bs, a.limit)
        except Exception as error:
            print(f"FAILED {tag}: {type(error).__name__}: {error}", flush=True)
            continue
        for split, array in out.items():
            np.save(target / f"{tag}_{split}.npy", array)
        print(f"done {tag} dim {out['train'].shape[1]} in {time.time() - start:.0f}s", flush=True)
