"""Train and evaluate the head on cached Gemma features (jevlike's train.py / eval.py), on MLX or PyTorch."""
from __future__ import annotations

import json
import random
import time

import numpy as np

from . import compute
from .features import FeatureSet


def _log_softmax(logits):
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def evaluate(head, fs: FeatureSet, batch_size: int = 64, shuffle_context: bool = False) -> dict:
    """top1, top3, 10-bin ECE (on the max probability), n."""
    ops = compute.ops(head.backend)
    top1 = top3 = 0
    nll = brier = 0.0
    conf_bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, sum(conf), sum(correct)
    for s in range(0, len(fs), batch_size):
        idx = list(range(s, min(s + batch_size, len(fs))))
        ctx, cm, opt, om, labels = fs.batch(idx, shuffle_context=shuffle_context)
        logp = _log_softmax(ops.forward(head, ctx, cm, opt, om))
        probs = np.exp(logp)
        targets = np.arange(probs.shape[-1])[None, :] == labels[:, None]
        nll -= logp[np.arange(len(idx)), labels].sum()
        brier += (((probs - targets) ** 2) * om).sum()
        for o, p, y in zip(np.argsort(-probs, axis=-1), probs.max(-1), labels):
            hit = o[0] == y
            top1 += hit
            top3 += y in o[:3]
            b = min(9, int(p * 10))
            conf_bins[b][0] += 1
            conf_bins[b][1] += p
            conf_bins[b][2] += hit
    n = len(fs)
    ece = sum(c * abs(sc / c - sh / c) for c, sc, sh in conf_bins if c) / n
    return {"top1": float(top1 / n), "top3": float(top3 / n), "ece": float(ece), "nll": float(nll / n),
            "brier": float(brier / n), "examples": n}


def train(train_path: str, val_path: str, out: str, rank: int = 256, epochs: int = 8, batch_size: int = 64,
          lr: float = 5e-4, weight_decay: float = 1e-4, seed: int = 7,
          brier_weight: float = 0.0, selection: str = "top1") -> dict:
    import math
    if not math.isfinite(brier_weight) or brier_weight < 0 or selection not in ('top1', 'nll', 'brier'):
        raise ValueError('invalid Brier weight or checkpoint selection metric')
    ops = compute.ops()
    random.seed(seed)
    tr, va = FeatureSet(train_path), FeatureSet(val_path)
    head = ops.new_head(tr.meta["hidden"], rank, seed)
    print(f"train {len(tr)} / val {len(va)} examples, hidden {tr.meta['hidden']}, rank {rank}, "
          f"{ops.parameter_count(head) / 1e6:.2f}M head params, backend {head.backend}", flush=True)
    optimizer = ops.Optimizer(head, lr, weight_decay=weight_decay, clip=1.0)
    best, best_state, history = -float('inf'), None, []
    selected_top1 = None
    idx_all = list(range(len(tr)))
    for epoch in range(1, epochs + 1):
        t = time.perf_counter()
        random.shuffle(idx_all)
        total = 0.0
        for s in range(0, len(tr), batch_size):
            batch = tr.batch(idx_all[s : s + batch_size])
            total += optimizer.step([(batch, None, 1.0, brier_weight)]) * len(batch[-1])
        val = evaluate(head, va, batch_size)
        rec = {"epoch": epoch, "train_loss": total / len(tr), **{f"val_{k}": v for k, v in val.items() if k != "examples"},
               "seconds": round(time.perf_counter() - t, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        metric = val[selection] if selection == 'top1' else -val[selection]
        if metric > best:
            best = metric
            selected_top1 = val['top1']
            best_state = ops.snapshot(head)
    ops.restore(head, best_state)
    head.save(out, {"train": train_path, "validation": val_path, "epochs": epochs, "lr": lr,
                    "batch_size": batch_size, "seed": seed, "best_val_top1": selected_top1,
                    "brier_weight": brier_weight, "brier_convention": "sum_over_valid_candidates",
                    "selection": selection, "backend": head.backend,
                    "features_meta": tr.meta, "history": history})
    return {"best_val_top1": selected_top1, "checkpoint": out}
