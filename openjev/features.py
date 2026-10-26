"""Frozen-Gemma feature extraction for the trainable head (Route A).

Stops Gemma before the LM head and returns final-layer hidden states. The context is encoded
once (all token vectors kept). Options are encoded on their own, batched, and masked-mean
pooled to one vector each, exactly as jevlike does: the head must do the matching, and the
shuffled-context control stays meaningful. With ``contextual=True`` the options are instead
run as continuations of the context via the prefix-shared cache; the pooled vectors then
already contain the match, which makes the head trivial and the control useless.
Features are cached to an .npz so head training never touches Gemma again.

Runs on MLX or, through the scorer's PyTorch engine, on CUDA/ROCm/MPS/CPU (independent options only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .scorer import OptionScorer, iter_jsonl


class FeatureExtractor:
    def __init__(self, scorer: OptionScorer, contextual: bool = False) -> None:
        self.s = scorer
        self.contextual = contextual
        if scorer.backend == "torch":
            if contextual:
                raise ValueError("contextual features need the MLX backend")
            self.hidden = scorer._engine.hidden_size
            return
        global mx
        import mlx.core as mx
        m = scorer.model
        inner = m.language_model if hasattr(m, "language_model") else m
        self.core = inner.model  # Gemma3Model: embeddings + layers + final norm, no LM head
        self.hidden = inner.args.hidden_size

    def extract(self, context: str, options: list[str], chat: bool | None = None, sep: str | None = None):
        """Return (context_h (Lc, H) float16, options_h (N, H) float16)."""
        ctx_ids = self.s.context_ids(context, chat=chat, sep=sep)
        opts = [self.s.option_ids(o) for o in options]
        if self.s.backend == "torch":
            return self.s._engine.features(ctx_ids, opts, self.s.bos_id, self.s.pad_id, self.s.batch_size)
        from mlx_lm.models.cache import KVCache

        cache = [KVCache() for _ in self.s.model.layers]
        ctx_h = self.core(mx.array(ctx_ids)[None], cache=cache)[0]  # (Lc, H)
        mx.eval(ctx_h, *[c.keys for c in cache], *[c.values for c in cache])
        if not self.contextual:  # options stand alone: prefix is just BOS
            cache = [KVCache() for _ in self.s.model.layers]
            mx.eval(self.core(mx.array([self.s.bos_id])[None], cache=cache))

        pooled = []
        for start in range(0, len(opts), self.s.batch_size):
            chunk = opts[start : start + self.s.batch_size]
            n, L = len(chunk), max(len(x) for x in chunk)
            arr = mx.array([x + [self.s.pad_id] * (L - len(x)) for x in chunk])
            mask = mx.array([[1.0] * len(x) + [0.0] * (L - len(x)) for x in chunk])[..., None]
            h = self.core(arr, cache=self.s._expand(cache, n))  # (n, L, H)
            mean = (h.astype(mx.float32) * mask).sum(1) / mask.sum(1)
            mx.eval(mean)
            pooled.append(np.array(mean, dtype=np.float16))
        return np.array(ctx_h.astype(mx.float16)), np.concatenate(pooled, 0)


def extract_dataset(scorer: OptionScorer, data: str, out: str, limit: int = 0, chat: bool = False, sep: str = "",
                    contextual: bool = False) -> dict:
    """Cache features for a jevlike JSONL {context, options, label} file into one .npz."""
    fx = FeatureExtractor(scorer, contextual=contextual)
    rows = list(iter_jsonl(data))
    if limit:
        rows = rows[:limit]
    arrays: dict[str, np.ndarray] = {}
    labels = []
    t = time.perf_counter()
    for i, row in enumerate(rows):
        c, o = fx.extract(row["context"], row["options"], chat=chat, sep=sep)
        arrays[f"ctx_{i}"] = c
        arrays[f"opt_{i}"] = o
        labels.append(int(row.get("label", -1)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} examples, {(time.perf_counter() - t) / (i + 1) * 1000:.0f} ms each", flush=True)
    arrays["labels"] = np.array(labels, dtype=np.int32)
    meta = {"n": len(rows), "hidden": fx.hidden, "source": data, "chat": chat, "sep": sep, "contextual": contextual,
            "seconds": round(time.perf_counter() - t, 1)}
    arrays["meta"] = np.array(json.dumps(meta))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return meta


class FeatureSet:
    """In-memory view of a cached .npz: lists of (Lc, H) contexts, (N, H) options, labels."""

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.labels = z["labels"]
        self.ctx = [z[f"ctx_{i}"] for i in range(len(self.labels))]
        self.opt = [z[f"opt_{i}"] for i in range(len(self.labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: list[int], shuffle_context: bool = False):
        """Pad a batch as numpy arrays: ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N), labels (B,)."""
        ctx_src = [self.ctx[i] for i in idx]
        if shuffle_context:  # jevlike's control: every example sees another example's context
            ctx_src = ctx_src[1:] + ctx_src[:1]
        H = self.meta["hidden"]
        Lc = max(c.shape[0] for c in ctx_src)
        N = max(self.opt[i].shape[0] for i in idx)
        B = len(idx)
        ctx = np.zeros((B, Lc, H), np.float16)
        cm = np.zeros((B, Lc), np.float32)
        opt = np.zeros((B, N, H), np.float16)
        om = np.zeros((B, N), np.float32)
        for b, (c, i) in enumerate(zip(ctx_src, idx)):
            ctx[b, : c.shape[0]] = c
            cm[b, : c.shape[0]] = 1
            o = self.opt[i]
            opt[b, : o.shape[0]] = o
            om[b, : o.shape[0]] = 1
        return ctx, cm, opt, om, self.labels[idx]
