"""jevlike's AttentionHead, ported to MLX: one cross-attention layer that scores each option
against the context tokens. Options are queries, context tokens are keys/values, and the
per-option logit is the dot product of the option query with its attended context vector.

`openjev.head_torch` is the same head in PyTorch with the same interface and weight names.
Both take and return numpy arrays at the boundary (`forward`, `Optimizer.step`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

NEG = -1e9
BACKEND = 'mlx'


class AttentionHead(nn.Module):
    backend = BACKEND

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.hidden, self.rank = hidden, rank
        self.ctx_norm = nn.LayerNorm(hidden)
        self.opt_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(hidden, rank, bias=False)
        self.value = nn.Linear(hidden, rank, bias=False)

    def __call__(self, ctx, ctx_mask, opt, opt_mask):
        """ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N) -> logits (B, N)."""
        ctx = self.ctx_norm(ctx.astype(mx.float32))
        opt = self.opt_norm(opt.astype(mx.float32))
        q = self.query(opt)                       # (B, N, r)
        k = self.key(ctx)                         # (B, Lc, r)
        v = self.value(ctx)
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.rank)   # (B, N, Lc)
        scores = mx.where(ctx_mask[:, None, :] > 0, scores, NEG)
        attn = mx.softmax(scores, axis=-1)
        attended = attn @ v                       # (B, N, r)
        logits = (q * attended).sum(-1) / math.sqrt(self.rank)       # (B, N)
        return mx.where(opt_mask > 0, logits, NEG)

    # ------------------------------------------------------------ persistence
    def save(self, path: str, config: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.save_weights(str(p))
        p.with_suffix(".json").write_text(json.dumps({"hidden": self.hidden, "rank": self.rank, **config}, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple["AttentionHead", dict]:
        cfg = json.loads(Path(path).with_suffix(".json").read_text())
        head = cls(cfg["hidden"], cfg["rank"])
        head.load_weights(path)
        head.eval()
        return head, cfg


def new_head(hidden: int, rank: int, seed: int) -> AttentionHead:
    mx.random.seed(seed)
    return AttentionHead(hidden, rank)


def parameter_count(head) -> int:
    return sum(v.size for _, v in tree_flatten(head.parameters()))


def forward(head, ctx, cm, opt, om) -> np.ndarray:
    """Logits (B, N) as float64 numpy, from numpy inputs."""
    logits = head(mx.array(ctx), mx.array(cm), mx.array(opt), mx.array(om))
    mx.eval(logits)
    return np.array(logits, dtype=np.float64)


def _loss(head, terms):
    """Sum over terms of scale * mean(weight * cross-entropy) (+ brier * Brier score)."""
    total = 0.
    for (ctx, cm, opt, om, labels), weights, scale, brier in terms:
        logits = head(ctx, cm, opt, om)
        ce = nn.losses.cross_entropy(logits, labels, reduction='none')
        loss = (ce if weights is None else ce * weights).mean()
        if brier:
            targets = mx.arange(logits.shape[-1])[None, :] == labels[:, None]
            loss = loss + brier * (((mx.softmax(logits, axis=-1) - targets) ** 2) * om).sum(-1).mean()
        total = total + scale * loss
    return total


class Optimizer:
    """AdamW on one head. `step(terms)` takes [(batch, weights or None, scale, brier_weight)],
    where batch is numpy (ctx, cm, opt, om, labels), and returns the loss."""

    def __init__(self, head, learning_rate, weight_decay=0., clip=None):
        self.head, self.clip = head, clip
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.grad = nn.value_and_grad(head, _loss)

    def step(self, terms) -> float:
        terms = [(tuple(mx.array(x) for x in batch), None if w is None else mx.array(w, dtype=mx.float32), scale, brier)
                 for batch, w, scale, brier in terms]
        loss, grads = self.grad(self.head, terms)
        if self.clip:
            grads, _ = optim.clip_grad_norm(grads, self.clip)
        self.optimizer.update(self.head, grads)
        mx.eval(self.head.parameters(), self.optimizer.state, loss)
        return loss.item()


def snapshot(head):
    return [(k, mx.array(v)) for k, v in tree_flatten(head.parameters())]


def restore(head, state) -> None:
    head.load_weights(state)
    mx.eval(head.parameters())
