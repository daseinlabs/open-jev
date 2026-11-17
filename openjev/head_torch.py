"""The AttentionHead in PyTorch (CUDA, ROCm, MPS, CPU), with the interface of `openjev.head`.

Weight names and shapes match the MLX head, so a head.safetensors saved by either loads in both.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from . import compute

NEG = -1e9
BACKEND = 'torch'


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

    @property
    def device(self):
        return self.query.weight.device

    def forward(self, ctx, ctx_mask, opt, opt_mask):
        """ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N) -> logits (B, N)."""
        ctx = self.ctx_norm(ctx.float())
        opt = self.opt_norm(opt.float())
        q = self.query(opt)
        k = self.key(ctx)
        v = self.value(ctx)
        scores = (q @ k.transpose(1, 2)) / math.sqrt(self.rank)
        scores = scores.masked_fill(ctx_mask[:, None, :] <= 0, NEG)
        attended = scores.softmax(-1) @ v
        logits = (q * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(opt_mask <= 0, NEG)

    def save(self, path: str, config: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.state_dict().items()}, str(p))
        p.with_suffix(".json").write_text(json.dumps({"hidden": self.hidden, "rank": self.rank, **config}, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple["AttentionHead", dict]:
        cfg = json.loads(Path(path).with_suffix(".json").read_text())
        head = cls(cfg["hidden"], cfg["rank"])
        head.load_state_dict(load_file(path))
        return head.to(compute.device()).eval(), cfg


def new_head(hidden: int, rank: int, seed: int) -> AttentionHead:
    torch.manual_seed(seed)
    return AttentionHead(hidden, rank).to(compute.device())


def parameter_count(head) -> int:
    return sum(p.numel() for p in head.parameters())


def _tensors(head, batch):
    return tuple(torch.as_tensor(np.asarray(x)).to(head.device) for x in batch)


def forward(head, ctx, cm, opt, om) -> np.ndarray:
    """Logits (B, N) as float64 numpy, from numpy inputs."""
    with torch.inference_mode():
        return head(*_tensors(head, (ctx, cm, opt, om))).float().cpu().numpy().astype(np.float64)


class _AdamW(torch.optim.Optimizer):
    """MLX's AdamW: decoupled weight decay and Adam without bias correction, so the
    learning rates tuned on MLX mean the same here (bias correction would make the first
    steps ~3x smaller)."""

    def __init__(self, params, lr, weight_decay=0., betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (b1, b2), lr = group['betas'], group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['m'], state['v'] = torch.zeros_like(p), torch.zeros_like(p)
                m, v = state['m'], state['v']
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - lr * group['weight_decay'])
                p.sub_(lr * m / (v.sqrt() + group['eps']))


class Optimizer:
    """AdamW on one head. `step(terms)` takes [(batch, weights or None, scale, brier_weight)],
    where batch is numpy (ctx, cm, opt, om, labels), and returns the loss."""

    def __init__(self, head, learning_rate, weight_decay=0., clip=None):
        self.head, self.clip = head, clip
        self.optimizer = _AdamW(head.parameters(), learning_rate, weight_decay)

    def step(self, terms) -> float:
        self.head.train()
        total = 0.
        for batch, weights, scale, brier in terms:
            ctx, cm, opt, om, labels = _tensors(self.head, batch)
            labels = labels.long()
            logits = self.head(ctx, cm, opt, om)
            ce = F.cross_entropy(logits, labels, reduction='none')
            loss = (ce if weights is None else ce * torch.as_tensor(weights, dtype=torch.float32, device=ce.device)).mean()
            if brier:
                targets = F.one_hot(labels, logits.shape[-1]).float()
                loss = loss + brier * (((logits.softmax(-1) - targets) ** 2) * om).sum(-1).mean()
            total = total + scale * loss
        self.optimizer.zero_grad()
        total.backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.head.parameters(), self.clip)
        self.optimizer.step()
        self.head.eval()
        return total.item()


def snapshot(head):
    return copy.deepcopy(head.state_dict())


def restore(head, state) -> None:
    head.load_state_dict(state)
