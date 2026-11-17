"""Stage 5 of docs/video-decision-model-plan.md: the game's episodes and closed-loop summary.

The game is a zero-shot test of the trained model (never trained on). Episodes
(``openworld.expert``): a random map, start and instruction (go to a landmark, cross the river,
get onto a bridge). The target can always be found from the start by turning on the spot (it may
start behind the player). The map-aware expert gives the reference path (SPL) and move
(agreement); the model never sees the map. The runner: ``vdm_cams``.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from . import storage
from .history import log_run

SPLIT_SEED = {"dev": 100_000, "test": 200_000}  # maps never shared between splits
MAX_STEPS, MIN_PATH = 150, 12.0


def episode_task(seed):
    """(world, start pose, task) for an episode seed, or None if its task is trivial, unreachable
    or its target cannot be seen from the start by turning."""
    from openworld import expert as E
    from openworld.world import generate
    w = generate(seed)
    nav = E.Nav.of(w)
    rng = np.random.default_rng(seed)
    x, z, h = E.spawn(w, rng, nav)
    task = E.make_task(w, rng, nav, x, z)
    left = E.remaining(task, x, z)
    if not np.isfinite(left) or left < MIN_PATH or task.done(w, x, z) or not E.findable(w, task, x, z):
        return None
    return w, (x, z, h), task


def episodes(split, n):
    """The first ``n`` usable episode seeds of a split."""
    seed = SPLIT_SEED[split]
    while n:
        if (ep := episode_task(seed)) is not None:
            yield seed, ep
            n -= 1
        seed += 1


def same_side(w, task, x, z):
    """A landmark task needs no river crossing (other kinds: always True)."""
    from openworld import expert as E
    if task.kind != "landmark":
        return True
    l = next(l for l in w.landmarks if l.name == task.target)
    return E.side(w, x, z) == E.side(w, l.x, l.z)


def landmark_episodes(split, n):
    """The first ``n`` landmark episodes of the split that need no river crossing."""
    fair = ((s, ep) for s, ep in episodes(split, 10 ** 6)
            if ep[2].kind == "landmark" and same_side(ep[0], ep[2], *ep[1][:2]))
    return itertools.islice(fair, n)


def angle_to(x, z, h, tx, tz):
    """Degrees from straight ahead to (tx, tz), positive = left."""
    from openworld import expert as E
    return (E.heading_to(x, z, tx, tz) - h + 180) % 360 - 180


def expert_steps(w, pose, task):
    from openworld import expert as E
    x, z, h = pose
    for s in range(MAX_STEPS):
        a = E.expert(w, task, x, z, h)
        if a == E.STOP:
            return s
        x, z, h = E.move(w, x, z, h, a)[-1]
    return MAX_STEPS


def summary(tag, split, path):
    rs = [json.loads(l) for l in open(path)]
    groups = {"all": rs, **{k: [r for r in rs if r["kind"] == k] for k in ("landmark", "cross", "bridge")}}
    mean = lambda v, k: float(np.mean([r[k] for r in v if r[k] is not None] or [np.nan]))
    res = {g: {"n": len(v), **{k: mean(v, k) for k in ("success", "reached", "stopped_early", "timed_out", "spl", "agree")}}
           for g, v in groups.items() if v}
    lines = [f"# Game, closed loop, zero-shot: {tag} ({split})", "",
             "| episodes | n | success | reached goal | stopped early | timed out | SPL | moves agree with expert |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {g} | {v['n']} | {v['success']:.0%} | {v['reached']:.0%} | {v['stopped_early']:.0%} | "
              f"{v['timed_out']:.0%} | {v['spl']:.2f} | {v['agree']:.0%} |" for g, v in res.items()]
    text = "\n".join(lines) + "\n"
    with storage.drive_guard(), open(os.path.splitext(path)[0] + ".md", "w") as f:
        f.write(text)
    print(text)
    log_run("vdm-5", "eval", {"what": f"game closed loop ({os.path.basename(path)})", "tag": tag, "split": split}, None, None, res)
