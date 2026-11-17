"""One JSON line per run (config, data, seed, results) in ``<DATA_ROOT>/history.jsonl``."""
from __future__ import annotations

import json
import os
import subprocess
import time

from .storage import DATA_ROOT, drive_guard, require_root


def git_sha() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def log_run(step: str, kind: str, config: dict, data: dict | None = None, seed: int | None = None,
            results: dict | None = None, root: str = DATA_ROOT, **extra) -> dict:
    """Append a run record. ``kind`` is e.g. "smoke", "eval", "train", "data"."""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "kind": kind, "git": git_sha(),
           "config": config, "data": data or {}, "seed": seed, "results": results or {}, **extra}
    require_root(root)
    with drive_guard(root), open(os.path.join(root, "history.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
