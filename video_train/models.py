"""Model weights live on the external drive; a run copies the ones it needs to the Mac.

    python -m video_train.models status
    python -m video_train.models stage gemma-4-E4B-it-mlx-8bit    # drive -> models/<name>
    python -m video_train.models release gemma-4-E4B-it-mlx-8bit  # delete the local copy

Store: ``<DATA_ROOT>/models/<name>``. Local copies: ``models/<name>`` (loading from the HDD
would take minutes per run). Scripts call :func:`local_model`, which stages a missing model
automatically; nothing deletes a local copy except ``release``.

``models/gemma-4-E4B-it-config`` is the tokenizer/config/processor files of Gemma 4 E4B
without weights (a few MB). It stays on the Mac: prompt encoding needs only these.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from . import storage

LOCAL = "models"
CONFIG = os.path.join(LOCAL, "gemma-4-E4B-it-config")  # tokenizer + config + processor, no weights


def store_path(name):
    return storage.path("models", name)


def _has_weights(d):
    return os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d))


def _size(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


def local_model(name: str) -> str:
    """``models/<name>``, copied from the drive first if it is not on the Mac."""
    dst = os.path.join(LOCAL, name)
    if _has_weights(dst):
        return dst
    storage.require_root()
    src = store_path(name)
    if not _has_weights(src):
        raise FileNotFoundError(f"model {name!r} is neither at {dst} nor at {src}")
    size = _size(src)
    free = shutil.disk_usage(LOCAL if os.path.isdir(LOCAL) else ".").free
    if size > free - 10e9:
        raise OSError(f"staging {name} needs {size / 1e9:.1f} GB; only {free / 1e9:.1f} GB free on the Mac "
                      "(keeping 10 GB spare). Release a model first: python -m video_train.models release <name>")
    print(f"staging {name} ({size / 1e9:.1f} GB) from the external drive, ~{size / 30e6 / 60:.0f} min ...",
          flush=True)
    tmp, t0 = dst + ".staging", time.time()
    shutil.rmtree(tmp, ignore_errors=True)
    with storage.drive_guard():
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(".cache", "._*"))
    if _size(tmp) != _size(src) - _size_ignored(src):
        raise OSError(f"incomplete copy of {name}; rerun to retry")
    os.replace(tmp, dst)
    print(f"staged {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dst


def _size_ignored(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs
               if "/.cache" in r or r.endswith(".cache") or f.startswith("._"))


def release(name: str) -> None:
    dst = os.path.join(LOCAL, name)
    if not _has_weights(dst):
        print(f"{name}: no local copy")
        return
    storage.require_root()
    if not _has_weights(store_path(name)):
        raise SystemExit(f"refusing to delete {dst}: no copy at {store_path(name)}")
    size = _size(dst)
    shutil.rmtree(dst)
    print(f"released {name} ({size / 1e9:.1f} GB freed)")


def status() -> None:
    names = set(os.listdir(LOCAL)) if os.path.isdir(LOCAL) else set()
    try:
        storage.require_root()
        if os.path.isdir(storage.path("models")):
            names |= set(os.listdir(storage.path("models")))
    except storage.DriveMissing:
        print("(external drive not available)")
    for n in sorted(x for x in names if not x.startswith(".")):
        loc, sto = os.path.join(LOCAL, n), store_path(n)
        print(f"{n:32s} mac: {'%.1f GB' % (_size(loc) / 1e9) if _has_weights(loc) else ('config only' if os.path.isdir(loc) else '-'):>12s}"
              f"   drive: {'%.1f GB' % (_size(sto) / 1e9) if _has_weights(sto) else '-':>8s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("status", "stage", "release"))
    ap.add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "status":
            status()
        for n in args.names:
            local_model(n) if args.cmd == "stage" else release(n)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
