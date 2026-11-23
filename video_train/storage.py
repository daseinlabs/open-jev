"""Where the video-training data lives, and a loud failure when that drive goes away.

Everything bulky (videos, manifests, teacher labels, results, checkpoints) lives under
``DATA_ROOT``, on an external USB hard disk by default. Override it with
``OPENJEV_VIDEO_DATA``. Random-access feature caches go on the internal disk instead
(``LOCAL_CACHE``): the external disk is an HDD with ~18 ms per random read.
"""
from __future__ import annotations

import errno
import os
import subprocess
import sys
from contextlib import contextmanager

DATA_ROOT = os.environ.get("OPENJEV_VIDEO_DATA", "/Volumes/One Touch/open-jev-video-model-data")
LOCAL_CACHE = os.environ.get("OPENJEV_VIDEO_CACHE", os.path.expanduser("~/.cache/openjev-video"))


class DriveMissing(RuntimeError):
    pass


def _mount_point(path: str) -> str | None:
    """``/Volumes/<name>`` for a path on an external volume, else None."""
    parts = os.path.abspath(path).split(os.sep)
    return os.sep.join(parts[:3]) if len(parts) > 2 and parts[1] == "Volumes" else None


def require_root(root: str = DATA_ROOT) -> str:
    """Return ``root`` (created if needed), or raise DriveMissing if its volume is not mounted."""
    mp = _mount_point(root)
    if mp and not os.path.ismount(mp):
        raise DriveMissing(f"external drive {mp!r} is not mounted; reconnect it and rerun "
                           f"(data root: {root!r}, override with OPENJEV_VIDEO_DATA)")
    try:
        os.makedirs(root, exist_ok=True)
        probe = os.path.join(root, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise DriveMissing(f"data root {root!r} is not writable: {exc}") from exc
    return root


def path(*parts: str, root: str = DATA_ROOT) -> str:
    return os.path.join(root, *parts)


_DRIVE_ERRNOS = {errno.EIO, errno.ENXIO, errno.ENODEV, errno.ENOENT, errno.ESTALE, errno.EROFS}


@contextmanager
def drive_guard(root: str = DATA_ROOT):
    """Turn an I/O error caused by the drive disappearing into a clear DriveMissing error."""
    try:
        yield
    except OSError as exc:
        mp = _mount_point(root)
        if exc.errno in _DRIVE_ERRNOS and mp and not os.path.ismount(mp):
            raise DriveMissing(f"external drive {mp!r} disconnected during the run ({exc}); "
                               "reconnect it and rerun: the script resumes from its last checkpoint") from exc
        raise


def check_alive(root: str = DATA_ROOT) -> None:
    """Cheap per-iteration check for long loops."""
    mp = _mount_point(root)
    if mp and not os.path.ismount(mp):
        raise DriveMissing(f"external drive {mp!r} disconnected; reconnect it and rerun to resume")


def keep_awake() -> subprocess.Popen | None:
    """Hold off idle, disk and system sleep for this process's lifetime (macOS ``caffeinate``)."""
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
    except OSError:
        return None
