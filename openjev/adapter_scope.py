"""Which LoRA adapter a request uses, scoped to that request.

The choice is a context variable: each thread and asyncio task sees its own value, and
FastAPI copies it into the worker thread that runs a sync endpoint. A plain
``ThreadPoolExecutor.submit`` does **not** copy it, so any handoff to another thread must
go through :func:`run_scoped` (or ``contextvars.copy_context().run``), otherwise the work
silently runs on the base weights.

No ML imports here: the server uses this on every backend. ``video_train.adapters`` holds
the MLX layers that read the choice.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager

VIDEO = "video"
_ACTIVE: contextvars.ContextVar[str | None] = contextvars.ContextVar("openjev_adapter", default=None)


@contextmanager
def adapter_scope(name: str | None):
    """Use adapter ``name`` (None: base weights) for model calls made inside this block."""
    token = _ACTIVE.set(name)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def request_scope(has_video: bool):
    """The video adapter for requests with video input, the base model for everything else."""
    return adapter_scope(VIDEO if has_video else None)


def active_adapter() -> str | None:
    return _ACTIVE.get()


def run_scoped(has_video: bool, fn, *args, executor=None, **kwargs):
    """Run ``fn`` under the request's adapter, optionally on ``executor``, carrying the scope across."""
    with request_scope(has_video):
        if executor is None:
            return fn(*args, **kwargs)
        ctx = contextvars.copy_context()
        return executor.submit(ctx.run, fn, *args, **kwargs).result()
