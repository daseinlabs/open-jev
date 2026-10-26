"""Where decision models run: MLX on Apple silicon, PyTorch everywhere else (CUDA, ROCm, MPS, CPU).

Features are cached as plain arrays and heads are saved as safetensors with the same
parameter names on both backends, so a head trained on one backend loads on the other.

Selection order: `select(...)` (the CLI's --backend/--device), then the OPENJEV_BACKEND and
OPENJEV_DEVICE environment variables, then auto: MLX on Apple silicon when it is installed,
otherwise PyTorch on the first CUDA/ROCm GPU, then MPS, then CPU.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform

BACKENDS = ('auto', 'mlx', 'torch')
_selected = {'backend': None, 'device': None}


def select(backend: str | None = None, device: str | None = None) -> None:
    """Choose the backend (auto, mlx, torch) and PyTorch device (auto, cuda, cuda:N, mps, cpu)."""
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    _selected.update(backend=backend, device=device)


def backend() -> str:
    name = _selected['backend'] or os.environ.get('OPENJEV_BACKEND') or 'auto'
    if name not in BACKENDS:
        raise ValueError(f'OPENJEV_BACKEND must be one of {BACKENDS}')
    if name != 'auto':
        return name
    apple = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return 'mlx' if apple and importlib.util.find_spec('mlx') else 'torch'


def device() -> str:
    """The PyTorch device name (ignored by MLX, which always uses Metal)."""
    name = _selected['device'] or os.environ.get('OPENJEV_DEVICE') or 'auto'
    if name != 'auto':
        return name
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def ops(name: str | None = None):
    """The head module for a backend: AttentionHead, new_head, forward, Optimizer, snapshot, restore."""
    name = name or backend()
    if name == 'mlx':
        if not importlib.util.find_spec('mlx'):
            raise RuntimeError('MLX needs Apple silicon; use the torch backend (--backend torch or OPENJEV_BACKEND=torch)')
        return importlib.import_module('openjev.head')
    try:
        return importlib.import_module('openjev.head_torch')
    except ImportError as exc:
        raise RuntimeError('The torch backend needs PyTorch: pip install torch') from exc


def versions(name: str | None = None) -> dict:
    """Libraries whose versions change extracted features."""
    name = name or backend()
    packages = ('mlx', 'mlx-lm') if name == 'mlx' else ('torch',)
    result = {'backend': name}
    for package in (*packages, 'transformers', 'numpy'):
        result[package] = importlib.metadata.version(package)
    return result
