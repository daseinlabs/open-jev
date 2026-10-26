"""Find the frozen backbone: a local directory, or a Hugging Face repo downloaded once into the HF cache.

Order: the explicit argument (`--model`), then $OPENJEV_MODEL, then DEFAULT_REPO. A trained
model records its backbone's content fingerprint, so a copy downloaded on another machine is
accepted as long as its files are identical.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO = 'parsecai/gemma-3-4b-it'
# Every file the fingerprint covers (decision.model_identity), and nothing else.
PATTERNS = ['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt']


def resolve(model: str | os.PathLike | None = None) -> str:
    """Return a local backbone directory, downloading `model` from Hugging Face if it is a repo id."""
    model = str(model or os.environ.get('OPENJEV_MODEL') or DEFAULT_REPO)
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path)
    if model.startswith(('.', '/', '~')) or model.count('/') != 1:
        raise ValueError(f'{model}: no such backbone directory')
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model, allow_patterns=PATTERNS)
    except Exception as exc:
        raise ValueError(f'{model}: not a local directory, and downloading it from Hugging Face failed ({exc}). '
                         f'If the repo is gated, accept its terms on huggingface.co and run `hf auth login`.') from exc
