"""Per-request LoRA adapters for serving: the video adapter applies only to video requests.

One copy of the base model holds every adapter. Each wrapped linear layer reads the adapter
to use from a context variable, so the choice is scoped to the request that set it: another
thread or asyncio task scoring at the same time keeps its own setting, and nothing is ever
toggled process-wide. The choice is made when a layer is called (MLX builds the graph then),
so lazy evaluation later cannot pick up another request's adapter.

    model = load_model(...)
    install_adapter(model, "video", "checkpoints/step7/adapter")   # saved by save_adapter
    with request_scope(has_video=True):    # video request -> "video" adapter
        ...score...
    with request_scope(has_video=False):   # text / image request -> base weights
        ...score...
"""
from __future__ import annotations

import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from openjev.adapter_scope import VIDEO, active_adapter, adapter_scope, request_scope, run_scoped  # noqa: F401
from openjev.adapter_scope import _ACTIVE


class SwitchLoRALinear(nn.Module):
    """A frozen (possibly quantized) linear layer plus named LoRA deltas, chosen per call."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.adapters: dict[str, dict] = {}
        self.freeze()

    def add(self, name: str, lora_a: mx.array, lora_b: mx.array, scale: float):
        self.adapters[name] = {"a": lora_a, "b": lora_b, "scale": mx.array(scale, dtype=mx.float32)}
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        name = _ACTIVE.get()
        if name is None:
            return y
        ad = self.adapters.get(name)
        if ad is None:
            raise KeyError(f"adapter {name!r} is not installed on this model")
        return y + (ad["scale"] * ((x @ ad["a"]) @ ad["b"])).astype(x.dtype)


def save_adapter(model, path: str, rank: int, alpha: float, extra: dict | None = None) -> None:
    """Save the LoRA weights of a model prepared with ``mlx_model.attach_lora`` (training)."""
    os.makedirs(path, exist_ok=True)
    weights = {k: v for k, v in tree_flatten(model.trainable_parameters()) if k.endswith(("lora_a", "lora_b"))}
    if not weights:
        raise ValueError("model has no LoRA weights")
    mx.save_safetensors(os.path.join(path, "adapter.safetensors"), weights)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"rank": rank, "alpha": alpha, "scale": alpha / rank, "format": "openjev-lora-v1",
                   **(extra or {})}, f, indent=1)


def install_adapter(model, name: str, path: str) -> int:
    """Wrap the targeted linear layers (once) and register adapter ``name`` from ``path``.

    Returns the number of layers that received the adapter. The base weights are untouched.
    """
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    weights = mx.load(os.path.join(path, "adapter.safetensors"))
    mx.eval(weights)  # materialize now: lazy loads belong to this thread's stream
    layers = {}
    for key in weights:
        mod_path = key.rsplit(".", 1)[0]
        layers.setdefault(mod_path, {})[key.rsplit(".", 1)[1]] = weights[key]
    n = 0
    for mod_path, ab in layers.items():
        parent_path, attr = mod_path.rsplit(".", 1)
        parent = _get(model, parent_path)
        layer = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
        if not isinstance(layer, SwitchLoRALinear):
            layer = SwitchLoRALinear(layer)
            setattr(parent, attr, layer)
        layer.add(name, ab["lora_a"], ab["lora_b"], cfg["scale"])
        n += 1
    return n


def _get(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj
