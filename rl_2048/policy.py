"""Gemma 3 4B as a 2048 policy, scored the openjev way, with trainable LoRA.

The policy is openjev's option scoring: the board goes in as the context, the
four moves are the options, and a softmax over each option's log-probability
(openjev ``norm="sum"``) gives a probability per move. All four moves are
single tokens, so one forward pass scores them all; this matches
``OptionScorer.score(prompt, ACTIONS, norm="sum", chat=True)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

from openjev.scorer import DEFAULT_MODEL, OptionScorer

from .game import ACTIONS, render

PROMPT = """You are playing 2048 on a 4x4 board. A move slides every tile up, down, left or right. When two equal tiles collide they merge into one tile with their sum, and you earn that many points. After each move a new 2 or 4 appears in an empty cell. The game ends when no move changes the board. Keep the board open and build large tiles.

Board ('.' is empty):
{board}

Valid moves: {valid}
Reply with one word: up, down, left or right."""


def prompt_for(board: list[int], valid: list[str]) -> str:
    return PROMPT.format(board=render(board), valid=", ".join(valid))


class GemmaPolicy:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        lora_layers: int = 8,
        lora_rank: int = 8,
        lora_scale: float = 20.0,
        temperature: float = 1.0,
    ) -> None:
        self.temperature = temperature
        # Reuse openjev's loader and chat-template handling so prompts match the server.
        self.scorer = OptionScorer(model_path, chat=True, backend="mlx")
        self.model, self.tok = self.scorer.model, self.scorer.tok
        self.action_ids = []
        for a in ACTIONS:
            ids = self.scorer.option_ids(a)
            if len(ids) != 1:
                raise RuntimeError(f"expected {a!r} to be one token, got {ids}")
            self.action_ids.append(ids[0])
        self.action_ids = mx.array(self.action_ids)

        self.model.freeze()
        if adapter_path:
            cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            lora_layers, p = cfg["num_layers"], cfg["lora_parameters"]
            lora_rank, lora_scale = p["rank"], p["scale"]
        self.lora_config = {
            "fine_tune_type": "lora",
            "num_layers": lora_layers,
            "lora_parameters": {"rank": lora_rank, "scale": lora_scale, "dropout": 0.0},
        }
        linear_to_lora_layers(self.model, lora_layers, self.lora_config["lora_parameters"])
        if adapter_path:
            self.model.load_weights(str(Path(adapter_path) / "adapters.safetensors"), strict=False)
        self._lora = [m for _, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        self._scales = [m.scale for m in self._lora]

    def encode(self, board: list[int], valid: list[str]) -> mx.array:
        return mx.array(self.scorer.context_ids(prompt_for(board, valid)))

    def log_probs(self, model: nn.Module, ids: mx.array) -> mx.array:
        """Log-probability of each move (shape 4). Differentiable w.r.t. LoRA weights.

        ``temperature`` > 1 flattens the distribution. The base model is nearly
        certain about every board, which would make gradients vanish; a softer
        policy explores and learns. The greedy choice (argmax) is unaffected.
        """
        logits = model(ids[None])[0, -1].astype(mx.float32)
        scores = logits[self.action_ids] - mx.logsumexp(logits)  # openjev norm="sum" scores
        scores = scores / self.temperature
        return scores - mx.logsumexp(scores)

    def probs(self, ids: mx.array) -> mx.array:
        p = mx.exp(self.log_probs(self.model, ids))
        mx.eval(p)
        return p

    def reference_probs(self, ids: mx.array) -> mx.array:
        """Move probabilities of the original model: LoRA switched off for one pass."""
        for m in self._lora:
            m.scale = 0.0
        try:
            return self.probs(ids)
        finally:
            for m, s in zip(self._lora, self._scales):
                m.scale = s

    def save(self, path: str | Path) -> None:
        """Write an mlx_lm-compatible adapter (loadable by OptionScorer(adapter_path=...))."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path / "adapters.safetensors"),
                            dict(tree_flatten(self.model.trainable_parameters())))
        (path / "adapter_config.json").write_text(json.dumps(self.lora_config, indent=2))
