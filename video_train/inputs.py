"""Gemma 4 prompt inputs built by OpenJev's own (processor-verified) code, without loading weights.

Training and evaluation both go through here, so the MLX model sees exactly the token layout
and pixels that ``openjev.torch_backend`` scores with (``tests/test_gemma4_processor_parity.py``).
"""
from __future__ import annotations

import types
from dataclasses import dataclass

import numpy as np

LETTERS = list("ABCDEFGH")
LABELS = LETTERS[:3]  # Perception Test: 3 options


@dataclass
class Encoded:
    input_ids: list[int]
    pixel_values: np.ndarray  # (frames, patches, 768)
    position_ids: np.ndarray  # (frames, patches, 2)
    prefix_len: int           # tokens up to and including the last frame's end marker


class Gemma4Inputs:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from openjev.scorer import OptionScorer
        from openjev.torch_backend import TorchBackend

        self.cfg, self.tok = AutoConfig.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)
        b = TorchBackend.__new__(TorchBackend)
        b.torch, b.tok, b.model = torch, self.tok, types.SimpleNamespace(config=self.cfg)
        b.frame_mode, b.frame_soft_tokens, b.model_path, b._processor = "images", None, model_path, None
        s = OptionScorer.__new__(OptionScorer)
        s._engine, s.tok, s.chat, s.sep, s.bos_id = b, self.tok, False, "", self.tok.bos_token_id
        self._backend, self._scorer = b, s
        ids = [self.tok.encode(l, add_special_tokens=False) for l in LETTERS]
        if any(len(x) != 1 for x in ids):
            raise ValueError("option letters must be single tokens")
        self._letter_ids = [x[0] for x in ids]
        self.label_ids = self.letter_ids(len(LABELS))

    def letter_ids(self, n: int) -> list[int]:
        """Token ids of the first ``n`` option letters (A, B, ...)."""
        if not 2 <= n <= len(LETTERS):
            raise ValueError(f"need 2-{len(LETTERS)} options, got {n}")
        return self._letter_ids[:n]

    def encode(self, prompt: str, frames: list, frames_as: str = "images", soft_tokens: int | None = None,
               timestamps=None, chat: bool = False) -> Encoded:
        b = self._backend
        b.frame_mode, b.frame_soft_tokens = frames_as, soft_tokens
        enc = self._scorer._multimodal_inputs(prompt, frames, chat=chat, sep="", timestamps=timestamps)
        ids = enc["input_ids"]
        eoi = self.cfg.eoi_token_id
        prefix_len = max(i for i, t in enumerate(ids) if t == eoi) + 1
        return Encoded(ids, enc["pixel_values"].numpy(), enc["position_ids"].numpy(), prefix_len)
