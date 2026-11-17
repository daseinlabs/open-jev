"""Qwen3-VL on MLX (mlx-vlm 0.7.8): inputs, option-letter scoring and LoRA training.

Inputs always come from the official ``transformers`` Qwen3-VL processor (chat template, video
patches, ``<t seconds>`` timestamps), fed frames we decode ourselves (``openjev.video``), so
MLX and the torch reference see identical token ids and pixels. The vision encoder runs once
per video, outside any gradient; its output and the DeepStack features (added to the first LLM
layers) are reused by every question of that video.

Training packs all questions of a video into one multi-turn conversation (video, Q1, A1, Q2,
A2, ...): one forward and backward per video, cross-entropy on the assistant tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

LETTERS = list("ABCDEFGH")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MC_SUFFIX = "Answer with the option's letter from the given choices directly."


def mc_text(question: str, options: list[str]) -> str:
    return "\n".join([question] + [f"{l}. {o}" for l, o in zip(LETTERS, options)] + [MC_SUFFIX])


@dataclass
class Encoded:
    input_ids: np.ndarray       # (1, L) int
    pixel_values: np.ndarray    # (patches, 1536)
    grid_thw: np.ndarray        # (videos, 3)
    text: str                   # prompt text with video pads collapsed (for checks)


class Inputs:
    """Official processor with a fixed frame count and a per-frame pixel budget."""

    def __init__(self, model_path: str, tokens_per_group: int = 192):
        # Named classes, not AutoProcessor: importing mlx-vlm's qwen3_vl re-registers the Auto
        # classes to its own processor, which ignores the pixel budget (8x more vision tokens).
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        self.proc = Qwen3VLProcessor.from_pretrained(
            model_path, image_processor=Qwen2VLImageProcessor.from_pretrained(model_path),
            video_processor=Qwen3VLVideoProcessor.from_pretrained(model_path))
        if type(self.proc.tokenizer).__module__.startswith("mlx_vlm"):
            raise TypeError("tokenizer resolved to mlx-vlm's class, expected the official one")
        self.tok = self.proc.tokenizer
        self.pixels_per_frame = tokens_per_group * 32 * 32  # one token = 32x32 pixels over 2 frames
        ids = [self.tok.encode(l, add_special_tokens=False) for l in LETTERS]
        if any(len(x) != 1 for x in ids):
            raise ValueError("option letters must be single tokens")
        self.letter_ids = [x[0] for x in ids]
        self.im_start, self.im_end = self.tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
        self.assistant_ids = self.tok.encode("assistant\n", add_special_tokens=False)

    def encode(self, turns: list[tuple[str, str | None]], video, add_generation_prompt: bool) -> Encoded:
        """``turns``: (user text, assistant answer or None); the video goes before the first question.
        ``video``: one video, or a list of (label, video) with the same frame count: each label's
        text goes just before its video."""
        from transformers.video_utils import VideoMetadata
        vids = video if isinstance(video, list) else [(None, video)]
        first = [c for label, _ in vids for c in ([{"type": "text", "text": label}] if label else []) + [{"type": "video"}]]
        msgs = []
        for i, (q, a) in enumerate(turns):
            content = (first if i == 0 else []) + [{"type": "text", "text": q}]
            msgs.append({"role": "user", "content": content})
            if a is not None:
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        n = len(vids[0][1].frames)
        # frames_indices / fps = our own sample times, so the prompt's "<t seconds>" are exact.
        metas = [VideoMetadata(total_num_frames=n, fps=1.0, frames_indices=list(v.timestamps),
                               width=v.width, height=v.height, duration=v.duration_s) for _, v in vids]
        out = self.proc(text=[text], videos=[v.frames for _, v in vids], video_metadata=metas, do_sample_frames=False,
                        size={"shortest_edge": 4096, "longest_edge": n * self.pixels_per_frame},
                        cap_pixels_per_frame=False, return_tensors="np")
        ids = out["input_ids"]
        shown = self.tok.decode([t for t in ids[0].tolist() if t != self.proc.video_token_id])
        return Encoded(ids, out["pixel_values_videos"], out["video_grid_thw"], shown)

    def answer_turns(self, ids: np.ndarray) -> list[tuple[list[int], list[int]]]:
        """Per assistant turn, (positions, targets): its tokens (answer + <|im_end|>) as next-token targets."""
        ids, a, out = ids[0].tolist(), self.assistant_ids, []
        for i, t in enumerate(ids):
            if t == self.im_start and ids[i + 1:i + 1 + len(a)] == a:
                j, pos = i + 1 + len(a), []
                while ids[j] != self.im_end:
                    pos.append(j)
                    j += 1
                pos.append(j)
                out.append(([p - 1 for p in pos], [ids[p] for p in pos]))
        return out


def load_model(path: str):
    from mlx_vlm.utils import load
    model, _ = load(path)
    if not model.config.text_config.tie_word_embeddings:
        raise ValueError("expected tied embeddings (Qwen3-VL-4B); the letter readout uses embed_tokens")
    model.eval()
    return model


def vision_features(model, enc: Encoded):
    """Frozen encoder output and DeepStack features for one video."""
    feats, deep = model.vision_tower(mx.array(enc.pixel_values).astype(model.vision_tower.patch_embed.proj.weight.dtype),
                                     mx.array(enc.grid_thw))
    out = mx.stop_gradient(feats), [mx.stop_gradient(d) for d in deep]
    mx.eval(out)
    return out


def hidden(model, enc: Encoded, vis, cache=None) -> mx.array:
    """(L, hidden) final-norm hidden states of the whole prompt (``cache``: filled if given)."""
    from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter
    feats, deep = vis
    ids = mx.array(enc.input_ids)
    lm = model.language_model
    vmask = ids == model.config.video_token_index
    embeds = masked_scatter(lm.model.embed_tokens(ids), mx.broadcast_to(vmask[..., None], (*ids.shape, feats.shape[-1])),
                            feats)
    pos, _ = lm.get_rope_index(ids, None, mx.array(enc.grid_thw), None)
    return lm.model(ids, inputs_embeds=embeds, cache=cache, position_ids=pos, visual_pos_masks=vmask,
                    deepstack_visual_embeds=deep)[0]


@dataclass
class Prefix:
    """KV cache of the prompt up to the last video token, shared by every question of a video."""
    ids: list[int]
    cache: list


def prefill(model, enc: Encoded, vis) -> Prefix:
    """Run the prompt up to and including the last video token once (inference only)."""
    from mlx_vlm.models.cache import KVCache
    ids = enc.input_ids[0].tolist()
    vid = model.config.video_token_index
    n = max(i for i, t in enumerate(ids) if t == vid) + 1
    sub = Encoded(enc.input_ids[:, :n], enc.pixel_values, enc.grid_thw, "")
    cache = [KVCache() for _ in model.language_model.model.layers]
    hidden(model, sub, vis, cache)
    mx.eval([c.keys for c in cache])
    return Prefix(ids[:n], cache)


def _suffix(model, enc: Encoded, pre: Prefix):
    """(hidden states of the tokens after the prefix, next text position); cache extended in place."""
    ids, n = enc.input_ids[0].tolist(), len(pre.ids)
    if ids[:n] != pre.ids:
        raise ValueError("prompt does not start with the cached prefix")
    for c in pre.cache:
        c.offset = n  # drop the previous question's tokens
    lm = model.language_model
    pos, _ = lm.get_rope_index(mx.array(enc.input_ids), None, mx.array(enc.grid_thw), None)
    h = lm.model(mx.array([ids[n:]]), cache=pre.cache, position_ids=pos[..., n:])[0]
    return h, int(pos[:, 0, -1].max()) + 1


def hidden_last(model, enc: Encoded, pre: Prefix) -> mx.array:
    """Final hidden state at the last prompt position, from the cached prefix."""
    return _suffix(model, enc, pre)[0][-1]


def generate(model, inp: Inputs, enc: Encoded, pre: Prefix, max_tokens: int = 24) -> str:
    """Greedy answer after the prompt (text positions after the video advance by one per token)."""
    h, p = _suffix(model, enc, pre)
    lm, out = model.language_model, []
    tok = int(mx.argmax(logits_at(model, h[-1])))
    while tok != inp.im_end and len(out) < max_tokens:
        out.append(tok)
        h = lm.model(mx.array([[tok]]), cache=pre.cache, position_ids=mx.full((3, 1, 1), p))[0]
        p += 1
        tok = int(mx.argmax(logits_at(model, h[-1])))
    return inp.tok.decode(out)


def logits_at(model, h: mx.array, rows: list[int] | None = None) -> mx.array:
    """float32 logits from hidden states, over ``rows`` of the tied head (all of it if None).

    bf16 logits step by 0.125 at this scale, which visibly moves option probabilities.
    """
    E, h = model.language_model.model.embed_tokens, h.astype(mx.float32)
    if rows is None:
        return E.as_linear(h).astype(mx.float32)
    return h @ E(mx.array(rows)).astype(mx.float32).T


def attach_lora(model, rank: int = 8, alpha: float = 16.0, checkpoint: bool = True):
    """LoRA on the language model's linear layers only (vision encoder and merger stay frozen)."""
    from mlx_vlm.trainer.peft.utils import get_peft_model
    model = get_peft_model(model, LORA_TARGETS, rank=rank, alpha=alpha, dropout=0.0, verbose=False)
    names = [k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())]
    if not names or any(not k.startswith("language_model.") for k in names):
        raise ValueError("LoRA must sit on the language model only")
    if checkpoint:
        _checkpoint_layers(type(model.language_model.model.layers[0]))
    return model


def _checkpoint_layers(cls):
    """Recompute each decoder layer in the backward pass (training only, no KV cache)."""
    if getattr(cls, "_openjev_checkpointed", False):
        return
    orig = cls.__call__

    def call(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None):
        if not self.training or cache is not None:
            return orig(self, x, mask, cache, position_ids, position_embeddings)
        pe = tuple(position_embeddings) if position_embeddings is not None else ()

        def inner(params, xi, *pei):
            self.update(params)
            return orig(self, xi, mask, None, position_ids, tuple(pei) if pei else None)

        return mx.checkpoint(inner)(self.trainable_parameters(), x, *pe)

    cls.__call__ = call
    cls._openjev_checkpointed = True


def turn_ce(model, inp: Inputs, enc: Encoded, vis) -> tuple[mx.array, int]:
    """(sum over assistant turns of each turn's mean token cross-entropy, number of turns).

    Each answer counts once whatever its length: a 15-token LLaVA-Video answer weighs as much
    as a one-letter one.
    """
    turns = inp.answer_turns(enc.input_ids)
    pos = [p for ps, _ in turns for p in ps]
    w = mx.array([1.0 / len(ps) for ps, _ in turns for _ in ps])
    h = hidden(model, enc, vis)[mx.array(pos)]
    ce = nn.losses.cross_entropy(logits_at(model, h), mx.array([t for _, ts in turns for t in ts]), reduction="none")
    return (ce * w).sum(), len(turns)
