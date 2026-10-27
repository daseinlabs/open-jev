"""Gemma 4 E4B on MLX (mlx-vlm 0.7.8) for option-letter scoring and LoRA training.

Option-letter readout: the logits of the letter tokens at the last prompt position, the same
readout OpenJev scores with (single-token options, ``norm="sum"``).

Two equivalent ways to score the questions of one video:
- ``letter_logits``: one full sequence per question;
- ``letter_logits_grouped``: the video prefix once, then every question as a batch that reuses
  its KV cache. Gradients flow through the shared prefix (``video_train/check_grouping.py``
  verifies loss and gradients match).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import KVCache

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(path: str):
    from mlx_vlm.utils import load
    model, _ = load(path)
    # Unused here, and its relative-position embedding breaks Module.freeze().
    model.audio_tower = model.embed_audio = None
    model.eval()
    return model


def attach_lora(model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, checkpoint: bool = True):
    """LoRA on the language model's linear layers only; everything else frozen."""
    from mlx_vlm.trainer.peft.utils import get_peft_model
    model = get_peft_model(model, LORA_TARGETS, rank=rank, alpha=alpha, dropout=dropout, verbose=False)
    if checkpoint:
        _checkpoint_layers(type(model.language_model.model.layers[0]))
    for layer in model.language_model.model.layers:
        layer.self_attn.rope = _IntOffsetRope(layer.self_attn.rope)
    return model


def _checkpoint_layers(cls):
    """Gradient checkpointing for every decoder layer of class ``cls``, safe with a KV cache.

    mlx-vlm's ``grad_checkpoint`` re-runs the layer in the backward pass, which appends to the
    cache a second time. Here the cache contents go in and come out as checkpoint inputs and
    outputs, and the real cache is updated once, outside.
    """
    if getattr(cls, "_openjev_checkpointed", False):
        return
    orig = cls.__call__

    def call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
        # The patch is class-wide: inference (eval mode, e.g. a serving model) runs the layer as is.
        if not self.training or (cache is not None and not isinstance(cache, _ConcatCache)):
            return orig(self, x, mask, cache, per_layer_input=per_layer_input, shared_kv=shared_kv, offset=offset)
        has_pli, has_skv = per_layer_input is not None, shared_kv is not None
        has_prev = cache is not None and cache.keys is not None
        prev_off = cache.offset if cache is not None else 0
        arrays = [x] + ([per_layer_input] if has_pli else []) + (list(shared_kv) if has_skv else []) \
            + ([cache.keys, cache.values] if has_prev else [])

        def inner(params, *arrs):
            self.update(params)
            arrs = list(arrs)
            xi = arrs.pop(0)
            pli = arrs.pop(0) if has_pli else None
            skv = (arrs.pop(0), arrs.pop(0)) if has_skv else None
            c = None
            if cache is not None:
                c = _ConcatCache()
                if has_prev:
                    c.keys, c.values = arrs.pop(0), arrs.pop(0)
                c.offset = prev_off
            h, kvs, _ = orig(self, xi, mask, c, per_layer_input=pli, shared_kv=skv, offset=offset)
            return h, kvs[0], kvs[1]

        h, k, v = mx.checkpoint(inner)(self.trainable_parameters(), *arrays)
        if has_skv:
            out_off = offset
        elif cache is not None:
            out_off = mx.array(prev_off)
            cache.keys, cache.values, cache.offset = k, v, prev_off + x.shape[1]
        else:
            out_off = 0
        return h, (k, v), out_off

    cls.__call__ = call
    cls._openjev_checkpointed = True


class _IntOffsetRope(nn.Module):
    """mlx-vlm passes a cached prefix's offset as an mx.array; RoPE has no VJP for an array
    offset, so backprop through a KV-cached prefix fails. A scalar offset is a constant: pass an int."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, offset=0):
        if isinstance(offset, mx.array) and offset.ndim == 0:
            offset = offset.item()
        return self.inner(x, offset=offset)


def vision_features(model, pixel_values, position_ids, frame_batch: int = 2):
    """Frozen encoder + projector output, (frames, tokens_per_frame, hidden).

    ``frame_batch`` frames at a time: all 16 frames at 280 tokens in one call peaked at
    13.7 GB with the 8-bit model loaded, which swaps on an 18 GB Mac.
    """
    pv, pos = mx.array(pixel_values), mx.array(position_ids)
    out = []
    for s in range(0, pv.shape[0], frame_batch):
        f = model.encode_image(pv[s:s + frame_batch], pos[s:s + frame_batch])
        mx.eval(f)
        out.append(f)
    return mx.stop_gradient(mx.concatenate(out, axis=1))  # each call returns (1, frames*tokens, hidden)


def _soft_token_id(model, frames_as: str) -> int:
    return model.config.video_token_id if frames_as == "video" else model.config.image_token_id


def _embed(model, ids: mx.array, feats: mx.array | None, frames_as: str):
    """Main and per-layer embeddings, with the vision features scattered into the soft tokens."""
    kw = {}
    if feats is not None:
        key = "video" if frames_as == "video" else "image"
        # The source tensor only switches the scatter on; the cached features are what is used.
        kw = {("pixel_values_videos" if key == "video" else "pixel_values"): feats,
              f"cached_{key}_features": feats}
    out = model.get_input_embeddings(input_ids=ids, **kw)
    return out.inputs_embeds, out.per_layer_inputs


def _hidden(model, ids, feats, frames_as, cache=None, keep=1):
    """Final-norm hidden states (the last ``keep`` positions, or all if 0)."""
    embeds, ple = _embed(model, ids, feats, frames_as)
    h = model.language_model.model(None, inputs_embeds=embeds, per_layer_inputs=ple, cache=cache,
                                   logits_to_keep=keep)
    return h[:, -keep:] if keep else h


def _letter_readout(model, h, label_ids):
    """Letter logits from final hidden states, in float32.

    The full-vocabulary head returns bf16 logits; at Gemma's logit scale (~20-30) bf16 steps
    are 0.125, which visibly moves the option probabilities. Only the letter rows are needed.
    """
    lm = model.language_model
    rows = lm.model.embed_tokens(mx.array(label_ids)).astype(mx.float32)  # tied head rows
    logits = h.astype(mx.float32) @ rows.T
    cap = lm.final_logit_softcapping
    return mx.tanh(logits / cap) * cap if cap else logits


def letter_logits(model, ids: list[int], feats, frames_as: str, label_ids: list[int]) -> mx.array:
    """(n_labels,) float32 letter logits for one full prompt."""
    return _letter_readout(model, _hidden(model, mx.array(ids)[None], feats, frames_as)[0, -1], label_ids)


class _ConcatCache(KVCache):
    """KVCache that grows by concatenation (no in-place writes), so gradients reach the prefix."""

    def update_and_fetch(self, keys, values):
        self.keys = keys if self.keys is None else mx.concatenate([self.keys, keys], axis=2)
        self.values = values if self.values is None else mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values


def letter_logits_grouped(model, prefix_ids: list[int], feats, frames_as: str, suffixes: list[list[int]],
                          label_ids: list[int]) -> mx.array:
    """(n_questions, n_labels) letter logits: shared video prefix once, questions as one batch.

    Sliding-window layers get a plain (non-rotating) cache; their mask applies the window, so
    attention is the same as in a single full pass. Suffixes are right-padded; causal attention
    keeps padding invisible to the real tokens, and each row is read at its own last token.
    """
    lm = model.language_model
    n_cached = lm.model.first_kv_shared_layer_idx
    cache = [_ConcatCache() for _ in range(n_cached)]
    _hidden(model, mx.array(prefix_ids)[None], feats, frames_as, cache=cache, keep=1)
    k = len(suffixes)
    for c in cache:
        c.keys, c.values = mx.repeat(c.keys, k, axis=0), mx.repeat(c.values, k, axis=0)
    width = max(map(len, suffixes))
    pad = model.config.text_config.pad_token_id or 0
    ids = mx.array([s + [pad] * (width - len(s)) for s in suffixes])
    h = _hidden(model, ids, None, frames_as, cache=cache, keep=0)  # (k, width, hidden)
    last = mx.array([len(s) - 1 for s in suffixes])
    return _letter_readout(model, h[mx.arange(k), last], label_ids)


def letter_ce(logits: mx.array, target) -> mx.array:
    return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), mx.array(target).reshape(-1),
                                   reduction="mean")
