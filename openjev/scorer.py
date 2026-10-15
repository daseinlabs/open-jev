"""Zero-shot option scoring with Gemma 3 via MLX or PyTorch (Route B in docs/design/one-pass-option-scoring.md).

The context is prefilled once. Its KV cache is then expanded across the batch
dimension so every option is scored in one padded forward pass that shares the
context prefix. No decoding happens anywhere; the score is the log-probability the
language model assigns to the option tokens given the context.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import platform

DEFAULT_MODEL = "models/gemma-3-4b-it"
NORMS = ("mean", "sum", "pmi")


@dataclass(frozen=True)
class OptionScore:
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float

    def to_dict(self) -> dict:
        return asdict(self)


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


class OptionScorer:
    """Scores pre-written options as continuations of a context with a causal LM.

    Args:
        model_path: local directory or Hugging Face repo id.
        backend: auto selects MLX on Apple silicon, PyTorch elsewhere.
        device: PyTorch device (auto, cpu, cuda, cuda:N, mps).
        dtype: PyTorch weight dtype override (float32, float16, bfloat16).
        vision_dtype: PyTorch dtype for the vision tower only (defaults to ``dtype``).
        batch_size: maximum number of options scored in one forward pass.
        chat: wrap the context in Gemma's chat template (user turn, generation
            prompt appended) so options are scored as the start of the reply.
        sep: literal string placed between context and each option (ignored
            when ``chat`` is set, since the template already ends the turn).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        batch_size: int = 8,
        chat: bool = False,
        sep: str = "",
        adapter_path: str | None = None,
        backend: str = "auto",
        device: str = "auto",
        dtype: str | None = None,
        vision_dtype: str | None = None,
    ) -> None:
        if backend not in ("auto", "mlx", "torch"):
            raise ValueError("backend must be auto, mlx, or torch")
        self.backend = ("mlx" if platform.system() == "Darwin" and platform.machine() == "arm64"
                        else "torch") if backend == "auto" else backend
        self._engine = None
        if self.backend == "torch":
            from .torch_backend import TorchBackend
            self._engine = TorchBackend(model_path, device, adapter_path, dtype=dtype, vision_dtype=vision_dtype)
            self.model, self.tok = self._engine.model, self._engine.tok
            self.device = str(self._engine.device)
        else:
            if device not in ("auto", "mps"):
                raise ValueError("MLX uses Metal; use --backend torch for --device selection")
            global mx, KVCache
            try:
                import mlx.core as mx
                from mlx_lm import load
                from mlx_lm.models.cache import KVCache
            except ImportError as exc:
                raise RuntimeError("MLX requires Apple silicon; use --backend torch") from exc
            self.model, self.tok = load(model_path, adapter_path=adapter_path)
            self.device = "metal"
        self.batch_size = max(1, batch_size)
        self.chat = chat
        self.sep = sep
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.bos_id = self.tok.bos_token_id
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ text
    def context_ids(self, context: str, chat: bool | None = None, sep: str | None = None) -> list[int]:
        chat = self.chat if chat is None else chat
        sep = self.sep if sep is None else sep
        if chat:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tok.encode(text, add_special_tokens=False)
            if self.bos_id is not None and ids[:1] != [self.bos_id]:
                ids = [self.bos_id] + ids
            return ids
        ids = self.tok.encode(context + sep)  # adds BOS
        if self.bos_id is not None and ids[:1] != [self.bos_id]:
            ids = [self.bos_id] + ids
        return ids

    def _multimodal_inputs(self, context: str, images: list, chat: bool | None, sep: str | None) -> dict:
        chat = self.chat if chat is None else chat
        sep = self.sep if sep is None else sep
        if chat:  # images go at the start of the user turn
            text = self.tok.apply_chat_template([{"role": "user", "content": context}],
                                                tokenize=False, add_generation_prompt=True)
            head = "<start_of_turn>user\n"
            if head not in text:
                raise ValueError("chat template has no user turn to attach images to")
            pre, _, post = text.partition(head)
            enc = self._engine.encode_multimodal(post, images, add_bos=False)
            head_ids = self.tok.encode(pre + head, add_special_tokens=False)
            if self.bos_id is not None and head_ids[:1] != [self.bos_id]:
                head_ids = [self.bos_id] + head_ids
            import torch
            enc["token_type_ids"] = torch.cat([torch.zeros(len(head_ids), dtype=enc["token_type_ids"].dtype),
                                               enc["token_type_ids"]])
            enc["input_ids"] = head_ids + enc["input_ids"]
            return enc
        return self._engine.encode_multimodal(context + sep, images, add_bos=True)

    def option_ids(self, option: str) -> list[int]:
        ids = self.tok.encode(option, add_special_tokens=False)
        if not ids:
            raise ValueError(f"option tokenises to nothing: {option!r}")
        return ids

    # --------------------------------------------------------------- prefill
    def _prefill(self, ids: list[int]) -> tuple[list[KVCache], mx.array]:
        if not ids:
            raise ValueError("context must contain tokens (or the tokenizer must define BOS)")
        if self._engine is not None:
            return self._engine.prefill(ids)
        cache = [KVCache() for _ in self.model.layers]
        logits = self.model(mx.array(ids)[None], cache=cache)
        last = logits[0, -1]
        mx.eval(last, *[c.keys for c in cache], *[c.values for c in cache])
        return cache, last

    @staticmethod
    def _expand(cache: list[KVCache], n: int) -> list[KVCache]:
        out = []
        for c in cache:
            e = KVCache()
            e.offset = c.offset
            e.keys = mx.repeat(c.keys, n, axis=0)
            e.values = mx.repeat(c.values, n, axis=0)
            out.append(e)
        return out

    def _score_with_prefix(
        self, cache: list[KVCache], last: mx.array, opts: list[list[int]]
    ) -> list[float]:
        """Sum of log p(option tokens | prefix) for each option, sharing the prefix cache."""
        if self._engine is not None:
            return self._engine.score_with_prefix(cache, last, opts, self.batch_size, self.pad_id)
        sums: list[float] = []
        for start in range(0, len(opts), self.batch_size):
            chunk = opts[start : start + self.batch_size]
            n, L = len(chunk), max(len(x) for x in chunk)
            arr = mx.array([x + [self.pad_id] * (L - len(x)) for x in chunk])
            mask = mx.array([[1.0] * len(x) + [0.0] * (L - len(x)) for x in chunk])
            logits = self.model(arr, cache=self._expand(cache, n))  # (n, L, V)
            first = mx.broadcast_to(last[None, None, :], (n, 1, logits.shape[-1]))
            pred = mx.concatenate([first, logits[:, :-1]], axis=1).astype(mx.float32)
            lse = mx.logsumexp(pred, axis=-1)  # (n, L)
            tgt = mx.take_along_axis(pred, arr[..., None], axis=-1)[..., 0]
            s = ((tgt - lse) * mask).sum(-1)
            mx.eval(s)
            sums.extend(s.tolist())
        return sums

    # ---------------------------------------------------------------- public
    def score(
        self,
        context: str,
        options: Sequence[str],
        norm: str = "mean",
        chat: bool | None = None,
        sep: str | None = None,
        images: Sequence | None = None,
        image_features=None,
    ) -> list[OptionScore]:
        """Score ``options`` as continuations of ``context``.

        ``images`` (PIL images, e.g. video frames) are placed before the context, in order.
        Image input currently needs the torch backend and a Gemma 3 multimodal checkpoint.
        ``image_features`` (from the engine's ``image_features``) reuses vision-tower output.
        """
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if len(options) < 2:
            raise ValueError("need at least two options")
        t0 = time.perf_counter()
        opts = [self.option_ids(o) for o in options]
        if images:
            if self._engine is None:
                raise ValueError("image/video input currently needs the torch backend (--backend torch)")
            enc = self._multimodal_inputs(context, list(images), chat=chat, sep=sep)
            ctx_ids = enc["input_ids"]
            cache, last = self._engine.prefill_multimodal(enc, image_features=image_features)
        else:
            ctx_ids = self.context_ids(context, chat=chat, sep=sep)
            cache, last = self._prefill(ctx_ids)
        t1 = time.perf_counter()
        sums = self._score_with_prefix(cache, last, opts)
        t2 = time.perf_counter()

        uncond: list[float] | None = None
        if norm == "pmi":
            if self.bos_id is None:
                raise ValueError("PMI requires a tokenizer with a BOS token")
            base_cache, base_last = self._prefill([self.bos_id])
            uncond = self._score_with_prefix(base_cache, base_last, opts)
        t3 = time.perf_counter()
        self.last_timing = {
            "prefill_s": t1 - t0,
            "options_s": t2 - t1,
            "uncond_s": t3 - t2,
            "total_s": t3 - t0,
            "context_tokens": len(ctx_ids),
            "images": len(images or ()),
            "option_tokens": sum(len(o) for o in opts),
        }

        means = [s / len(o) for s, o in zip(sums, opts)]
        if norm == "sum":
            scores = sums
        elif norm == "mean":
            scores = means
        else:
            scores = [s - u for s, u in zip(sums, uncond)]
        probs = _softmax(scores)
        return [
            OptionScore(
                option=o,
                n_tokens=len(ids),
                logprob_sum=s,
                logprob_mean=m,
                logprob_uncond=(uncond[i] if uncond is not None else None),
                score=sc,
                probability=p,
            )
            for i, (o, ids, s, m, sc, p) in enumerate(
                zip(options, opts, sums, means, scores, probs)
            )
        ]

    def score_naive(self, context: str, options: Sequence[str]) -> list[float]:
        """Reference: re-encode context + option from scratch per option, no cache.

        Returns the sum of option-token log-probs. Used to verify the prefix-shared
        batched path and to benchmark against it.
        """
        ctx = self.context_ids(context)
        if self._engine is not None:
            return self._engine.score_naive(ctx, [self.option_ids(o) for o in options])
        out = []
        for o in options:
            oid = self.option_ids(o)
            ids = mx.array(ctx + oid)[None]
            logits = self.model(ids)[0].astype(mx.float32)
            pred = logits[len(ctx) - 1 : len(ctx) - 1 + len(oid)]
            lp = pred - mx.logsumexp(pred, axis=-1, keepdims=True)
            s = mx.take_along_axis(lp, mx.array(oid)[:, None], axis=-1).sum()
            mx.eval(s)
            out.append(s.item())
        return out


def iter_jsonl(path: str) -> Iterable[dict]:
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
