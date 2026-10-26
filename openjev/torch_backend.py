"""PyTorch inference on CPU, CUDA/ROCm and MPS, with shared-prefix scoring."""
from __future__ import annotations

import copy


class TorchBackend:
    def __init__(self, model_path: str, device: str, adapter_path: str | None):
        if adapter_path:
            raise ValueError("MLX adapters are not compatible with the torch backend; use base HF weights")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError('Install the PyTorch backend with pip install -e ".[torch]"') from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda[:N], or mps")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable: install a GPU-enabled PyTorch build and GPU driver")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this machine")
        # Gemma's activations overflow float16 (NaN hidden states), so use bfloat16 where
        # the device has it (CUDA from Ampere, MPS) and float32 otherwise (e.g. T4, V100, CPU).
        dtype = torch.float32
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        elif self.device.type == "mps":
            dtype = torch.bfloat16
        config = AutoConfig.from_pretrained(model_path)
        # Gemma 3 4B checkpoints wrap the language model in a multimodal model.
        # Use its native wrapper so checkpoint names and tied weights load correctly.
        cls = Gemma3ForConditionalGeneration if config.model_type == "gemma3" else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_path, torch_dtype=dtype).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path)

    @property
    def text_model(self):
        """The decoder without the LM head (embeddings, layers, final norm)."""
        inner = self.model.model
        return getattr(inner, "language_model", inner)

    @property
    def hidden_size(self):
        config = self.model.config
        return getattr(config, "text_config", config).hidden_size

    def _tensor(self, ids):
        return self.torch.tensor(ids, dtype=self.torch.long, device=self.device)

    def prefill(self, ids):
        with self.torch.inference_mode():
            output = self.model(input_ids=self._tensor([ids]), use_cache=True)
            last = output.logits[0, -1].clone()
            # Synchronize so public timing includes GPU execution.
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
            elif self.device.type == "mps":
                self.torch.mps.synchronize()
            return (output.past_key_values, len(ids)), last

    def score_with_prefix(self, prefix, last, options, batch_size, pad_id):
        torch = self.torch
        cache, prefix_length = prefix
        sums = []
        with torch.inference_mode():
            for start in range(0, len(options), batch_size):
                chunk = options[start:start + batch_size]
                n, length = len(chunk), max(map(len, chunk))
                ids = self._tensor([x + [pad_id] * (length - len(x)) for x in chunk])
                mask = torch.arange(length, device=self.device)[None, :] < self._tensor([len(x) for x in chunk])[:, None]
                expanded = copy.deepcopy(cache)
                expanded.batch_repeat_interleave(n)
                attention_mask = torch.cat((torch.ones((n, prefix_length), device=self.device, dtype=torch.bool), mask), dim=1)
                positions = torch.arange(prefix_length, prefix_length + length, device=self.device)
                output = self.model(input_ids=ids, past_key_values=expanded,
                                    attention_mask=attention_mask,
                                    position_ids=positions[None, :].expand(n, -1), use_cache=True)
                pred = torch.cat((last[None, None, :].expand(n, 1, -1), output.logits[:, :-1]), dim=1).float()
                target = pred.gather(-1, ids[..., None]).squeeze(-1)
                scores = ((target - pred.logsumexp(-1)) * mask).sum(-1)
                sums.extend(scores.cpu().tolist())
        return sums

    def score_naive(self, context, options):
        if not context:
            raise ValueError("context must contain tokens")
        scores = []
        with self.torch.inference_mode():
            for option in options:
                logits = self.model(input_ids=self._tensor([context + option]), use_cache=False).logits[0]
                pred = logits[len(context) - 1:len(context) - 1 + len(option)].float()
                target = pred.gather(-1, self._tensor(option)[:, None]).squeeze(-1)
                scores.append((target - pred.logsumexp(-1)).sum().item())
        return scores

    def features(self, context, options, bos_id, pad_id, batch_size):
        """Final hidden states: the context's (Lc, H), and each option's mean over its own
        tokens encoded after BOS alone (N, H); both float16 numpy."""
        import numpy as np
        torch = self.torch
        with torch.inference_mode():
            ctx = self.text_model(input_ids=self._tensor([context])).last_hidden_state[0]
            pooled = []
            for start in range(0, len(options), batch_size):
                chunk = options[start:start + batch_size]
                length = max(map(len, chunk))
                ids = self._tensor([[bos_id] + x + [pad_id] * (length - len(x)) for x in chunk])
                mask = self._tensor([[1] * (len(x) + 1) + [0] * (length - len(x)) for x in chunk])
                h = self.text_model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 1:].float()
                m = mask[:, 1:, None].float()
                pooled.append(((h * m).sum(1) / m.sum(1)).cpu())
            return (ctx.float().cpu().numpy().astype(np.float16),
                    torch.cat(pooled).numpy().astype(np.float16))
