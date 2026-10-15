"""PyTorch inference on CPU, CUDA/ROCm and MPS, with shared-prefix scoring."""
from __future__ import annotations

import copy


class TorchBackend:
    def __init__(self, model_path: str, device: str, adapter_path: str | None, dtype: str | None = None,
                 vision_dtype: str | None = None):
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
        dtype_name, dtype = dtype, torch.float32
        for name in (dtype_name, vision_dtype):
            if name not in (None, "float32", "float16", "bfloat16"):
                raise ValueError("dtype must be float32, float16 or bfloat16")
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == "mps":
            dtype = torch.float16
        if dtype_name:
            dtype = getattr(torch, dtype_name)
        config = AutoConfig.from_pretrained(model_path)
        # Gemma 3 4B checkpoints wrap the language model in a multimodal model.
        # Use its native wrapper so checkpoint names and tied weights load correctly.
        cls = Gemma3ForConditionalGeneration if config.model_type == "gemma3" else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_path, torch_dtype=dtype).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model_path, self.dtype, self._processor = model_path, dtype, None
        self.vision_dtype = getattr(torch, vision_dtype) if vision_dtype else dtype
        if vision_dtype and hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
            # Gemma 3 overflows in float16 in the language model but not in SigLIP, where
            # float16 tracks float32 far closer than bfloat16 does.
            self.model.model.vision_tower.to(self.vision_dtype)
            self.model.model.multi_modal_projector.to(self.vision_dtype)

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

    # ---------------------------------------------------------- images/video
    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        return self._processor

    def encode_multimodal(self, text: str, images: list, add_bos: bool = True) -> dict:
        """Token ids, token types and pixels for ``text`` preceded by ``images``, in order.

        Each image becomes the processor's full image sequence (``\\n\\n<start_of_image>``,
        256 soft tokens, ``<end_of_image>\\n\\n``) so layout and preprocessing match HF.
        """
        if getattr(self.model.config, "model_type", None) != "gemma3":
            raise ValueError("image input needs a Gemma 3 multimodal checkpoint")
        if not images:
            raise ValueError("images must be non-empty")
        proc = self.processor
        out = proc(text=proc.boi_token * len(images) + text, images=[list(images)],
                   return_tensors="pt", add_special_tokens=add_bos)
        return {"input_ids": out["input_ids"][0].tolist(), "token_type_ids": out["token_type_ids"][0],
                "pixel_values": out["pixel_values"]}

    def image_features(self, pixel_values, frame_batch: int = 2):
        """Projected SigLIP features, (n_images, tokens_per_image, hidden), ``frame_batch`` at a time."""
        torch = self.torch
        with torch.inference_mode():
            feats = [self.model.model.get_image_features(pv.to(self.device, self.vision_dtype), return_dict=True).pooler_output
                     for pv in pixel_values.split(frame_batch)]
            feats = torch.cat(feats).to(self.dtype)
            self._sync()
        return feats

    def prefill_multimodal(self, enc: dict, frame_batch: int = 2, chunk_tokens: int = 2048, image_features=None):
        """Prefill text + images; returns the same ``(cache, length), last_logits`` as :meth:`prefill`.

        Image features are computed ``frame_batch`` frames at a time and scattered into the
        embeddings exactly as ``Gemma3Model.forward`` does. The sequence is then prefilled in
        chunks that only end on image-block boundaries, so the bidirectional within-image mask
        HF builds from ``token_type_ids`` is the same as in a single pass. ``image_features``
        from :meth:`image_features` skips the vision tower (e.g. several questions, one video).
        """
        torch = self.torch
        model = self.model
        cfg = model.config
        ids = self._tensor([enc["input_ids"]])
        tt = enc["token_type_ids"].to(self.device)[None]
        is_img = ids == cfg.image_token_id
        with torch.inference_mode():
            embeds = model.get_input_embeddings()(ids.masked_fill(is_img, 0))
            if image_features is None:
                image_features = self.image_features(enc["pixel_values"], frame_batch)
            feats = image_features.to(self.device, embeds.dtype).reshape(-1, embeds.shape[-1])
            if feats.shape[0] != int(is_img.sum()):
                raise ValueError("image token count does not match image features")
            embeds = embeds.masked_scatter(is_img[..., None], feats)
            del feats
            cache, n = None, ids.shape[1]
            for s, e in _chunks(tt[0].tolist(), chunk_tokens):
                # HF indexes the block mask by absolute position, so pass token types from 0.
                out = model(inputs_embeds=embeds[:, s:e], token_type_ids=tt[:, :e], past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
            last = out.logits[0, -1].clone()
            self._sync()
        return (cache, n), last

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

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


def _chunks(token_types: list[int], size: int) -> list[tuple[int, int]]:
    """Split [0, n) into spans of about ``size`` tokens that never cut through an image block."""
    n, spans, start = len(token_types), [], 0
    while start < n:
        end = min(n, start + size)
        while 0 < end < n and token_types[end] == 1 and token_types[end - 1] == 1:
            end += 1  # extend to the end of the image block
        spans.append((start, end))
        start = end
    return spans
