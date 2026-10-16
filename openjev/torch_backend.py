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
        # Gemma 3 4B and Gemma 4 checkpoints wrap the language model in a multimodal model.
        # Use its native wrapper so checkpoint names and tied weights load correctly.
        if config.model_type == "gemma3":
            cls = Gemma3ForConditionalGeneration
        elif config.model_type == "gemma4":
            from transformers import Gemma4ForConditionalGeneration as cls
        else:
            cls = AutoModelForCausalLM
        model = cls.from_pretrained(model_path, torch_dtype=dtype)
        if config.model_type == "gemma4":
            _slim_gemma4(model, self.device)
        self.model = model.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model_path, self.dtype, self._processor = model_path, dtype, None
        # Gemma 4 only: how frames enter the model (see encode_multimodal) and their token budget.
        self.frame_mode, self.frame_soft_tokens = "images", None
        self.vision_dtype = getattr(torch, vision_dtype) if vision_dtype else dtype
        if vision_dtype and hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
            # Gemma 3 overflows in float16 in the language model but not in SigLIP, where
            # float16 tracks float32 far closer than bfloat16 does.
            self.model.model.vision_tower.to(self.vision_dtype)
            projector = getattr(self.model.model, "multi_modal_projector", None) or self.model.model.embed_vision
            projector.to(self.vision_dtype)

    @property
    def model_type(self) -> str | None:
        return getattr(self.model.config, "model_type", None)

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

    def encode_multimodal(self, text: str, images: list, add_bos: bool = True, timestamps=None) -> dict:
        """Token ids, token types and pixels for ``text`` preceded by ``images``, in order.

        Gemma 3: each image becomes the processor's full image sequence (``\\n\\n<start_of_image>``,
        256 soft tokens, ``<end_of_image>\\n\\n``) so layout and preprocessing match HF.
        Gemma 4, ``frame_mode="images"`` (default): each frame is a separate image, laid out as
        HF's image placeholder (``<|image>`` + soft tokens + ``<image|>``), 280 tokens per frame.
        Gemma 4, ``frame_mode="video"``: the frames are one video, laid out as HF's video placeholder
        (``mm:ss <|image>`` + soft tokens + ``<image|>`` per frame, space separated), 70 tokens per
        frame. ``timestamps`` are the frame times in seconds (default one frame per second).
        ``frame_soft_tokens`` overrides the per-frame budget (70, 140, 280, 560 or 1120).
        """
        model_type = self.model_type
        if model_type not in ("gemma3", "gemma4"):
            raise ValueError("image input needs a Gemma 3 or Gemma 4 multimodal checkpoint")
        if not images:
            raise ValueError("images must be non-empty")
        if model_type == "gemma4":
            return self._encode_gemma4_frames(text, list(images), add_bos, timestamps)
        proc = self.processor
        out = proc(text=proc.boi_token * len(images) + text, images=[list(images)],
                   return_tensors="pt", add_special_tokens=add_bos)
        return {"input_ids": out["input_ids"][0].tolist(), "token_type_ids": out["token_type_ids"][0],
                "pixel_values": out["pixel_values"]}

    @property
    def soft_tokens_per_frame(self) -> int:
        if self.frame_mode not in ("images", "video"):
            raise ValueError("frame_mode must be images or video")
        return self.frame_soft_tokens or (280 if self.frame_mode == "images" else 70)

    def _encode_gemma4_frames(self, text: str, frames: list, add_bos: bool, timestamps) -> dict:
        # Built by hand rather than with Gemma4Processor, which needs torchvision.
        tok, cfg = self.tok, self.model.config
        pix = self.preprocess_frames(frames)
        video = self.frame_mode == "video"
        soft_id = cfg.video_token_id if video else cfg.image_token_id
        boi, eoi, soft = (tok.convert_ids_to_tokens(i) for i in (cfg.boi_token_id, cfg.eoi_token_id, soft_id))
        frame = boi + soft * pix["tokens_per_frame"] + eoi
        if video:
            if timestamps is None:
                timestamps = list(range(len(frames)))
            if len(timestamps) != len(frames):
                raise ValueError("need one timestamp per frame")
            head = " ".join(f"{int(t // 60):02d}:{int(t % 60):02d} {frame}" for t in timestamps)
        else:  # consecutive images, as the chat template renders [image, image, ..., text]
            head = frame * len(frames)
        ids = tok.encode(head + text, add_special_tokens=False)
        if add_bos and tok.bos_token_id is not None and ids[:1] != [tok.bos_token_id]:
            ids = [tok.bos_token_id] + ids
        tt = self.torch.tensor([int(i == soft_id) for i in ids])
        if int(tt.sum()) != len(frames) * pix["tokens_per_frame"]:
            raise ValueError("the tokenizer did not keep the vision placeholder tokens intact")
        return {"input_ids": ids, "token_type_ids": tt, "pixel_values": pix["pixel_values"],
                "position_ids": pix["position_ids"]}

    def preprocess_frames(self, frames: list) -> dict:
        """Pixels for the vision tower: ``pixel_values`` (plus ``position_ids`` on Gemma 4)."""
        if self.model_type != "gemma4":
            return {"pixel_values": self.processor.image_processor(list(frames), return_tensors="pt")["pixel_values"]}
        # Gemma 4: aspect-preserving resize to ``soft_tokens_per_frame``, [0, 1] pixels, 16 px
        # patches. The PIL image processor does the same steps as the video processor, so video
        # mode does not need torchvision either.
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
        vc = self.model.config.vision_config
        ip = Gemma4ImageProcessorPil(patch_size=vc.patch_size, pooling_kernel_size=vc.pooling_kernel_size)
        out = ip([f.convert("RGB") for f in frames], max_soft_tokens=self.soft_tokens_per_frame, return_tensors="pt")
        counts = {int(n) for n in out["num_soft_tokens_per_image"]}
        if len(counts) != 1:
            raise ValueError("all frames of a video must have the same size")
        return {"pixel_values": out["pixel_values"], "position_ids": out["image_position_ids"],
                "tokens_per_frame": counts.pop()}

    def frame_features(self, frames: list, frame_batch: int = 2):
        """Vision-tower output for ``frames``, reusable via ``image_features=`` in prefill."""
        pix = self.preprocess_frames(frames)
        return self.image_features(pix["pixel_values"], frame_batch, position_ids=pix.get("position_ids"))

    def image_features(self, pixel_values, frame_batch: int = 2, position_ids=None):
        """Projected vision features, (n_images, tokens_per_image, hidden), ``frame_batch`` at a time."""
        torch = self.torch
        if self.model_type == "gemma4":
            with torch.inference_mode():
                feats = []
                for pv, pos in zip(pixel_values.split(frame_batch), position_ids.split(frame_batch)):
                    pv, pos = pv.to(self.device, self.vision_dtype), pos.to(self.device)
                    if self.frame_mode == "video":
                        out = self.model.model.get_video_features(pv[None], pos[None], return_dict=True)
                    else:
                        out = self.model.model.get_image_features(pv, pos, return_dict=True)
                    feats.append(torch.cat(out.pooler_output).reshape(len(pv), -1, out.pooler_output[0].shape[-1]))
                feats = torch.cat(feats).to(self.dtype)
                self._sync()
            return feats
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
        if self.model_type == "gemma4":
            return self._prefill_gemma4(enc, frame_batch, chunk_tokens, image_features)
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

    def _prefill_gemma4(self, enc: dict, frame_batch: int, chunk_tokens: int, image_features):
        """Gemma 4 version of :meth:`prefill_multimodal`, mirroring ``Gemma4Model.forward``.

        Vision soft tokens embed as pad for both the main and the per-layer (PLE) embedding,
        then the vision features replace the main embedding. E-series models attend causally
        to vision tokens (no bidirectional block mask), so any chunking is exact.
        """
        torch = self.torch
        model = self.model
        cfg = model.config
        ids = self._tensor([enc["input_ids"]])
        is_vid = (ids == cfg.video_token_id) | (ids == cfg.image_token_id)
        with torch.inference_mode():
            llm_ids = ids.masked_fill(is_vid, cfg.text_config.pad_token_id)
            embeds = model.get_input_embeddings()(llm_ids)
            lm = model.model.language_model
            ple = lm.get_per_layer_inputs(llm_ids, None) if cfg.text_config.hidden_size_per_layer_input else None
            if image_features is None:
                image_features = self.image_features(enc["pixel_values"], frame_batch, enc["position_ids"])
            feats = image_features.to(self.device, embeds.dtype).reshape(-1, embeds.shape[-1])
            if feats.shape[0] != int(is_vid.sum()):
                raise ValueError("vision token count does not match vision features")
            embeds = embeds.masked_scatter(is_vid[..., None], feats)
            del feats
            cache, n = None, ids.shape[1]
            for s in range(0, n, chunk_tokens):
                e = min(n, s + chunk_tokens)
                out = model(inputs_embeds=embeds[:, s:e], per_layer_inputs=None if ple is None else ple[:, s:e],
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                if self.device.type == "mps":
                    # The growing KV cache frees a block of a new size every chunk; MPS keeps them
                    # all cached, which ran 18 GB out of memory at 64 frames (17k tokens).
                    torch.mps.empty_cache()
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


def _slim_gemma4(model, device) -> None:
    """Drop the audio tower and keep the per-layer embedding table on the CPU.

    The PLE table (vocab x layers x 256, 2.8B parameters in E4B) is only ever indexed, so
    it stays in host memory instead of on the GPU; its rows are moved per call.
    """
    import torch

    class OffloadedEmbedding(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self._table = [embedding]  # a list, so model.to(device) leaves it on the CPU

        def forward(self, ids):
            return self._table[0](ids.cpu()).to(device)

    model.model.audio_tower = None
    model.model.embed_audio = None
    lm = model.model.language_model
    if getattr(lm, "embed_tokens_per_layer", None) is not None and device.type != "cpu":
        lm.embed_tokens_per_layer = OffloadedEmbedding(lm.embed_tokens_per_layer)


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
