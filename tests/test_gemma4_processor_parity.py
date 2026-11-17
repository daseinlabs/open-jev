"""OpenJev's hand-built Gemma 4 frame inputs vs the official Gemma4Processor.

Needs the E4B tokenizer/processor files (models/gemma-4-E4B-it-config; no weights)
and torchvision (``pip install -e ".[parity]"``); skipped otherwise.
"""
import os
import types
import unittest

import numpy as np
import torch

MODEL = next((p for p in ("models/gemma-4-E4B-it-config", "models/gemma-4-E4B-it")
              if os.path.exists(f"{p}/processor_config.json")), "models/gemma-4-E4B-it-config")
TEXT = "Question: what happens next?\nChoose exactly one option.\nOptions:\n- A\n- B\n- C\n\nAnswer:\n"

try:
    import torchvision  # noqa: F401
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
    from transformers.video_utils import VideoMetadata
    HAVE_DEPS = os.path.exists(f"{MODEL}/processor_config.json")
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "needs torchvision and the Gemma 4 E4B config files")
class Gemma4ProcessorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openjev.scorer import OptionScorer
        cls.cfg, cls.tok = AutoConfig.from_pretrained(MODEL), AutoTokenizer.from_pretrained(MODEL)
        # Not AutoProcessor: importing mlx-vlm re-registers it for Gemma 4 with its own processor.
        cls.proc = Gemma4Processor.from_pretrained(MODEL)
        cls.OptionScorer = OptionScorer
        rng = np.random.default_rng(0)
        # Three "clips" with different aspect ratios: smooth gradients plus noise, 16 frames each.
        cls.clips = []
        for w, h in ((1920, 1080), (640, 480), (720, 1280)):
            base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
            frames = [Image.fromarray(np.clip(base * (0.5 + i / 32) + rng.normal(0, 20, (h, w, 3)), 0, 255)
                                      .astype(np.uint8)) for i in range(16)]
            cls.clips.append((frames, [(i + 0.5) * 14.7 / 16 for i in range(16)]))

    def backend(self, mode, soft=None):
        from openjev.torch_backend import TorchBackend
        b = TorchBackend.__new__(TorchBackend)
        b.torch, b.tok, b.model = torch, self.tok, types.SimpleNamespace(config=self.cfg)
        b.frame_mode, b.frame_soft_tokens, b.model_path, b._processor = mode, soft, MODEL, None
        return b

    def scorer(self, backend):
        s = self.OptionScorer.__new__(self.OptionScorer)
        s._engine, s.tok, s.chat, s.sep, s.bos_id = backend, self.tok, True, "", self.tok.bos_token_id
        return s

    def official(self, mode, frames, ts, chat, soft):
        media = [{"type": "image"}] * len(frames) if mode == "images" else [{"type": "video"}]
        if chat:
            text = self.proc.apply_chat_template([{"role": "user", "content": media + [{"type": "text", "text": TEXT}]}],
                                                 tokenize=False, add_generation_prompt=True)
        else:
            text = (self.proc.image_token * len(frames) if mode == "images" else self.proc.video_token) + TEXT
        if mode == "images":
            return self.proc(text=[text], images=[frames], return_tensors="pt", add_special_tokens=False,
                             images_kwargs={"max_soft_tokens": soft})
        # fps=1000 with frame indices t*1000 makes the processor's timestamps equal ``ts``.
        md = VideoMetadata(total_num_frames=10 ** 7, fps=1000.0, frames_indices=[round(t * 1000) for t in ts])
        return self.proc(text=[text], videos=[np.stack([np.asarray(f) for f in frames])], return_tensors="pt",
                         add_special_tokens=False,
                         videos_kwargs={"video_metadata": [md], "do_sample_frames": False, "max_soft_tokens": soft})

    def check(self, mode, soft, chat):
        for frames, ts in self.clips:
            b = self.backend(mode, soft)
            if chat:
                ours = self.scorer(b)._multimodal_inputs(TEXT, frames, chat=True, sep="", timestamps=ts)
            else:
                ours = b._encode_gemma4_frames(TEXT, frames, True, ts)
            ref = self.official(mode, frames, ts, chat, soft)
            ids = ours["input_ids"] if chat else ours["input_ids"][1:]  # raw text: the processor adds no <bos>
            self.assertEqual(ids, ref["input_ids"][0].tolist())
            pv = ref["pixel_values" if mode == "images" else "pixel_values_videos"]
            pos = ref["image_position_ids" if mode == "images" else "video_position_ids"]
            self.assertTrue(torch.equal(ours["position_ids"], pos.reshape(ours["position_ids"].shape)))
            # PIL vs torchvision bicubic resize: a few pixels differ by 1-2 levels out of 255.
            d = (ours["pixel_values"].float() - pv.float().reshape(ours["pixel_values"].shape)).abs()
            self.assertLessEqual(float(d.max()), 3 / 255)
            self.assertLess(float(d.mean()), 1e-3)

    def test_images_280(self):
        self.check("images", 280, chat=False)
        self.check("images", 280, chat=True)

    def test_images_560(self):
        self.check("images", 560, chat=False)

    def test_video_70(self):
        self.check("video", 70, chat=False)
        self.check("video", 70, chat=True)


if __name__ == "__main__":
    unittest.main()
