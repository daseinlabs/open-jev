"""Video prefill on a tiny random Gemma 4 multimodal model (no downloads)."""
import unittest

import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration

from openjev.torch_backend import TorchBackend

IMG, VID, BOI, EOI, PATCH, POOL = 37, 40, 38, 39, 4, 2
FRAME = 16
TOKENS_PER_FRAME = (FRAME // PATCH) ** 2 // POOL ** 2


class TorchVideoGemma4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = Gemma4Config(
            text_config=dict(vocab_size=48, vocab_size_per_layer_input=48, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             global_head_dim=8, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
                             sliding_window=6, max_position_embeddings=256, pad_token_id=0,
                             layer_types=['sliding_attention', 'full_attention'] * 2,
                             use_bidirectional_attention=None),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                               num_key_value_heads=2, head_dim=8, global_head_dim=8, patch_size=PATCH,
                               pooling_kernel_size=POOL, position_embedding_size=64,
                               default_output_length=TOKENS_PER_FRAME),
            audio_config=None, image_token_id=37, video_token_id=VID, boi_token_id=BOI, eoi_token_id=EOI,
            audio_token_id=41, boa_token_id=42, eoa_token_id=43)
        engine = TorchBackend.__new__(TorchBackend)
        engine.torch, engine.device = torch, torch.device('cpu')
        engine.dtype = engine.vision_dtype = torch.float32
        engine.frame_mode, engine.frame_soft_tokens = 'video', None
        engine.model = Gemma4ForConditionalGeneration(config).eval()
        cls.engine = engine

    def setUp(self):
        self.engine.frame_mode = 'video'

    def inputs(self, n_frames, text=(5, 6, 7, 8, 9, 10, 11), soft=VID):
        ids = [2]
        for i in range(n_frames):
            ids += ([12, 13] if soft == VID else []) + [BOI] + [soft] * TOKENS_PER_FRAME + [EOI]
        ids += list(text)
        side = FRAME // PATCH
        xs, ys = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='xy')
        pos = torch.stack([xs, ys], -1).reshape(-1, 2)[None].repeat(n_frames, 1, 1)
        return {'input_ids': ids, 'token_type_ids': torch.tensor([int(i == soft) for i in ids]),
                'pixel_values': torch.rand(n_frames, side * side, PATCH * PATCH * 3), 'position_ids': pos}

    def hf(self, enc, extra=()):
        with torch.inference_mode():
            return self.engine.model(input_ids=torch.tensor([enc['input_ids'] + list(extra)]),
                                     pixel_values_videos=enc['pixel_values'][None],
                                     video_position_ids=enc['position_ids'][None]).logits[0]

    def test_matches_hf_forward_for_any_chunking(self):
        enc = self.inputs(3)
        ref = self.hf(enc)[-1]
        for chunk in (1, 5, 9, 1000):
            for frame_batch in (1, 2, 3):
                with self.subTest(chunk=chunk, frame_batch=frame_batch):
                    (_, n), last = self.engine.prefill_multimodal(enc, frame_batch=frame_batch, chunk_tokens=chunk)
                    self.assertEqual(n, len(enc['input_ids']))
                    torch.testing.assert_close(last, ref, atol=1e-5, rtol=1e-5)

    def test_frames_as_images_match_hf_forward(self):
        self.engine.frame_mode = 'images'
        enc = self.inputs(3, soft=IMG)
        with torch.inference_mode():
            ref = self.engine.model(input_ids=torch.tensor([enc['input_ids']]), pixel_values=enc['pixel_values'],
                                    image_position_ids=enc['position_ids']).logits[0, -1]
        for chunk, frame_batch in ((1, 1), (5, 2), (1000, 3)):
            with self.subTest(chunk=chunk, frame_batch=frame_batch):
                (_, _), last = self.engine.prefill_multimodal(enc, frame_batch=frame_batch, chunk_tokens=chunk)
                torch.testing.assert_close(last, ref, atol=1e-5, rtol=1e-5)

    def test_video_changes_the_output(self):
        enc = self.inputs(2)
        (_, _), a = self.engine.prefill_multimodal(enc)
        (_, _), b = self.engine.prefill_multimodal(dict(enc, pixel_values=torch.rand_like(enc['pixel_values'])))
        self.assertGreater((a - b).abs().max().item(), 1e-4)

    def test_precomputed_features(self):
        enc = self.inputs(3)
        (_, _), fresh = self.engine.prefill_multimodal(enc)
        feats = self.engine.image_features(enc['pixel_values'], frame_batch=2, position_ids=enc['position_ids'])
        self.assertEqual(tuple(feats.shape), (3, TOKENS_PER_FRAME, 32))
        (_, _), reused = self.engine.prefill_multimodal(dict(enc, pixel_values=None), image_features=feats)
        torch.testing.assert_close(reused, fresh)

    def test_options_after_video_match_full_sequence(self):
        enc = self.inputs(2)
        prefix, last = self.engine.prefill_multimodal(enc, chunk_tokens=4)
        options = [[14, 15], [16], [17, 18, 19]]
        got = self.engine.score_with_prefix(prefix, last, options, batch_size=2, pad_id=0)
        for opt, score in zip(options, got):
            logits = self.hf(enc, opt)
            pred = logits[len(enc['input_ids']) - 1:-1].log_softmax(-1)
            want = pred.gather(-1, torch.tensor(opt)[:, None]).sum().item()
            self.assertAlmostEqual(score, want, places=4)


if __name__ == '__main__':
    unittest.main()
