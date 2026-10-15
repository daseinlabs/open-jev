"""Image/video prefill on a tiny random Gemma 3 multimodal model (no downloads)."""
import unittest

import torch
from transformers import Gemma3Config, Gemma3ForConditionalGeneration

from openjev.torch_backend import TorchBackend, _chunks

IMG, TOKENS_PER_IMAGE = 30, 4


class TorchVideoPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = Gemma3Config(
            text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=6,
                             max_position_embeddings=256, layer_types=['sliding_attention', 'full_attention']),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=56, patch_size=14),
            mm_tokens_per_image=TOKENS_PER_IMAGE, image_token_index=IMG, boi_token_index=28, eoi_token_index=29)
        engine = TorchBackend.__new__(TorchBackend)
        engine.torch, engine.device = torch, torch.device('cpu')
        engine.dtype = engine.vision_dtype = torch.float32
        engine.model = Gemma3ForConditionalGeneration(config).eval()
        cls.engine = engine

    def inputs(self, n_images, text=(5, 6, 7, 8, 9, 10, 11)):
        ids = [1]
        for _ in range(n_images):
            ids += [3, 28] + [IMG] * TOKENS_PER_IMAGE + [29, 3]
        ids += list(text)
        tt = torch.tensor([int(i == IMG) for i in ids])
        return {'input_ids': ids, 'token_type_ids': tt, 'pixel_values': torch.randn(n_images, 3, 56, 56)}

    def test_matches_hf_forward_for_any_chunking(self):
        enc = self.inputs(3)
        with torch.inference_mode():
            ref = self.engine.model(input_ids=torch.tensor([enc['input_ids']]), pixel_values=enc['pixel_values'],
                                    token_type_ids=enc['token_type_ids'][None]).logits[0, -1]
        for chunk in (1, 5, 9, 1000):
            for frame_batch in (1, 2, 3):
                with self.subTest(chunk=chunk, frame_batch=frame_batch):
                    (_, n), last = self.engine.prefill_multimodal(enc, frame_batch=frame_batch, chunk_tokens=chunk)
                    self.assertEqual(n, len(enc['input_ids']))
                    torch.testing.assert_close(last, ref, atol=1e-5, rtol=1e-5)

    def test_precomputed_image_features(self):
        enc = self.inputs(3)
        (_, _), fresh = self.engine.prefill_multimodal(enc)
        feats = self.engine.image_features(enc['pixel_values'], frame_batch=3)
        self.assertEqual(tuple(feats.shape), (3, TOKENS_PER_IMAGE, 32))
        (_, _), reused = self.engine.prefill_multimodal(dict(enc, pixel_values=None), image_features=feats)
        torch.testing.assert_close(reused, fresh)

    def test_bidirectional_mask_is_applied(self):
        enc = self.inputs(2)
        (_, _), with_mask = self.engine.prefill_multimodal(enc)
        causal = dict(enc, token_type_ids=torch.zeros_like(enc['token_type_ids']))
        (_, _), without = self.engine.prefill_multimodal(causal)
        self.assertGreater((with_mask - without).abs().max().item(), 1e-4)

    def test_options_after_images_match_full_sequence(self):
        enc = self.inputs(2)
        prefix, last = self.engine.prefill_multimodal(enc, chunk_tokens=4)
        options = [[12, 13], [14], [15, 16, 17]]
        got = self.engine.score_with_prefix(prefix, last, options, batch_size=2, pad_id=0)
        for opt, score in zip(options, got):
            ids = enc['input_ids'] + opt
            tt = torch.cat([enc['token_type_ids'], torch.zeros(len(opt), dtype=torch.long)])
            with torch.inference_mode():
                logits = self.engine.model(input_ids=torch.tensor([ids]), pixel_values=enc['pixel_values'],
                                           token_type_ids=tt[None]).logits[0]
            pred = logits[len(enc['input_ids']) - 1:-1].log_softmax(-1)
            want = pred.gather(-1, torch.tensor(opt)[:, None]).sum().item()
            self.assertAlmostEqual(score, want, places=4)

    def test_chunks_never_split_an_image(self):
        tt = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
        for size in range(1, 15):
            spans = _chunks(tt, size)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(tt))
            for (_, e), (s, _) in zip(spans, spans[1:]):
                self.assertEqual(e, s)
                self.assertFalse(tt[e] == 1 and tt[e - 1] == 1)


if __name__ == '__main__':
    unittest.main()
