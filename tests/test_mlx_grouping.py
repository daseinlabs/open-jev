"""Per-video grouping on a tiny random float32 Gemma 4 (MLX): same loss and LoRA gradients as
one pass per question. Covers sliding-window layers (prefix longer than the window), KV-shared
layers, per-layer inputs, right-padded suffixes and the cache-safe gradient checkpointing."""
import unittest

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

IMG, VID, BOI, EOI = 50, 51, 52, 53
TOK_PER_FRAME, FRAMES, HIDDEN = 4, 3, 32


def tiny_model():
    d = {
        "model_type": "gemma4", "image_token_id": IMG, "video_token_id": VID, "boi_token_id": BOI,
        "eoi_token_id": EOI, "audio_token_id": 54, "audio_config": None,
        "text_config": {"model_type": "gemma4_text", "hidden_size": HIDDEN, "num_hidden_layers": 6,
                        "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
                        "head_dim": 8, "global_head_dim": 16, "vocab_size": 64, "vocab_size_per_layer_input": 64,
                        "hidden_size_per_layer_input": 8, "num_kv_shared_layers": 2, "sliding_window": 6,
                        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"] * 2,
                        "use_double_wide_mlp": False},
        "vision_config": {"model_type": "gemma4_vision", "hidden_size": 16, "intermediate_size": 32,
                          "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                          "head_dim": 8, "patch_size": 4, "pooling_kernel_size": 2, "default_output_length": 4},
    }
    cfg = ModelConfig.from_dict(d)
    cfg.text_config = TextConfig.from_dict(d["text_config"])
    cfg.vision_config = VisionConfig.from_dict(d["vision_config"])
    return Model(cfg)


@unittest.skipUnless(HAVE_MLX, "needs mlx-vlm")
class MlxGroupingTests(unittest.TestCase):
    def run_case(self, frames_as, checkpoint):
        from video_train import mlx_model as mm
        mx.random.seed(0)
        model = tiny_model()
        mx.eval(model.parameters())
        model.audio_tower = model.embed_audio = None
        model = mm.attach_lora(model, rank=4, alpha=8, checkpoint=checkpoint)
        flat = [(k, mx.random.normal(v.shape) * 0.1 if k.endswith("lora_b") else v)
                for k, v in tree_flatten(model.trainable_parameters())]
        model.update(tree_unflatten(flat))
        model.train()
        soft = VID if frames_as == "video" else IMG
        prefix = [2]
        for f in range(FRAMES):
            prefix += ([10 + f, 11] if frames_as == "video" else []) + [BOI] + [soft] * TOK_PER_FRAME + [EOI]
        suffixes = [[5, 6, 7, 8, 9], [12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22]]
        targets, labels = [0, 2, 1], [30, 31, 32]
        feats = mx.random.normal((FRAMES, TOK_PER_FRAME, HIDDEN))

        def per_question(m):
            return mx.mean(mx.stack([mm.letter_ce(mm.letter_logits(m, prefix + s, feats, frames_as, labels), t)
                                     for s, t in zip(suffixes, targets)]))

        def grouped(m):
            return mm.letter_ce(mm.letter_logits_grouped(m, prefix, feats, frames_as, suffixes, labels), targets)

        (l1, g1), (l2, g2) = (nn.value_and_grad(model, f)(model) for f in (per_question, grouped))
        g1, g2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
        self.assertAlmostEqual(l1.item(), l2.item(), places=5)
        num = sum(float(mx.sum((g1[k] - g2[k]) ** 2)) for k in g1)
        den = sum(float(mx.sum(g1[k] ** 2)) for k in g1)
        self.assertGreater(den, 0)
        self.assertLess((num / den) ** 0.5, 1e-4)
        return (num / den) ** 0.5

    def test_images_no_checkpoint(self):
        self.run_case("images", checkpoint=False)

    def test_video_with_checkpoint(self):
        self.run_case("video", checkpoint=True)


if __name__ == "__main__":
    unittest.main()
