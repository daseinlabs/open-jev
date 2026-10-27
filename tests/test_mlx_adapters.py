"""Per-request video adapter on a tiny random Gemma 4 (MLX).

- outside a video scope the model gives exactly the base outputs;
- inside it, the same outputs as the LoRA model it was trained as;
- concurrent threads with different scopes each get their own weights.
"""
import os
import tempfile
import threading
import unittest

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from tests.test_mlx_grouping import HIDDEN, IMG, tiny_model
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False


@unittest.skipUnless(HAVE_MLX, "needs mlx-vlm")
class MlxAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from video_train import adapters, mlx_model as mm
        cls.ad, cls.mm = adapters, mm
        mx.random.seed(0)
        base = tiny_model()
        mx.eval(base.parameters())
        base.audio_tower = base.embed_audio = None
        cls.base_params = dict(tree_flatten(base.parameters()))
        trained = mm.attach_lora(base, rank=4, alpha=8, checkpoint=False)
        flat = [(k, mx.random.normal(v.shape) * 0.2 if k.endswith("lora_b") else v)
                for k, v in tree_flatten(trained.trainable_parameters())]
        trained.update(tree_unflatten(flat))
        trained.eval()
        cls.tmp = tempfile.mkdtemp()
        adapters.save_adapter(trained, cls.tmp, rank=4, alpha=8)
        cls.ids = [2, 52] + [IMG] * 4 + [53, 5, 6, 7, 8, 9]
        cls.feats = mx.random.normal((1, 4, HIDDEN))
        cls.labels = [30, 31, 32]
        cls.want_lora = cls.logits(trained)
        # A fresh copy of the base model, with the adapter installed for serving.
        serve = tiny_model()
        serve.update(tree_unflatten(list(cls.base_params.items())))
        serve.audio_tower = serve.embed_audio = None
        serve.eval()
        cls.want_base = cls.logits(serve)
        cls.n_layers = adapters.install_adapter(serve, adapters.VIDEO, cls.tmp)
        cls.serve = serve

    @classmethod
    def logits(cls, model):
        out = cls.mm.letter_logits(model, cls.ids, cls.feats, "images", cls.labels)
        mx.eval(out)
        return out

    def test_installs_on_language_model_only(self):
        self.assertGreater(self.n_layers, 0)
        keys = list(mx.load(os.path.join(self.tmp, "adapter.safetensors")))
        self.assertTrue(keys and all(k.startswith("language_model.") for k in keys))

    def test_base_outside_scope_and_for_non_video_requests(self):
        self.assertTrue(mx.allclose(self.logits(self.serve), self.want_base, atol=1e-5))
        with self.ad.request_scope(has_video=False):
            self.assertTrue(mx.allclose(self.logits(self.serve), self.want_base, atol=1e-5))

    def test_video_request_uses_adapter(self):
        with self.ad.request_scope(has_video=True):
            got = self.logits(self.serve)
        self.assertTrue(mx.allclose(got, self.want_lora, atol=1e-4))
        self.assertFalse(mx.allclose(self.want_lora, self.want_base, atol=1e-3))
        self.assertIsNone(self.ad.active_adapter())  # scope ended

    def test_concurrent_requests_keep_their_own_weights(self):
        barrier, errors = threading.Barrier(4), []

        def worker(has_video):
            try:
                with self.ad.request_scope(has_video=has_video):
                    barrier.wait()  # all four requests are inside their scopes at once
                    for _ in range(5):
                        got = self.logits(self.serve)
                        want = self.want_lora if has_video else self.want_base
                        if not mx.allclose(got, want, atol=1e-4):
                            errors.append(has_video)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(v,)) for v in (True, False, True, False)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
