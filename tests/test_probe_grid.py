"""Step 9 v2: 4x4 cell pooling, Gemma's token grid, the probe's shapes, the shuffled-video control."""
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    from video_train import probe_grid as pg
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class GridTests(unittest.TestCase):
    def test_grid_pool_matches_torch_adaptive_pool(self):
        import torch
        for rows, cols in [(8, 15), (10, 13), (14, 14), (16, 16), (8, 17)]:
            x = np.random.default_rng(0).standard_normal((rows * cols, 5)).astype(np.float32)
            ours = pg.grid_pool(x, rows, cols)
            ref = torch.nn.functional.adaptive_avg_pool2d(torch.tensor(x).reshape(rows, cols, 5).permute(2, 0, 1), 4)
            np.testing.assert_allclose(ours, ref.permute(1, 2, 0).reshape(16, 5).numpy(), rtol=1e-5, atol=1e-6)

    def test_grid_pool_keeps_regions_apart(self):
        x = np.zeros((8, 16, 1), np.float32)
        x[:, 12:] = 1.0  # only the right quarter is bright
        cells = pg.grid_pool(x.reshape(-1, 1), 8, 16).reshape(4, 4)
        np.testing.assert_array_equal(cells[:, 3], 1.0)
        np.testing.assert_array_equal(cells[:, :3], 0.0)

    def test_gemma_cells_follow_frames(self):
        frames, rows, cols = 2, 8, 15
        f = np.zeros((1, frames * rows * cols, 3), np.float32)
        f[0, rows * cols:] = 1.0  # second frame
        cells = pg.gemma_cells(f, frames, rows, cols)
        self.assertEqual(cells.shape, (32, 3))
        np.testing.assert_array_equal(cells[:16], 0.0)
        np.testing.assert_array_equal(cells[16:], 1.0)
        with self.assertRaises(ValueError):
            pg.gemma_cells(f, frames, 10, 13)

    def test_probe_shapes_and_text_only(self):
        import torch
        B, K = 4, 3
        last, opts = torch.randn(B, 32), torch.randn(B, K, 32)
        self.assertEqual(tuple(pg.build_probe({"_text": 32})(last, opts, {}).shape), (B, K))
        both = pg.build_probe({"_text": 32, "gemma": 16, "vjepa2": 8})
        video = {"gemma": (torch.randn(B, 256, 16), torch.rand(B, 256)), "vjepa2": (torch.randn(B, 128, 8), torch.rand(B, 128))}
        self.assertEqual(tuple(both(last, opts, video).shape), (B, K))

    def test_shuffle_map_never_keeps_a_video(self):
        vids = {f"v{i}" for i in range(531)}
        m = pg.shuffle_map(vids)
        self.assertEqual(set(m), vids)
        self.assertEqual(set(m.values()), vids)
        self.assertTrue(all(k != v for k, v in m.items()))


try:
    from video_train.inputs import Gemma4Inputs
    from video_train.models import CONFIG
    import os
    HAVE_CONFIG = os.path.isdir(CONFIG)
except ImportError:
    HAVE_CONFIG = False


@unittest.skipUnless(HAVE_TORCH and HAVE_CONFIG, "needs the Gemma 4 config")
class GemmaGridTests(unittest.TestCase):
    def test_grid_follows_aspect_ratio(self):
        enc = Gemma4Inputs(CONFIG)
        self.assertEqual(pg.gemma_grid(enc, 1920, 1080), (8, 15))
        self.assertEqual(pg.gemma_grid(enc, 640, 480), (10, 13))
        self.assertEqual(pg.gemma_grid(enc, 1080, 1920), (15, 8))


if __name__ == "__main__":
    unittest.main()
