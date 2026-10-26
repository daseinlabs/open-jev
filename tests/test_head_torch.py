"""The PyTorch head on CPU: loss, masking, reload, and parity with the MLX head when MLX is present."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from openjev import compute
from openjev import head_torch as T


class TorchHeadTests(unittest.TestCase):
    def setUp(self):
        compute.select('torch', 'cpu')
        self.addCleanup(compute.select)
        rng = np.random.default_rng(0)
        self.ctx = rng.normal(size=(2, 5, 8)).astype(np.float16)
        self.opt = rng.normal(size=(2, 3, 8)).astype(np.float16)
        self.cm, self.om = np.ones((2, 5), np.float32), np.ones((2, 3), np.float32)

    def test_masking_permutation_and_reload(self):
        head = T.new_head(8, 4, seed=1)
        original = T.forward(head, self.ctx, self.cm, self.opt, self.om)
        perm = [2, 0, 1]
        np.testing.assert_allclose(original[:, perm], T.forward(head, self.ctx, self.cm, self.opt[:, perm], self.om), atol=1e-6)
        ctx = np.concatenate([self.ctx, np.full((2, 2, 8), 100, np.float16)], 1)
        opt = np.concatenate([self.opt, np.full((2, 1, 8), 100, np.float16)], 1)
        cm = np.concatenate([self.cm, np.zeros((2, 2), np.float32)], 1)
        om = np.concatenate([self.om, np.zeros((2, 1), np.float32)], 1)
        padded = T.forward(head, ctx, cm, opt, om)
        np.testing.assert_allclose(original, padded[:, :3], atol=1e-5)
        self.assertTrue((padded[:, 3] < -1e8).all())
        with tempfile.TemporaryDirectory() as d:
            head.save(str(Path(d)/'head.safetensors'), {'test': True})
            restored, config = T.AttentionHead.load(str(Path(d)/'head.safetensors'))
        self.assertTrue(config['test'])
        np.testing.assert_array_equal(original, T.forward(restored, self.ctx, self.cm, self.opt, self.om))

    def test_ce_brier_loss_and_snapshot(self):
        class Fixed(T.AttentionHead):
            def forward(self, *args):
                return self.query.weight.sum() * 0 + torch.tensor([[0., 0., -1e9], [0., 0., 0.]])
        head = Fixed(8, 4)
        mask = np.array([[1., 1., 0.], [1., 1., 1.]], np.float32)
        batch = (self.ctx, self.cm, self.opt, mask, np.array([0, 2]))
        loss = T.Optimizer(head, 0.).step([(batch, None, 1.0, 0.7)])
        self.assertAlmostEqual(loss, (math.log(2) + math.log(3)) / 2 + 0.7 * (0.5 + 2 / 3) / 2, places=5)
        head = T.new_head(8, 4, seed=1)
        state, before = T.snapshot(head), T.forward(head, self.ctx, self.cm, self.opt, self.om)
        T.Optimizer(head, 1e-2).step([((self.ctx, self.cm, self.opt, self.om, np.array([0, 1])), np.array([1., -1.]), 1.0, 0.)])
        self.assertFalse(np.allclose(before, T.forward(head, self.ctx, self.cm, self.opt, self.om)))
        T.restore(head, state)
        np.testing.assert_array_equal(before, T.forward(head, self.ctx, self.cm, self.opt, self.om))

    @unittest.skipUnless(importlib.util.find_spec('mlx'), 'needs MLX')
    def test_weights_load_across_backends(self):
        from openjev import head as M
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/'head.safetensors')
            M.new_head(8, 4, seed=3).save(path, {})
            mlx_head, _ = M.AttentionHead.load(path)
            torch_head, _ = T.AttentionHead.load(path)
        import mlx.core as mx
        batch = (self.ctx, self.cm, self.opt, self.om, np.array([0, 2]))
        with mx.stream(mx.cpu):  # Metal matmuls are lower precision; the CPU matches to float32
            np.testing.assert_allclose(M.forward(mlx_head, *batch[:4]), T.forward(torch_head, *batch[:4]), atol=1e-5)
            optimizers = [ops.Optimizer(h, 1e-2, weight_decay=1e-2, clip=1.0) for ops, h in ((M, mlx_head), (T, torch_head))]
            for _ in range(5):  # same optimizer semantics, so training stays in step
                self.assertAlmostEqual(*[o.step([(batch, None, 1.0, 0.5)]) for o in optimizers], places=4)
            np.testing.assert_allclose(M.forward(mlx_head, *batch[:4]), T.forward(torch_head, *batch[:4]), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
