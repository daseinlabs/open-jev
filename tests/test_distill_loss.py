"""Step 7 loss: teacher softening and alpha * CE + (1 - alpha) * KL over option letters."""
import math
import unittest

import numpy as np

try:
    import mlx.core as mx
    from video_train.distill import distill_loss, soften
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False


@unittest.skipUnless(HAVE_MLX, "needs mlx")
class DistillLossTests(unittest.TestCase):
    def test_soften(self):
        p = np.array([[0.7, 0.2, 0.1], [1.0, 0.0, 0.0]])
        q = soften(p, 1.5)
        np.testing.assert_allclose(q.sum(1), 1.0)
        expect = p[0] ** (1 / 1.5) / (p[0] ** (1 / 1.5)).sum()
        np.testing.assert_allclose(q[0], expect)
        self.assertLess(q[0].max(), 0.7)          # flatter
        np.testing.assert_allclose(q[1], p[1])    # a one-hot stays one-hot
        np.testing.assert_allclose(soften(p, 1.0), p)

    def _case(self):
        logits = mx.array([[2.0, 0.5, -1.0], [0.0, 1.0, 0.0]])
        answers = mx.array([0, 2])
        teacher = mx.array([[0.6, 0.3, 0.1], [0.0, 0.5, 0.5]])
        return logits, answers, teacher

    def _reference(self, logits, answers, teacher):
        lg = np.array(logits, dtype=np.float64)
        logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        t = np.array(teacher, dtype=np.float64)
        ce = -logp[np.arange(len(lg)), np.array(answers)]
        kl = np.array([sum(q * (math.log(q) - lp) for q, lp in zip(tr, lr) if q > 0) for tr, lr in zip(t, logp)])
        return ce, kl

    def test_parts_and_mix(self):
        logits, answers, teacher = self._case()
        ce, kl = self._reference(logits, answers, teacher)
        for alpha in (1.0, 0.5, 0.0):
            loss, c, k = distill_loss(logits, answers, teacher, alpha)
            self.assertAlmostEqual(c.item(), ce.sum(), places=5)
            self.assertAlmostEqual(k.item(), kl.sum(), places=5)   # zero teacher entries contribute 0
            self.assertAlmostEqual(loss.item(), alpha * ce.sum() + (1 - alpha) * kl.sum(), places=5)

    def test_kl_zero_when_student_matches_teacher(self):
        teacher = mx.array([[0.6, 0.3, 0.1]])
        _, _, k = distill_loss(mx.log(teacher), mx.array([0]), teacher, 0.0)
        self.assertAlmostEqual(k.item(), 0.0, places=6)

    def test_gradients(self):
        """d loss / d logits = alpha (p - onehot) + (1 - alpha) (p - q), per question."""
        logits, answers, teacher = self._case()
        alpha = 0.5
        g = mx.grad(lambda z: distill_loss(z, answers, teacher, alpha)[0])(logits)
        p = np.array(mx.softmax(logits, axis=-1))
        onehot = np.eye(3)[np.array(answers)]
        np.testing.assert_allclose(np.array(g), alpha * (p - onehot) + (1 - alpha) * (p - np.array(teacher)),
                                   atol=1e-6)


if __name__ == "__main__":
    unittest.main()
