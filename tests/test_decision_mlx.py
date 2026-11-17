"""GPU checks: OPENJEV_TEST_MLX=1 python -m unittest discover -s tests -p test_decision_mlx.py."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('OPENJEV_TEST_MLX') == '1', 'requires opt-in MLX/Metal device')
class MLXDecisionTests(unittest.TestCase):
    def test_ce_brier_loss_masks_padding_and_has_finite_gradients(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from openjev.head import _loss
        class FixedHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = mx.array([[0., 0., -1e9], [0., 0., 0.]])
            def __call__(self, *args):
                return self.logits
        head = FixedHead()
        mask = mx.array([[1., 1., 0.], [1., 1., 1.]])
        terms = [((None, None, None, mask, mx.array([0, 2])), None, 1.0, 0.7)]
        ce = (np.log(2)+np.log(3))/2
        brier = (0.5+2/3)/2
        value, grads = nn.value_and_grad(head, _loss)(head, terms)
        self.assertAlmostEqual(value.item(), ce+0.7*brier, places=6)
        self.assertTrue(np.isfinite(np.array(grads['logits'])).all())
        self.assertEqual(float(grads['logits'][0, 2].item()), 0.0)

    def test_masking_permutation_reload_and_repeated_requests(self):
        import mlx.core as mx
        import numpy as np
        from openjev.head import AttentionHead
        mx.random.seed(42)
        head=AttentionHead(8,4)
        head.eval()
        ctx=mx.random.normal((1,5,8))
        opt=mx.random.normal((1,3,8))
        original=head(ctx,mx.ones((1,5)),opt,mx.ones((1,3)))
        perm=[2,0,1]
        reordered=head(ctx,mx.ones((1,5)),opt[:,perm,:],mx.ones((1,3)))
        mx.eval(original,reordered)
        np.testing.assert_allclose(np.array(original)[:,perm],np.array(reordered),atol=1e-6)
        padded_ctx=mx.concatenate([ctx,mx.ones((1,2,8))*100],axis=1)
        padded_opt=mx.concatenate([opt,mx.ones((1,1,8))*100],axis=1)
        padded=head(padded_ctx,mx.array([[1,1,1,1,1,0,0]]),padded_opt,mx.array([[1,1,1,0]]))
        np.testing.assert_allclose(np.array(original),np.array(padded[:,:3]),atol=1e-6)
        mx.eval(head(ctx*2,mx.ones((1,5)),opt*3,mx.ones((1,3))))
        repeated=head(ctx,mx.ones((1,5)),opt,mx.ones((1,3)))
        np.testing.assert_array_equal(np.array(original),np.array(repeated))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'head.safetensors'
            head.save(str(p),{'test':True})
            restored,_=AttentionHead.load(str(p))
            np.testing.assert_array_equal(np.array(original),np.array(restored(ctx,mx.ones((1,5)),opt,mx.ones((1,3)))))


if __name__ == '__main__':
    unittest.main()
