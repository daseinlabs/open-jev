"""Step 11: video folds, temperature fitting, which adapter reads which question, the head's start."""
import math
import unittest
from unittest import mock

import numpy as np

try:
    from video_train import decision_head as dh
    from video_train import distill
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _rows(n_videos=40, per_video=3, T_true=2.0, seed=0):
    """Rows whose probabilities are softmax(z * T_true): calibrated at T = T_true."""
    rng = np.random.default_rng(seed)
    rows = {}
    for v in range(n_videos):
        for q in range(per_video):
            z = rng.normal(size=3)
            p_true = np.exp(z) / np.exp(z).sum()
            y = int(rng.choice(3, p=p_true))
            p = np.exp(z * T_true) / np.exp(z * T_true).sum()
            rows[(f"v{v}", q)] = {"probs": p.tolist(), "answer_id": y, "pred": int(p.argmax()),
                                  "correct": int(p.argmax()) == y, "area": "physics", "reasoning": None,
                                  "latency_s": 1.0, "decode_s": 0.0}
    return rows


@unittest.skipUnless(HAVE_MLX, "needs mlx")
class FoldTests(unittest.TestCase):
    def test_fold_split_is_fixed_balanced_and_order_free(self):
        ids = [f"video_{i}" for i in range(101)]
        a, b = distill.fold_of(ids, 2), distill.fold_of(list(reversed(ids)) + ids[:5], 2)
        self.assertEqual(a, b)
        self.assertEqual(sorted(np.bincount(list(a.values()))), [50, 51])

    def test_parse_fold(self):
        self.assertEqual(distill.parse_fold("1/2"), (1, 2))
        for bad in ("2/2", "0/1", "-1/2"):
            with self.assertRaises(ValueError):
                distill.parse_fold(bad)

    def test_fold_flag_reaches_train(self):
        seen = {}
        with mock.patch.object(distill, "train", side_effect=lambda a: seen.setdefault("args", a)):
            distill.main(["train", *dh.RECIPE, "--tag", "t", "--step", dh.STEP, "--fold", "0/2", "--dev-every", "0"])
        a = vars(seen["args"])
        self.assertEqual((a["fold"], a["epochs"], a["rank"], a["lr"], a["variant"]), ("0/2", 2, 8, 1e-4, "A"))

    def test_train_questions_are_read_by_the_other_half(self):
        fm = {"a": 0, "b": 1}
        self.assertEqual(dh.reader_of("train", "a", fm), "s11_half1")
        self.assertEqual(dh.reader_of("train", "b", fm), "s11_half0")
        self.assertEqual(dh.reader_of("dev", "a", fm), dh.FULL)
        self.assertEqual(dh.reader_of("nextqa", "a", {}), dh.FULL)
        self.assertEqual(set(dh.HALVES.values()), {"0/2", "1/2"})

    def test_holdout_is_seed_free_and_about_ten_percent(self):
        ids = [f"v{i}" for i in range(1000)]
        h = dh.holdout_videos(ids, 0.1)
        self.assertEqual(h, dh.holdout_videos(list(reversed(ids)), 0.1))
        self.assertEqual(len(h), 100)


@unittest.skipUnless(HAVE_MLX, "needs mlx")
class TemperatureTests(unittest.TestCase):
    def test_fit_recovers_the_overconfidence(self):
        rows = _rows(n_videos=800, T_true=2.0)
        keys = sorted(rows)
        T = dh.fit_temperature(dh.log_probs(rows, keys), np.array([rows[k]["answer_id"] for k in keys]))
        self.assertAlmostEqual(T, 2.0, delta=0.25)

    def test_temperature_never_changes_an_answer(self):
        rows = _rows()
        for T in (0.5, 1.0, 3.0):
            out = dh.apply_temperature(rows, T)
            self.assertTrue(all(out[k]["pred"] == rows[k]["pred"] for k in rows))

    def test_crossfit_scores_each_video_with_the_other_folds_T(self):
        rows = _rows(n_videos=60)
        cv, Ts, T = dh.crossfit(rows)
        folds = distill.fold_of([k[0] for k in rows], 2)
        keys = sorted(rows)
        for i in (0, 1):
            ks = [k for k in keys if folds[k[0]] == i]
            np.testing.assert_allclose([cv[k]["probs"] for k in ks], dh.tempered(dh.log_probs(rows, ks), Ts[i]))
        self.assertEqual(set(cv), set(rows))

    def test_verdict(self):
        base = {("v", i): {"correct": i < 50} for i in range(100)}
        better = {("v", i): {"correct": i < 80} for i in range(100)}
        self.assertEqual(dh.verdict(base, None)[0], "incomplete")
        self.assertEqual(dh.verdict(base, better)[0], "pass")
        self.assertEqual(dh.verdict(base, base)[0], "fail")


@unittest.skipUnless(HAVE_MLX and HAVE_TORCH, "needs mlx and torch")
class HeadTests(unittest.TestCase):
    def _batch(self, B=5, L=7, K=3, H=16):
        g = torch.Generator().manual_seed(0)
        h = torch.randn(B, L, H, generator=g)
        lengths = torch.tensor([7, 5, 6, 4, 7])[:B]
        mask = torch.arange(L)[None] < lengths[:, None]
        spans = torch.zeros(B, K, L, dtype=torch.bool)
        for b in range(B):
            for k in range(K):
                spans[b, k, k] = True
        return h, mask, spans, torch.randn(B, K, generator=g)

    def test_untrained_head_is_letter_scoring(self):
        head = dh.build_head(16, dict(dh.HP, width=8, heads=2)).eval()
        h, mask, spans, letter = self._batch()
        torch.testing.assert_close(head(h, mask, spans, letter), letter)

    def test_padding_does_not_leak(self):
        head = dh.build_head(16, dict(dh.HP, width=8, heads=2)).eval()
        torch.nn.init.normal_(head.mlp[-1].weight)
        h, mask, spans, letter = self._batch()
        h2 = h.clone()
        h2[~mask] = 100.0
        torch.testing.assert_close(head(h, mask, spans, letter), head(h2, mask, spans, letter))

    def test_batch_pads_and_keeps_spans(self):
        rows = [{"hidden": np.ones((n, 4), np.float16), "spans": np.eye(3, n, dtype=bool),
                 "letter": np.zeros(3, np.float32), "answer_id": 1} for n in (3, 5)]
        h, mask, spans, letter, y = dh._batch(rows, [0, 1], "cpu")
        self.assertEqual(tuple(h.shape), (2, 5, 4))
        self.assertEqual(mask.sum(1).tolist(), [3, 5])
        self.assertFalse(spans[0, :, 3:].any())
        self.assertEqual(y.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
