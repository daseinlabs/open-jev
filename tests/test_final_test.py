"""Step 12: test is reachable only through final_test, a smoke test stays on dev, the report's numbers."""
import unittest
from unittest import mock

try:
    from video_train import evaluate, final_test as ft
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False


def _rows(correct, probs_ok=(0.7, 0.2, 0.1)):
    bad = (0.2, 0.7, 0.1)
    return {("v", i): {"correct": c, "probs": list(probs_ok if c else bad), "pred": 0 if c else 1, "answer_id": 0,
                       "latency_s": 1.0, "decode_s": 0.5, "area": "physics", "reasoning": "descriptive"}
            for i, c in enumerate(correct)}


@unittest.skipUnless(HAVE_MLX, "needs mlx")
class FinalTestTests(unittest.TestCase):
    def test_evaluate_refuses_test_without_final(self):
        with mock.patch.object(evaluate.storage, "require_root", side_effect=AssertionError("got past the guard")):
            with self.assertRaises(SystemExit) as e:
                evaluate.main(["run", "--set", "test", "--tag", "x"])
        self.assertIn("step 12", str(e.exception))

    def _scored(self, smoke):
        calls = []
        with mock.patch.object(ft, "_evaluate", side_effect=calls.append), \
                mock.patch.object(ft.storage, "require_root"):
            ft.main(["score"] + (["--smoke"] if smoke else []))
        return calls

    def test_score_runs_both_arms_on_test_with_final(self):
        calls = self._scored(False)
        self.assertEqual(len(calls), 2)
        for argv in calls:
            self.assertEqual(argv[argv.index("--set") + 1], "test")
            self.assertIn("--final", argv)
            self.assertNotIn("--limit-videos", argv)
        tags = [a[a.index("--tag") + 1] for a in calls]
        self.assertEqual(tags, [ft.ZERO, ft.FULL])
        self.assertIn("--adapter", calls[1])
        self.assertNotIn("--adapter", calls[0])

    def test_smoke_never_touches_test(self):
        for argv in self._scored(True):
            self.assertEqual(argv[argv.index("--set") + 1], "dev")
            self.assertNotIn("--final", argv)
            self.assertTrue(argv[argv.index("--tag") + 1].endswith("_step12smoke"))

    def test_summary_pairs_and_keeps_answers_under_temperature(self):
        zero, full = _rows([True] * 5 + [False] * 15), _rows([True] * 15 + [False] * 5)
        m, (g, lost, p) = ft.summarize(zero, full, 1.5)
        self.assertEqual((g, lost), (10, 0))
        self.assertEqual(m[ft.FULL]["accuracy"], m[f"{ft.FULL} + T"]["accuracy"])
        self.assertLess(m[f"{ft.FULL} + T"]["mean_confidence"], m[ft.FULL]["mean_confidence"])


if __name__ == "__main__":
    unittest.main()
