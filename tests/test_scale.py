"""Step 8: each run changes one setting against s7_A, and the decision rule against s7_A."""
import unittest
from unittest import mock

try:
    from video_train import distill, scale
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False


def _rows(correct):
    return {("v", i): {"correct": c, "probs": [0.6, 0.4] if c else [0.4, 0.6], "answer_id": 0,
                       "latency_s": 1.0, "decode_s": 0.0, "area": "physics", "reasoning": "descriptive"}
            for i, c in enumerate(correct)}


@unittest.skipUnless(HAVE_MLX, "needs mlx")
class ScaleTests(unittest.TestCase):
    def _train_args(self, extra):
        seen = {}
        with mock.patch.object(distill, "train", side_effect=lambda a: seen.setdefault("args", a)):
            distill.main(["train", "--variant", "A", "--lr", scale.LR, "--tag", "t", "--step", scale.STEP, *extra])
        return vars(seen["args"])

    def test_one_change_per_run(self):
        base = self._train_args([])
        self.assertEqual(base["step"], "step8")
        for tag, flags in scale.RUNS.items():
            args = self._train_args(flags)
            changed = {k for k in args if args[k] != base[k]}
            if tag == "s8_A_r32":
                self.assertEqual(changed, {"rank", "lora_alpha"})
                # same update scale as s7_A: alpha / rank = 16 / 8
                self.assertEqual(args["lora_alpha"] / args["rank"], base["lora_alpha"] / base["rank"])
            else:
                self.assertEqual(changed, {"epochs"})

    def test_step7_checkpoints_unchanged(self):
        self.assertEqual(distill.ckpt_dir("x"), distill.storage.path("checkpoints", "step7", "x"))
        self.assertTrue(scale.adapter_dir(scale.S7).endswith("checkpoints/step7/" + scale.S7))
        self.assertTrue(scale.adapter_dir("s8_A_r32").endswith("checkpoints/step8/s8_A_r32"))

    def test_verdict(self):
        s7 = _rows([True] * 50 + [False] * 150)
        clear = _rows([True] * 50 + [True] * 40 + [False] * 110)   # +40, p << 0.05
        noise = _rows([True] * 50 + [True] * 3 + [False] * 147)    # +3, a tie
        worse = _rows([False] * 30 + [True] * 20 + [False] * 150)  # -30
        self.assertEqual(scale.verdict(s7, {"a": noise, "b": worse})[0], None)
        win, tests = scale.verdict(s7, {"a": noise, "b": clear})
        self.assertEqual(win, "b")
        self.assertEqual(tests["b"][:2], (40, 0))
        better = _rows([True] * 50 + [True] * 60 + [False] * 90)
        self.assertEqual(scale.verdict(s7, {"b": clear, "c": better})[0], "c")  # both pass: more accurate


if __name__ == "__main__":
    unittest.main()
