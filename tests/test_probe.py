"""Step 9: option token spans, Gemma pooling, the probe's shapes, the held-out split, the decision rule."""
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    from video_train import probe
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _rows(correct):
    return {("v", i): {"correct": c, "probs": [0.6, 0.4] if c else [0.4, 0.6], "answer_id": 0,
                       "latency_s": 0.0, "decode_s": 0.0, "area": "physics", "reasoning": "descriptive"}
            for i, c in enumerate(correct)}


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class ProbeTests(unittest.TestCase):
    def test_option_spans(self):
        prompt = "Question:\nIs it on?\n\nChoose exactly one option.\nOptions:\n- A: I don't know\n- B: no\n- C: yes\n\nAnswer:\n"
        # one "token" per word, with character offsets
        offsets, i = [], 0
        for w in prompt.replace("\n", " ").split(" "):
            if w:
                j = prompt.index(w, i)
                offsets.append((j, j + len(w)))
                i = j + len(w)
        spans = probe.option_spans(prompt, 3, offsets)
        words = [[prompt[offsets[k][0]:offsets[k][1]] for k in sp] for sp in spans]
        self.assertEqual(words, [["I", "don't", "know"], ["no"], ["yes"]])

    def test_pool_gemma_is_per_frame_mean(self):
        f = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 6, 4)  # 2 frames x 3 tokens
        out = probe.pool_gemma(f, 2)
        np.testing.assert_allclose(out, [f[0, :3].mean(0), f[0, 3:].mean(0)])
        with self.assertRaises(ValueError):
            probe.pool_gemma(f, 4)

    def test_probe_shapes(self):
        import torch
        B, K = 5, 3
        last, opts = torch.randn(B, 32), torch.randn(B, K, 32)
        text_only = probe.build_probe({"_text": 32})
        self.assertEqual(tuple(text_only(last, opts).shape), (B, K))
        both = probe.build_probe({"_text": 32, "gemma": 16, "vjepa2": 8})
        video = {"gemma": (torch.randn(B, 16, 16), torch.rand(B, 16)), "vjepa2": (torch.randn(B, 8, 8), torch.rand(B, 8))}
        self.assertEqual(tuple(both(last, opts, video).shape), (B, K))

    def test_holdout_is_fixed_and_about_ten_percent(self):
        vids = [f"video_{i}" for i in range(2000)]
        a, b = probe.holdout_videos(vids), probe.holdout_videos(list(reversed(vids)))
        self.assertEqual(a, b)
        self.assertTrue(150 < len(a) < 250)

    def test_arms_and_sources(self):
        self.assertEqual(probe.sources_of("text"), [])
        self.assertEqual(probe.sources_of("gemma+dinov2"), ["gemma", "dinov2"])
        self.assertEqual(set(probe.ARMS), {"text", "gemma", *probe.CANDIDATES,
                                           *(f"gemma+{x}" for x in probe.CANDIDATES)})

    def test_verdict(self):
        n = 400
        text = _rows([i < 200 for i in range(n)])
        gemma = _rows([i < 260 for i in range(n)])          # +60 over text
        same = _rows([i < 260 for i in range(n)])           # ties gemma
        better = _rows([i < 330 for i in range(n)])         # +70 over gemma
        slightly = _rows([i < 272 for i in range(n)])       # +12 over gemma: p ~ 0.0005 < 0.017
        rows = {"text": text, "gemma": gemma, **{x: same for x in probe.CANDIDATES},
                **{f"gemma+{x}": same for x in probe.CANDIDATES}}
        self.assertEqual(probe.verdict(rows)[0], "fail")
        rows["gemma+dinov2"], rows["gemma+vjepa2"] = slightly, better
        state, win, _ = probe.verdict(rows)
        self.assertEqual((state, win), ("pass", "vjepa2"))
        self.assertEqual(probe.verdict({**rows, "gemma": text})[0], "inconclusive")
        self.assertEqual(probe.verdict({**rows, "gemma+siglip2": None})[0], "incomplete")
        self.assertEqual(probe.verdict({"text": text})[0], "incomplete")

    def test_bonferroni_bar(self):
        n = 400
        gemma = _rows([i < 260 for i in range(n)])
        # 10 gained, 2 lost: p ~ 0.021, under 0.05 but over 0.05 / 3
        near = _rows([(i < 260 and i not in (0, 1)) or 260 <= i < 270 for i in range(n)])
        rows = {"text": _rows([i < 200 for i in range(n)]), "gemma": gemma,
                **{f"gemma+{x}": gemma for x in probe.CANDIDATES}, "gemma+siglip2": near}
        self.assertEqual(probe.verdict(rows)[0], "fail")


if __name__ == "__main__":
    unittest.main()
