"""Frame selection for step 3: choose K of N uniformly sampled candidate frames.

The baseline feeds 16 frames at the midpoints of 16 equal time bins. These selectors decode
N candidates the same way (midpoints of N bins, default 32) and keep K of them, in time order,
with the candidates' own timestamps:

- ``change``: no model. Frames are placed where the picture changes. Each candidate gets a
  weight = mean absolute difference (64x64 grayscale) from the previous candidate, plus a floor
  so static stretches still get frames; the K frames sit at equal steps of cumulative weight.
  With uniform change this gives (nearly) uniform frames, so it only departs from the baseline
  where the video has bursts of change. One selection per video.
- ``question``: SigLIP 2 scores each candidate against the question text (and each option);
  a frame's score is its best cosine similarity over those texts. Coverage is kept: the
  candidates are split into K/2 equal time groups and each group's best frame is kept, the
  other K/2 are the best remaining frames overall. One selection per question, so questions
  of a video only share an encoding when they pick the same frames.

Selections are deterministic; ties go to the earlier frame.
"""
from __future__ import annotations

import numpy as np

SELECTORS = ("uniform", "change", "question")
SIGLIP = "siglip2-base-patch16-224"  # models/<name>, staged from the drive like the Gemma weights
CHANGE_FLOOR = 0.25  # share of the mean change given to every candidate


def _gray(frames, side=64):
    from PIL import Image
    return np.stack([np.asarray(f.convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0
                     for f in frames])


def change_weights(frames) -> np.ndarray:
    """Per-candidate change from the previous candidate; the first frame gets the mean change."""
    g = _gray(frames)
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2)) if len(g) > 1 else np.zeros(0)
    mean = float(d.mean()) if d.size else 0.0
    w = np.concatenate([[mean], d])
    return w + CHANGE_FLOOR * mean + 1e-6


def pick_by_weight(weights, k: int) -> list[int]:
    """K distinct indices at equal steps (bin midpoints) of the cumulative weight."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    if k >= n:
        return list(range(n))
    cum = np.cumsum(w)
    targets = (np.arange(k) + 0.5) * cum[-1] / k
    picked = []
    for t in targets:
        i = int(np.searchsorted(cum, t - 1e-9 * cum[-1]))  # a target on a boundary goes to the earlier frame
        picked.append(min(i, n - 1))
    out = sorted(set(picked))
    if len(out) < k:  # several targets fell in one heavy frame: add its nearest free neighbours
        free = [i for i in range(n) if i not in out]
        dist = lambda i: min(abs(i - j) for j in out)
        out = sorted(out + sorted(free, key=lambda i: (dist(i), -w[i], i))[:k - len(out)])
    return out


def select_change(frames, k: int) -> list[int]:
    return pick_by_weight(change_weights(frames), k)


def pick_with_coverage(scores, k: int) -> list[int]:
    """Best frame of each of K//2 equal time groups, then the best remaining frames overall."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if k >= n:
        return list(range(n))
    groups = np.array_split(np.arange(n), max(1, k // 2))
    out = {int(g[np.argmax(s[g])]) for g in groups if len(g)}
    for i in sorted(range(n), key=lambda i: (-s[i], i)):
        if len(out) >= k:
            break
        out.add(i)
    return sorted(out)


class QuestionScorer:
    """SigLIP 2 image-text similarity on torch (MPS when available)."""

    def __init__(self, path: str):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float32).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(path)

    def image_embeddings(self, frames) -> np.ndarray:
        with self.torch.no_grad():
            x = self.proc(images=list(frames), return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**x)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
        e = e.float().cpu().numpy()
        return e / np.linalg.norm(e, axis=-1, keepdims=True)

    def text_embeddings(self, texts) -> np.ndarray:
        # SigLIP 2 was trained on lowercase text padded to 64 tokens.
        with self.torch.no_grad():
            x = self.proc(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                          truncation=True, return_tensors="pt").to(self.device)
            e = self.model.get_text_features(**x)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
        e = e.float().cpu().numpy()
        return e / np.linalg.norm(e, axis=-1, keepdims=True)


def question_texts(question: str, options) -> list[str]:
    return [question] + [f"{question} {o}" for o in options]


def question_scores(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Per frame: best cosine similarity over the question texts."""
    return (image_emb @ text_emb.T).max(axis=1)


def select_question(image_emb, text_emb, k: int) -> list[int]:
    return pick_with_coverage(question_scores(image_emb, text_emb), k)
