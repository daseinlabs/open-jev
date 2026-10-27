"""Before step 2: do MLX 4-bit / 8-bit agree with torch bf16 on ~100 dev questions?

    caffeinate -ims python -m video_train.precision_check run --backend torch-bf16
    caffeinate -ims python -m video_train.precision_check run --backend mlx-8bit
    python -m video_train.precision_check report     # exits 2 if MLX 8-bit fails the gate

Zero-shot, 16 frames in Gemma 4's video layout at 70 tokens per frame with timestamps (the
training format; ``--frames-as images`` gives 280-token images), no chat template,
option-letter probabilities. Each question is scored with its own full pass; the vision
features of a video are computed once and reused for its questions. The questions are the
first whole dev videos (manifest order) reaching ``--questions``.

Per-question rows go to ``results/precision_check/<backend>.jsonl`` (a rerun resumes);
``report`` writes ``results/precision_check/summary.md`` and logs to history.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from . import storage
from .history import log_run
from .inputs import LABELS
from .models import CONFIG, local_model
from .perception_data import load_manifest

BACKENDS = {"torch-bf16": "gemma-4-E4B-it", "mlx-8bit": "gemma-4-E4B-it-mlx-8bit",
            "mlx-4bit": "gemma-4-E4B-it-mlx-4bit"}  # model names; staged from the drive when missing
TOKENIZER = CONFIG
# Gate (amended 2026-10-16, docs/video-training-plan.md): MLX 8-bit must agree with torch bf16 on
# >= 98% of the questions torch decides by a margin >= DECIDED, and must not shift P(correct)
# by more than MAX_SHIFT on average. Near-ties (incl. exact bf16 ties in torch) are excluded.
AGREE, DECIDED, MAX_SHIFT = 0.98, 0.1, 0.01
OUT = ("results", "precision_check")


def questions(n):
    """The fixed precision-check questions, frozen in ``manifests/precision<n>.jsonl``.

    First written from the original 501-question dev set (its first whole videos reaching
    ``n`` questions, by video id), so the set stays the same when dev grows.
    """
    frozen = storage.path("manifests", f"precision{n}.jsonl")
    if os.path.exists(frozen):
        with storage.drive_guard():
            return [json.loads(l) for l in open(frozen)]
    rows, by_video = load_manifest("dev"), defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    out = []
    for v, qs in by_video.items():  # manifest order is sorted by video id
        if len(out) >= n:
            break
        out += qs
    with storage.drive_guard(), open(frozen, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


FORMATS = {"images": 280, "video": 70}  # frames-as -> tokens per frame


def out_path(backend, frames_as="images"):
    # images@280 keeps its original file names; other formats get a suffix.
    suffix = "" if frames_as == "images" else f"_{frames_as}{FORMATS[frames_as]}"
    return storage.path(*OUT, f"{backend}{suffix}.jsonl")


def prompt_for(q):
    from openjev.systemone import ChoiceQuestion, render_choice
    return render_choice(None, ChoiceQuestion(type="choice", instructions=q["question"],
                                              criteria=dict(zip(LABELS, q["options"]))))


class TorchScorer:
    def __init__(self, path, frames_as="images"):
        from openjev.scorer import OptionScorer
        self.scorer = OptionScorer(path, backend="torch", dtype="bfloat16", vision_dtype="float16",
                                   batch_size=len(LABELS))
        self.engine = self.scorer._engine
        self.engine.frame_mode, self.engine.frame_soft_tokens = frames_as, FORMATS[frames_as]
        self.scorer.score("warm up", ["a", "b"])

    def features(self, video):
        return self.engine.frame_features(video.frames, frame_batch=2)

    def probs(self, q, video, feats):
        prompt, labels = prompt_for(q)
        res = self.scorer.score(prompt, labels, norm="sum", images=video.frames, image_features=feats,
                                timestamps=video.timestamps)
        return [r.probability for r in res]


class MlxScorer:
    def __init__(self, path, frames_as="images"):
        import mlx.core as mx
        from . import mlx_model as mm
        from .inputs import Gemma4Inputs
        self.mx, self.mm = mx, mm
        self.model, self.enc = mm.load_model(path), Gemma4Inputs(TOKENIZER)
        self.frames_as = frames_as

    def features(self, video):
        e = self.enc.encode("x", video.frames, self.frames_as, FORMATS[self.frames_as], video.timestamps)
        f = self.mm.vision_features(self.model, e.pixel_values, e.position_ids)
        self.mx.eval(f)
        return f

    def probs(self, q, video, feats):
        prompt, _ = prompt_for(q)
        e = self.enc.encode(prompt, video.frames, self.frames_as, FORMATS[self.frames_as], video.timestamps)
        p = self.mx.softmax(self.mm.letter_logits(self.model, e.input_ids, feats, self.frames_as,
                                                  self.enc.label_ids))
        return np.array(p).astype(float).tolist()


def run(args):
    from openjev.video import load_video
    storage.require_root()
    awake = storage.keep_awake()
    qs = questions(args.questions)
    path = out_path(args.backend, args.frames_as)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        with storage.drive_guard():
            done = {(r["video_id"], r["qid"]) for r in map(json.loads, open(path))}
    todo = defaultdict(list)
    for q in qs:
        if (q["video_id"], q["qid"]) not in done:
            todo[q["video_id"]].append(q)
    print(f"{args.backend}: {len(qs)} questions over {len({q['video_id'] for q in qs})} videos, "
          f"{len(done)} done", flush=True)
    if not todo:
        return
    t = time.perf_counter()
    scorer = (TorchScorer if args.backend.startswith("torch") else MlxScorer)(local_model(BACKENDS[args.backend]), args.frames_as)
    print(f"loaded in {time.perf_counter() - t:.0f}s", flush=True)
    n = len(done)
    for vid, vqs in todo.items():
        storage.check_alive()
        t0 = time.perf_counter()
        with storage.drive_guard():
            video = load_video(storage.path(vqs[0]["video"]), n_frames=16)
        t1 = time.perf_counter()
        feats = scorer.features(video)
        t2 = time.perf_counter()
        for q in vqs:
            t3 = time.perf_counter()
            p = scorer.probs(q, video, feats)
            t4 = time.perf_counter()
            row = {"video_id": vid, "qid": q["qid"], "area": q["area"], "answer_id": q["answer_id"],
                   "backend": args.backend, "probs": p, "pred": int(np.argmax(p)),
                   "correct": int(np.argmax(p)) == q["answer_id"], "decode_s": (t1 - t0) / len(vqs),
                   "vision_s": (t2 - t1) / len(vqs), "score_s": t4 - t3,
                   "latency_s": (t2 - t0) / len(vqs) + (t4 - t3)}
            with storage.drive_guard(), open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            print(f"[{n}/{len(qs)}] {vid}/{q['qid']} pred {LABELS[row['pred']]} "
                  f"({'ok' if row['correct'] else 'x'}) {row['latency_s']:.1f}s", flush=True)
    if awake:
        awake.terminate()


def report(args):
    storage.require_root()
    rows = {}
    for b in BACKENDS:
        if os.path.exists(out_path(b, args.frames_as)):
            rows[b] = {(r["video_id"], r["qid"]): r for r in map(json.loads, open(out_path(b, args.frames_as)))}
    ref = rows.get("torch-bf16")
    if not ref:
        sys.exit("no torch-bf16 results yet")
    lines = ["# Precision check: MLX vs torch bf16 (dev, zero-shot)", "",
             f"16 frames as {args.frames_as}, {FORMATS[args.frames_as]} tokens/frame; option-letter probabilities.", "",
             "| backend | questions | accuracy | agree with torch | max abs prob diff | mean abs prob diff "
             "| model latency s/question | video decode s/question |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    res = {}
    for b, rs in rows.items():
        keys = sorted(set(rs) & set(ref))
        p, pr = np.array([rs[k]["probs"] for k in keys]), np.array([ref[k]["probs"] for k in keys])
        agree = float(np.mean(p.argmax(1) == pr.argmax(1)))
        d = np.abs(p - pr)
        top2 = np.sort(pr, 1)[:, -2:]
        decided = (top2[:, 1] - top2[:, 0]) >= DECIDED
        ans = np.array([ref[k]["answer_id"] for k in keys])
        res[b] = {"n": len(keys), "accuracy": float(np.mean([rs[k]["correct"] for k in keys])),
                  "agree": agree, "max_abs_prob_diff": float(d.max()), "mean_abs_prob_diff": float(d.mean()),
                  # Model compute only (vision encoder share + scoring). Video decoding reads the
                  # external HDD, so it is reported separately; model loading is never included.
                  "latency_s": float(np.mean([rs[k]["vision_s"] + rs[k]["score_s"] for k in keys])),
                  "decode_s": float(np.mean([rs[k]["decode_s"] for k in keys])),
"decided_n": int(decided.sum()),
                  "agree_decided": float(np.mean(p.argmax(1)[decided] == pr.argmax(1)[decided])),
                  "p_correct_shift": float(np.mean(p[np.arange(len(keys)), ans] - pr[np.arange(len(keys)), ans])),
                  "disagreements": [f"{k[0]}/{k[1]}" for k in keys if rs[k]["pred"] != ref[k]["pred"]]}
        r = res[b]
        lines.append(f"| {b} | {r['n']} | {r['accuracy']:.1%} | {r['agree']:.1%} | {r['max_abs_prob_diff']:.3f} "
                     f"| {r['mean_abs_prob_diff']:.4f} | {r['latency_s']:.1f} | {r['decode_s']:.1f} |")
    m8 = res.get("mlx-8bit")
    if m8:
        m8["pass"] = m8["agree_decided"] >= AGREE and abs(m8["p_correct_shift"]) <= MAX_SHIFT
        lines += ["", f"Rule: MLX 8-bit is used for dev evaluations if it agrees with torch bf16 on >= {AGREE:.0%} "
                      f"of the questions torch decides by a margin >= {DECIDED} and its mean shift in P(correct) "
                      f"is within +-{MAX_SHIFT}: **{'PASS' if m8['pass'] else 'FAIL'}** "
                      f"({m8['agree_decided']:.1%} of {m8['decided_n']} decided questions, shift "
                      f"{m8['p_correct_shift']:+.4f}; all answers {m8['agree']:.1%}, n={m8['n']})."]
    for b, r in res.items():
        if r["disagreements"] and b != "torch-bf16":
            lines.append(f"\n{b} disagreements: {', '.join(r['disagreements'])}")
    text = "\n".join(lines) + "\n"
    with storage.drive_guard(), open(storage.path(*OUT, "summary.md" if args.frames_as == "images"
                                                 else f"summary_{args.frames_as}{FORMATS[args.frames_as]}.md"), "w") as f:
        f.write(text)
    print(text)
    log_run("pre-2", "eval", {"what": "precision check", "frames": 16, "frames_as": args.frames_as,
                                  "soft_tokens": FORMATS[args.frames_as]},
            {"set": "dev", "questions": res[next(iter(res))]["n"]}, None, res)
    if m8 and not m8["pass"]:
        sys.exit(2)  # gate failed: stop any command chained after this report



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=BACKENDS, required=True)
    r.add_argument("--questions", type=int, default=100)
    r.add_argument("--frames-as", choices=FORMATS, default="video")
    rp = sub.add_parser("report")
    rp.add_argument("--frames-as", choices=FORMATS, default="video")
    args = ap.parse_args(argv)
    try:
        {"run": run, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
