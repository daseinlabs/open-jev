"""Zero-shot Perception Test multiple-choice eval (validation split) for video input.

Perception Test (google-deepmind/perception_test) annotations and videos are CC BY 4.0.
Each question has 3 options and is labelled with an area (memory, abstraction, physics,
semantics) and a reasoning type (descriptive, explanatory, predictive, counterfactual).

    python -m video_eval.perception_test fetch --questions 300
    python -m video_eval.perception_test run --frames 16
    python -m video_eval.perception_test run --frames 16 --model models/gemma-4-E4B-it
    python -m video_eval.perception_test run --frames 16 --model models/gemma-4-E4B-it --frames-as video
    python -m video_eval.perception_test report runs/video_eval/*.jsonl

``fetch`` samples whole videos with a fixed seed until it has enough questions, then pulls
only those videos out of the 75 GB validation zip with HTTP range requests. ``run`` scores
each question with OptionScorer on the torch backend: N frames, then the question rendered
exactly like a System One ``choice`` question with options A/B/C, scored by the label.

Frame features are computed once per video and reused across that video's questions to keep
the run short; the reported uncached latency still adds the full vision time to every
question (vision + language-model prefill + option scoring), i.e. what a request without any
cache would cost.

Gemma 3 gets the frames as images (896 px, 256 tokens each). Gemma 4 gets them as images too
by default (aspect ratio kept, 280 tokens each), which is the like-for-like comparison. With
``--frames-as video`` it gets its native video input instead: each frame labelled with its
``mm:ss`` timestamp, at 70 tokens per frame. ``--soft-tokens`` overrides the per-frame budget.

Every ``run`` logs itself under ``runs/video_eval/`` without any extra step: per-question rows
(``<stem>.jsonl``), the console (``<stem>.log``), the run settings (``<stem>.meta.json``), one
summary line per invocation (``history.jsonl``), and a regenerated ``report.md`` covering every
run in the folder, all models. The log, history and report are written even if a run stops early.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict

ANNOTATIONS_URL = "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip"
VIDEOS_URL = "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip"
DATA = "data/perception_test"
RUNS = "runs/video_eval"
LABELS = ["A", "B", "C"]
LEGACY_MODEL = "gemma-3-4b-it"  # rows written before the model was recorded
DEFAULT_SOFT_TOKENS = {"images": 280, "video": 70}  # Gemma 4 per-frame budgets


# -------------------------------------------------------------------- fetch
def fetch(args) -> None:
    from .remote_zip import open_remote_zip

    os.makedirs(f"{args.data}/videos", exist_ok=True)
    ann_path = f"{args.data}/mc_question_valid.json"
    if not os.path.exists(ann_path):
        with urllib.request.urlopen(ANNOTATIONS_URL, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("mc_question_valid.json", args.data)
    ann = json.load(open(ann_path))
    vids = sorted(v for v, x in ann.items() if x.get("mc_question"))
    random.Random(args.seed).shuffle(vids)
    rows, chosen = [], []
    for v in vids:
        if len(rows) >= args.questions:
            break
        chosen.append(v)
        for q in ann[v]["mc_question"]:
            rows.append({"video_id": v, "qid": q["id"], "question": q["question"], "options": q["options"],
                         "answer_id": q["answer_id"], "area": q["area"], "reasoning": q["reasoning"],
                         "tags": q.get("tag", []), "duration_s": ann[v]["metadata"]["num_frames"] /
                         ann[v]["metadata"]["frame_rate"]})
    with open(f"{args.data}/subset.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    z = open_remote_zip(VIDEOS_URL)
    total = 0
    for i, v in enumerate(chosen):
        out = f"{args.data}/videos/{v}.mp4"
        if not os.path.exists(out):
            data = z.read(f"videos/{v}.mp4")
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
        total += os.path.getsize(out)
        print(f"[{i + 1}/{len(chosen)}] {v} ({total / 1e6:.0f} MB so far)", flush=True)
    print(f"{len(rows)} questions over {len(chosen)} videos -> {args.data}/subset.jsonl")


# ---------------------------------------------------------------------- run
def model_tag(model: str) -> str:
    return os.path.basename(os.path.normpath(model))


def out_stem(args) -> str:
    tag = model_tag(args.model)
    if tag == LEGACY_MODEL:
        stem = f"perception_test_n{args.frames}"
    else:
        stem = f"perception_test_{tag}_{args.frames_as}_n{args.frames}"
        if args.soft_tokens and args.soft_tokens != DEFAULT_SOFT_TOKENS[args.frames_as]:
            stem += f"_st{args.soft_tokens}"
    return stem + ("_chat" if args.chat else "")


class _Tee:
    """Copies everything written to stdout into the run's log file."""

    def __init__(self, path):
        self.file, self.stdout = open(path, "a"), sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return sha.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args) -> None:
    out_path = args.out or f"{RUNS}/{out_stem(args)}.jsonl"
    folder = os.path.dirname(out_path) or "."
    os.makedirs(folder, exist_ok=True)
    stem = out_path[:-len(".jsonl")] if out_path.endswith(".jsonl") else out_path
    tee = _Tee(stem + ".log")
    sys.stdout = tee
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== {started} run {' '.join(sys.argv[1:])} -> {out_path}")
    status = "error"
    try:
        status = _run(args, out_path, stem)
    except KeyboardInterrupt:
        status = "interrupted"
        print("interrupted: rows so far are saved; rerun the same command to resume")
    finally:
        try:
            _log_history(args, out_path, started, status)
            report(argparse.Namespace(files=sorted(_all_runs(folder)), out=os.path.join(folder, "report.md")))
        finally:
            sys.stdout = tee.stdout
            tee.file.close()


def _run(args, out_path: str, stem: str) -> str:
    import torch
    from openjev.scorer import OptionScorer
    from openjev.systemone import ChoiceQuestion, render_choice
    from openjev.video import load_video

    rows = [json.loads(l) for l in open(f"{args.data}/subset.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(out_path):
        done = {(r["video_id"], r["qid"]) for r in map(json.loads, open(out_path))}
    by_video = defaultdict(list)
    for r in rows:
        if (r["video_id"], r["qid"]) not in done:
            by_video[r["video_id"]].append(r)
    if not by_video:
        print("nothing to do")
        return "complete"

    scorer = OptionScorer(args.model, backend="torch", dtype=args.dtype, vision_dtype=args.vision_dtype,
                          batch_size=len(LABELS))
    engine = scorer._engine
    engine.frame_mode, engine.frame_soft_tokens = args.frames_as, args.soft_tokens
    tag = model_tag(args.model)
    gemma4 = engine.model_type == "gemma4"
    meta = {"model": tag, "model_path": args.model, "model_type": engine.model_type, "frames": args.frames,
            "chat": args.chat, "frames_as": args.frames_as if gemma4 else "images",
            "soft_tokens": engine.soft_tokens_per_frame if gemma4 else None,
            "dtype": args.dtype, "vision_dtype": args.vision_dtype, "frame_batch": args.frame_batch,
            "state": args.state, "limit": args.limit, "device": str(engine.device), "git": _git_sha(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "machine": f"{platform.system()} {platform.machine()}", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(stem + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    scorer.score("warm up", ["a", "b"])
    n_done = len(done)
    with open(out_path, "a") as out:
        for vid, qs in by_video.items():
            t0 = time.perf_counter()
            video = load_video(f"{args.data}/videos/{vid}.mp4", n_frames=args.frames)
            t1 = time.perf_counter()
            feats = engine.frame_features(video.frames, frame_batch=args.frame_batch)
            t2 = time.perf_counter()
            decode_s, vision_s = t1 - t0, t2 - t1
            for q in qs:
                question = ChoiceQuestion(type="choice", instructions=q["question"],
                                          criteria=dict(zip(LABELS, q["options"])))
                prompt, labels = render_choice(args.state or None, question)
                res = scorer.score(prompt, labels, norm="sum", chat=args.chat, images=video.frames,
                                   image_features=feats, timestamps=video.timestamps)
                tm = scorer.last_timing
                probs = [x.probability for x in res]
                pred = max(range(len(probs)), key=probs.__getitem__)
                rec = {**{k: q[k] for k in ("video_id", "qid", "area", "reasoning", "tags", "answer_id", "duration_s")},
                       "model": tag, "n_frames": args.frames, "chat": args.chat, "frames_as": meta["frames_as"],
                       "soft_tokens": meta["soft_tokens"],
                       "probs": probs, "pred": pred,
                       "correct": pred == q["answer_id"], "top_prob": probs[pred],
                       "context_tokens": tm["context_tokens"], "option_tokens": tm["option_tokens"],
                       "decode_s": decode_s, "vision_s": vision_s,
                       # prefill_s includes processor tokenisation and the LM prefill; vision is reused
                       "prefill_s": tm["prefill_s"], "score_s": tm["options_s"],
                       "latency_s": decode_s + vision_s + tm["prefill_s"] + tm["options_s"]}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                print(f"[{n_done}/{len(rows)}] {vid} q{q['qid']} {q['area']:<11} "
                      f"{'ok ' if rec['correct'] else 'bad'} p={rec['top_prob']:.2f} "
                      f"tok={rec['context_tokens']} lat={rec['latency_s']:.1f}s", flush=True)
            del feats
            if engine.device.type == "mps":
                torch.mps.empty_cache()
    return "complete"


def _all_runs(folder: str) -> list[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.startswith("perception_test") and f.endswith(".jsonl")]


def _log_history(args, out_path: str, started: str, status: str) -> None:
    """One line per invocation in history.jsonl, summarising the run's file as it now stands."""
    rs = list(map(json.loads, open(out_path))) if os.path.exists(out_path) else []
    row = {"started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status,
           "model": model_tag(args.model), "frames": args.frames, "chat": args.chat,
           "frames_as": rs[-1].get("frames_as") if rs else None,
           "soft_tokens": rs[-1].get("soft_tokens") if rs else None, "file": out_path, "questions": len(rs),
           "accuracy": round(_acc(rs), 4) if rs else None, "ci95": round(_ci(rs), 4) if rs else None,
           "latency_mean_s": round(_mean([r["latency_s"] for r in rs]), 2) if rs else None,
           "tokens_mean": round(_mean([r["context_tokens"] + r["option_tokens"] for r in rs])) if rs else None,
           "git": _git_sha(), "argv": sys.argv[1:]}
    with open(os.path.join(os.path.dirname(out_path) or ".", "history.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"history: {status}, {row['questions']} questions"
          + (f", accuracy {100 * row['accuracy']:.1f}% ± {100 * row['ci95']:.1f}" if rs else ""))


# ------------------------------------------------------------------- report
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _acc(rs):
    return _mean([r["correct"] for r in rs])


def _ci(rs):
    p, n = _acc(rs), len(rs)
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")


def _key(r):
    return (r.get("model", LEGACY_MODEL), r["n_frames"], r.get("chat", False), r.get("frames_as", "images"),
            r.get("soft_tokens") or 0)


def _name(k, short=False):
    model, frames, chat, frames_as, st = k
    tail = (f" {frames_as}@{st}" if st else "") + (" chat" if chat else "")
    return f"N={frames}{tail}" if short else f"{frames}{tail}"


def report(args) -> str:
    runs = defaultdict(list)
    for path in args.files:
        for r in map(json.loads, open(path)):
            runs[_key(r)].append(r)
    keys = sorted(runs)
    lines = ["# Perception Test MC-VQA (validation subset), zero-shot", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from {len(args.files)} run files. "
             "Gemma 3 sees frames as images (896 px, ~261 tokens each). Gemma 4 rows say how the frames "
             "went in and the tokens per frame: images@280 is the like-for-like setup, video@70 is "
             "Gemma 4's native video input (mm:ss-labelled frames)._", "",
             "| Model | Frames | Questions | Accuracy (95% CI) | Tokens / request | Uncached latency, mean (p50) | "
             "of which vision | Top prob, correct | Top prob, wrong |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        rs = runs[k]
        lat = [r["latency_s"] for r in rs]
        lines.append(
            f"| {k[0]} | {_name(k)} | {len(rs)} | {100 * _acc(rs):.1f}% ± {100 * _ci(rs):.1f} | "
            f"{_mean([r['context_tokens'] + r['option_tokens'] for r in rs]):.0f} | "
            f"{_mean(lat):.1f}s ({statistics.median(lat):.1f}s) | {_mean([r['vision_s'] for r in rs]):.1f}s | "
            f"{_mean([r['top_prob'] for r in rs if r['correct']]):.3f} | "
            f"{_mean([r['top_prob'] for r in rs if not r['correct']]):.3f} |")

    # Same-question comparison on the smallest run (e.g. the 64-frame subset).
    common = set.intersection(*[{(r["video_id"], r["qid"]) for r in runs[k]} for k in keys]) if keys else set()
    if len(keys) > 1 and any(len(runs[k]) != len(common) for k in keys):
        lines += ["", f"Same {len(common)} questions in every run:", "",
                  "| Model | Frames | Accuracy |", "|---|---|---|"]
        for k in keys:
            rs = [r for r in runs[k] if (r["video_id"], r["qid"]) in common]
            lines.append(f"| {k[0]} | {_name(k)} | {100 * _acc(rs):.1f}% ± {100 * _ci(rs):.1f} |")

    for field in ("area", "reasoning"):
        cats = sorted({r[field] for k in keys for r in runs[k]})
        lines += ["", f"Accuracy by {field} (questions in brackets):", "",
                  f"| {field.title()} | " + " | ".join(f"{k[0]} {_name(k, short=True)}" for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for c in cats:
            cells = []
            for k in keys:
                rs = [r for r in runs[k] if r[field] == c]
                cells.append(f"{100 * _acc(rs):.1f}% ({len(rs)})" if rs else "–")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    if getattr(args, "out", None):
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"report: {args.out}")
    else:
        print(text)
    return text


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--questions", type=int, default=300)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--data", default=DATA)
    r = sub.add_parser("run")
    r.add_argument("--frames", type=int, default=16)
    r.add_argument("--limit", type=int, default=0, help="only the first N questions of the subset")
    r.add_argument("--model", default="models/gemma-3-4b-it", help="e.g. models/gemma-4-E4B-it")
    r.add_argument("--frames-as", choices=("images", "video"), default="images",
                   help="Gemma 4 only: frames as separate images (like Gemma 3) or as native video input")
    r.add_argument("--soft-tokens", type=int, default=None,
                   help="Gemma 4 only: soft tokens per frame (70, 140, 280, 560 or 1120; "
                        "default 280 for images, 70 for video)")
    r.add_argument("--dtype", default="bfloat16", help="language model dtype (float16 overflows to NaN on MPS)")
    r.add_argument("--vision-dtype", default="float16")
    r.add_argument("--frame-batch", type=int, default=2)
    r.add_argument("--chat", action="store_true", help="wrap the prompt in Gemma's chat template")
    r.add_argument("--state", default="", help="optional System One state text")
    r.add_argument("--data", default=DATA)
    r.add_argument("--out")
    rp = sub.add_parser("report")
    rp.add_argument("files", nargs="+")
    rp.add_argument("--out", help="write the report here instead of printing it")
    args = p.parse_args(argv)
    {"fetch": fetch, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
