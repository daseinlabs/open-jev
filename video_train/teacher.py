"""Step 4 teacher gate: d3-mini (vllm-sr/d3-mini, Apache-2.0) on Perception Test, torch on MPS.

    python -m video_train.teacher run --set dev                       # the gate (2,003 questions)
    python -m video_train.teacher run --set train --sample-videos 100 # contamination check
    python -m video_train.teacher run --set dev --limit-videos 3      # smoke test
    python -m video_train.teacher report

d3-mini is used as published: its own video reading (2 fps, at most 32 frames, 0.2 MP each),
prompt format and 255-way answer readout, through ``model.runtime.prepare / run``. Each question is
a Choice question: the question as instructions, the options as criteria keys in manifest order
(two dev questions repeat an option; the repeat gets a " (2)" suffix so the keys stay unique).
The state is empty.

Rows go to ``results/evals/<set>/<tag>.jsonl`` in the same format as ``video_train.evaluate``, so
its ``metrics`` and ``agree`` apply. ``probs`` follow the manifest's option order. Latency is
``runtime.run`` (model compute); ``decode_s`` is ``runtime.prepare`` (video decoding from the HDD
and preprocessing). Resumable. ``--set train`` with the full split is step 5 (teacher labels);
``--sample-videos`` takes a fixed random sample of videos (seed 0) that later runs extend.

Gate (``report``), against the student's best (video@140, ``--student``):
- accuracy on dev clearly above the student (paired), and NLL / ECE no worse, since its
  probabilities are the training target;
- train-sample accuracy vs dev accuracy: a much higher train score means d3-mini was trained on
  Perception Test train, so its train labels mostly repeat the answer key;
- time to label the 7,392 train questions (step 5).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

from . import storage
from .evaluate import load_results, metrics, result_path
from .history import log_run
from .models import local_model

MODEL = "d3-mini"
SETS = {"dev": "dev", "train": "train"}
TRAIN_QUESTIONS = 7392


def load_set(name):
    with storage.drive_guard():
        return [json.loads(line) for line in open(storage.path("manifests", f"{SETS[name]}.jsonl"))]


def criteria(options):
    """Options as Choice criteria keys, in order; a repeated option gets a suffix to stay unique."""
    keys = []
    for o in options:
        k, n = o, 2
        while k in keys:
            k, n = f"{o} ({n})", n + 1
        keys.append(k)
    return keys


def load_model(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(local_model(MODEL), trust_remote_code=True, device=device,
                                     local_files_only=True)


def run(args):
    storage.require_root()
    awake = storage.keep_awake()
    rows = load_set(args.set)
    out = result_path(args.set, args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    config = {"set": args.set, "tag": args.tag, "model": MODEL, "device": args.device}
    done = set(load_results(args.set, args.tag)) if os.path.exists(out) else set()
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    vids = list(by_video)
    if args.sample_videos:
        vids = sorted(random.Random(0).sample(sorted(vids), min(args.sample_videos, len(vids))))
    vids = [v for v in vids if any((v, q["qid"]) not in done for q in by_video[v])]
    if args.limit_videos:
        vids = vids[:args.limit_videos]
    todo = sum(len(by_video[v]) for v in vids)
    print(f"{args.tag} on {args.set}: {len(done)} done, {todo} to score over {len(vids)} videos", flush=True)
    if not vids:
        return
    model = load_model(args.device)
    rt = model.runtime
    n_done, t_start = 0, time.time()
    for vi, vid in enumerate(vids):
        storage.check_alive()
        qs = [q for q in by_video[vid] if (vid, q["qid"]) not in done]
        keys = {str(q["qid"]): criteria(q["options"]) for q in qs}
        questions = {str(q["qid"]): {"type": "choice", "instructions": q["question"],
                                     "criteria": {k: None for k in keys[str(q["qid"])]}} for q in qs}
        t0 = time.perf_counter()
        with storage.drive_guard():
            prepared = rt.prepare(None, questions, None, [storage.path(qs[0]["video"])])
        t1 = time.perf_counter()
        probabilities, tokens = rt.run(prepared)
        t2 = time.perf_counter()
        answers = rt.respond(prepared, probabilities, tokens)["answers"]
        lines = []
        for q in qs:
            a = answers[str(q["qid"])]
            if "error" in a:
                raise RuntimeError(f"{vid}/{q['qid']}: {a}")
            pr = np.array([a["probabilities"][k] for k in keys[str(q["qid"])]], float)
            lines.append(json.dumps({
                "video_id": vid, "qid": q["qid"], "area": q["area"], "reasoning": q.get("reasoning"),
                "answer_id": q["answer_id"], "probs": pr.tolist(), "pred": int(pr.argmax()),
                "correct": int(pr.argmax()) == q["answer_id"], **config,
                "n_video_questions": len(qs), "decode_s": (t1 - t0) / len(qs),
                "latency_s": (t2 - t1) / len(qs), "video_model_s": t2 - t1}))
        with storage.drive_guard(), open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
        n_done += len(qs)
        if (vi + 1) % 5 == 0 or vi + 1 == len(vids):
            el = time.time() - t_start
            print(f"[{len(done) + n_done}/{len(done) + todo}] {(t2 - t1) / len(qs):.1f}s/question model, "
                  f"eta {el / n_done * (todo - n_done) / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    s = metrics(list(load_results(args.set, args.tag).values()))
    log_run(args.step, "eval", config, {"manifest": SETS[args.set], "sample_videos": args.sample_videos}, None,
            {k: v for k, v in s.items() if not isinstance(v, dict)})
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}))


def paired(a, b):
    """b against a on shared questions: (gained, lost, two-sided McNemar p, normal approximation)."""
    keys = set(a) & set(b)
    w = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    lost = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    p = math.erfc(abs(w - lost) / math.sqrt(w + lost) / math.sqrt(2)) if w + lost else 1.0
    return w, lost, p


def report(args):
    storage.require_root()
    teacher = load_results("dev", args.tag)
    student = load_results("dev", args.student)
    keys = sorted(set(teacher) & set(student))
    mt, ms = metrics([teacher[k] for k in keys]), metrics([student[k] for k in keys])
    w, lost, p = paired(student, teacher)
    lines = [f"# step 4: teacher gate ({args.tag} vs {args.student})", "",
             f"Perception Test dev, {len(keys)} shared questions.", "",
             "| | accuracy | NLL | Brier | ECE | conf ≥ 0.8: coverage / accuracy | latency/question |",
             "|---|---|---|---|---|---|---|"]
    for name, m in ((args.student, ms), (args.tag, mt)):
        hi = f"{m['accuracy_conf_ge_0.8']:.1%}" if m["accuracy_conf_ge_0.8"] is not None else "-"
        lines.append(f"| {name} | {m['accuracy']:.1%} ± {m['ci95']:.1%} | {m['nll']:.3f} | {m['brier']:.3f} | "
                     f"{m['ece']:.3f} | {m['coverage_conf_ge_0.8']:.0%} / {hi} | {m['latency_s']:.2f} s |")
    lines += ["", f"Paired: {args.tag} gains {w}, loses {lost} (net {w - lost:+d}, p={p:.2g}).", "",
              "| area | n | " + args.student + " | " + args.tag + " |", "|---|---|---|---|"]
    for area, v in mt["by_area"].items():
        lines.append(f"| {area} | {v['n']} | {ms['by_area'][area]['accuracy']:.1%} | {v['accuracy']:.1%} |")
    res = {"dev": {k: v for k, v in mt.items() if not isinstance(v, dict)}, "paired": [w, lost, p]}
    if os.path.exists(result_path("train", args.tag)):
        tr = metrics(list(load_results("train", args.tag).values()))
        gap = tr["accuracy"] - mt["accuracy"]
        lines += ["", f"Train sample: {tr['accuracy']:.1%} ± {tr['ci95']:.1%} (n={tr['n']}) vs dev "
                  f"{mt['accuracy']:.1%}: gap {gap * 100:+.1f} points; train NLL {tr['nll']:.3f}, "
                  f"mean confidence {tr['mean_confidence']:.2f} (dev {mt['mean_confidence']:.2f})."]
        res["train"] = {k: v for k, v in tr.items() if not isinstance(v, dict)}
    per_q = mt["latency_s"] + mt["decode_s"]
    lines += ["", f"Labeling all {TRAIN_QUESTIONS:,} train questions at this speed: "
              f"~{TRAIN_QUESTIONS * per_q / 3600:.1f} h (model {mt['latency_s']:.2f} s + decode "
              f"{mt['decode_s']:.2f} s per question)."]
    text = "\n".join(lines)
    d = storage.path("results", args.step)
    os.makedirs(d, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(d, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    log_run(args.step, "eval", {"what": "teacher_gate", "tag": args.tag, "student": args.student}, {}, None, res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=SETS, required=True)
    r.add_argument("--tag", default="d3mini")
    r.add_argument("--device", default="mps")
    r.add_argument("--sample-videos", type=int, default=0, help="fixed random sample of videos (seed 0)")
    r.add_argument("--limit-videos", type=int, default=0, help="stop after this many videos (smoke test)")
    r.add_argument("--step", default="step4")
    p = sub.add_parser("report")
    p.add_argument("--tag", default="d3mini")
    p.add_argument("--student", default="video140")
    p.add_argument("--step", default="step4")
    args = ap.parse_args(argv)
    try:
        {"run": run, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
