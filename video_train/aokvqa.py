"""Image forgetting check: a fixed set of A-OKVQA validation multiple-choice questions.

    python -m video_train.aokvqa fetch                    # parquet -> drive, fixed 500-question set
    python -m video_train.aokvqa run --tag base           # score with a model, resumable
    python -m video_train.aokvqa run --tag s7_B_lr1e-04 --adapter <checkpoint dir>   # with a LoRA adapter
    python -m video_train.aokvqa report --before base --after <tag>   # paired comparison

A-OKVQA (allenai/aokvqa, Apache-2.0; COCO 2017 images): 4 options per question. Each image is
one Gemma 4 image (280 tokens), followed by the same OpenJev choice prompt as the video evals;
the readout is the option-letter probabilities (A-D). The question set is fixed by seed, so
before/after comparisons are paired.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import urllib.request

import numpy as np

from . import storage
from .history import log_run
from .inputs import LETTERS
from .models import CONFIG, local_model

URL = ("https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA/resolve/d1b0efa3a436e9101dfbde3752db7607da696c35/"
       "data/validation-00000-of-00001-b2bd0de231b6326a.parquet")
SHA256 = "419b38d94b1529b00d4aa0da0e945b946d8f028514183a267f87e25297ea404d"


def parquet_path():
    return storage.path("aokvqa", "validation.parquet")


def manifest_path(n):
    return storage.path("manifests", f"aokvqa_val{n}.jsonl")


def result_path(tag, n):
    return storage.path("results", "aokvqa", f"val{n}_{tag}.jsonl")


def fetch(args):
    storage.require_root()
    p = parquet_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with storage.drive_guard(), urllib.request.urlopen(URL, timeout=120) as r, open(p + ".part", "wb") as f:
            f.write(r.read())
        os.replace(p + ".part", p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if sha != SHA256:
        raise SystemExit(f"checksum mismatch for {p}: {sha}")
    import pyarrow.parquet as pq
    t = pq.read_table(p, columns=["question_id", "question", "choices", "correct_choice_idx"]).to_pylist()
    rows = sorted(t, key=lambda r: r["question_id"])
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.questions]
    with storage.drive_guard(), open(manifest_path(args.questions), "w") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "question": r["question"], "options": r["choices"],
                                "answer_id": r["correct_choice_idx"]}) + "\n")
    n_opts = {len(r["choices"]) for r in rows}
    print(f"{len(t)} validation questions; {len(rows)} selected (seed {args.seed}), options per question {n_opts}")
    log_run("forgetting", "data", {"what": "A-OKVQA val subset", "seed": args.seed, "questions": len(rows)},
            {"parquet_sha256": sha, "source": URL}, args.seed, {"selected": len(rows)})


def _images():
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(parquet_path(), columns=["question_id", "image"]).to_pylist()
    return {r["question_id"]: r["image"] for r in t}, Image


def run(args):
    import mlx.core as mx
    from openjev.systemone import ChoiceQuestion, render_choice
    from . import mlx_model as mm
    from .inputs import Gemma4Inputs
    storage.require_root()
    awake = storage.keep_awake()
    with storage.drive_guard():
        qs = [json.loads(l) for l in open(manifest_path(args.questions))]
    out = result_path(args.tag, args.questions)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["question_id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [q for q in qs if q["question_id"] not in done]
    print(f"{args.tag}: {len(qs)} questions, {len(done)} done", flush=True)
    if not todo:
        return
    with storage.drive_guard():
        images, Image = _images()
    enc = Gemma4Inputs(CONFIG)
    model = mm.load_model(local_model(args.model))
    scope = contextlib.nullcontext  # a factory: a @contextmanager scope can be entered only once
    if args.adapter:  # forced on: in serving, image requests never use the video adapter
        from .adapters import adapter_scope, install_adapter
        install_adapter(model, "video", args.adapter)
        scope = lambda: adapter_scope("video")
    for i, q in enumerate(todo):
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(images[q["question_id"]]["bytes"])).convert("RGB")
        prompt, _ = render_choice(None, ChoiceQuestion(type="choice", instructions=q["question"],
                                                       criteria=dict(zip(LETTERS, q["options"]))))
        e = enc.encode(prompt, [img], "images", args.soft_tokens)
        t1 = time.perf_counter()
        with scope():
            feats = mm.vision_features(model, e.pixel_values, e.position_ids)
            p = mx.softmax(mm.letter_logits(model, e.input_ids, feats, "images", enc.letter_ids(len(q["options"]))))
            p = np.array(p).astype(float)
        t2 = time.perf_counter()
        row = {"question_id": q["question_id"], "tag": args.tag, "model": args.model, "probs": p.tolist(),
               "adapter": args.adapter,
               "pred": int(p.argmax()), "answer_id": q["answer_id"], "correct": int(p.argmax()) == q["answer_id"],
               "decode_s": t1 - t0, "latency_s": t2 - t1}
        with storage.drive_guard(), open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"[{len(done) + i + 1}/{len(qs)}] model latency {row['latency_s']:.2f}s", flush=True)
    if awake:
        awake.terminate()
    summarize(args.tag, args.questions, args.model)


def _load(tag, n):
    with storage.drive_guard():
        return {r["question_id"]: r for r in map(json.loads, open(result_path(tag, n)))}


def _ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def stats(rows):
    rs = list(rows.values())
    acc = float(np.mean([r["correct"] for r in rs]))
    nll = float(np.mean([-math.log(max(r["probs"][r["answer_id"]], 1e-12)) for r in rs]))
    return {"n": len(rs), "accuracy": acc, "nll": nll,
            "ece": _ece([max(r["probs"]) for r in rs], [r["correct"] for r in rs]),
            "latency_s": float(np.mean([r["latency_s"] for r in rs]))}


def summarize(tag, n, model):
    s = stats(_load(tag, n))
    print(json.dumps(s))
    log_run("forgetting", "eval", {"what": "A-OKVQA", "tag": tag, "model": model, "soft_tokens": 280},
            {"manifest": f"aokvqa_val{n}"}, None, s)


def paired(before, after):
    """Accuracy change on the same questions with an exact McNemar test on the flips."""
    keys = sorted(set(before) & set(after))
    b = np.array([before[k]["correct"] for k in keys])
    a = np.array([after[k]["correct"] for k in keys])
    lost, gained = int(np.sum(b & ~a)), int(np.sum(~b & a))
    from scipy.stats import binomtest
    p = binomtest(gained, lost + gained, 0.5).pvalue if lost + gained else 1.0
    dp = np.mean([after[k]["probs"][after[k]["answer_id"]] - before[k]["probs"][before[k]["answer_id"]]
                  for k in keys])
    return {"n": len(keys), "acc_before": float(b.mean()), "acc_after": float(a.mean()),
            "lost": lost, "gained": gained, "mcnemar_p": float(p), "mean_change_p_correct": float(dp)}


def report(args):
    storage.require_root()
    r = paired(_load(args.before, args.questions), _load(args.after, args.questions))
    text = (f"# A-OKVQA forgetting check: {args.before} -> {args.after}\n\n"
            f"{r['n']} paired questions. Accuracy {r['acc_before']:.1%} -> {r['acc_after']:.1%}; "
            f"{r['lost']} lost, {r['gained']} gained (exact McNemar p = {r['mcnemar_p']:.3f}); "
            f"mean change in p(correct) {r['mean_change_p_correct']:+.3f}.\n")
    print(text)
    with storage.drive_guard(), open(storage.path("results", "aokvqa", f"{args.before}_vs_{args.after}.md"), "w") as f:
        f.write(text)
    log_run("forgetting", "eval", {"what": "A-OKVQA paired", "before": args.before, "after": args.after},
            {"manifest": f"aokvqa_val{args.questions}"}, None, r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--questions", type=int, default=500)
    f.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True, help="name of this configuration, e.g. base")
    r.add_argument("--model", default="gemma-4-E4B-it-mlx-8bit")
    r.add_argument("--questions", type=int, default=500)
    r.add_argument("--soft-tokens", type=int, default=280)
    r.add_argument("--adapter", default=None, help="LoRA adapter dir (saved by save_adapter), applied to every question")
    p = sub.add_parser("report")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--questions", type=int, default=500)
    args = ap.parse_args(argv)
    try:
        {"fetch": fetch, "run": run, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
