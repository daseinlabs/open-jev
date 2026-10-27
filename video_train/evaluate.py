"""Video evals on MLX: Perception Test dev and NExT-QA validation (fixed subset).

    python -m video_train.evaluate run --set dev --tag base_video70
    python -m video_train.evaluate run --set nextqa --tag base_video70
    python -m video_train.evaluate agree --set precision --a <tag> --b <tag>
    python -m video_train.evaluate report --step step2 --tag base

Sets: ``dev`` (Perception Test dev, 501 q), ``nextqa`` (NExT-QA val, fixed 1,003-question
subset), ``precision`` (the 102 dev questions of the precision check).

Default format (the training format): 16 frames in Gemma 4's video layout, 70 tokens per frame,
``mm:ss`` timestamps. ``--frames-as images --soft-tokens 280`` gives the earlier image layout.

Scoring modes:
- ``encode_once`` (default): each video's frames and prompt prefix are encoded once and all of
  its questions are scored as one batch on that cache (``mlx_model.letter_logits_grouped``).
- ``per_question``: one full pass per question (the precision check's method).

Frame selection (step 3, ``video_train.frame_select``): ``--select change`` or ``--select question``
decode ``--candidates`` (default 32) uniform frames and keep ``--frames`` of them. Selected frames
keep their own timestamps. With ``question`` each question has its own frames; questions are
grouped by identical selections, and encode_once applies within a group. Selection time (change
scores or SigLIP 2) counts as model latency; decoding the candidates counts as decode time.

    python -m video_train.evaluate run --set dev --tag change_video70 --select change --step step3
    python -m video_train.evaluate run --set dev --tag question_video70 --select question --step step3

Latency is model compute only (see the latency rule): vision encoder + prefix + questions,
divided over the video's questions. Video decoding (HDD) is reported separately.
Rows: ``results/evals/<set>/<tag>.jsonl`` (resumable). Logged to history.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

from . import frame_select, storage
from .history import log_run
from .inputs import LETTERS
from .models import CONFIG, local_model

SETS = {"dev": "dev", "nextqa": "nextqa_val1000", "precision": "dev"}
PRECISION_QUESTIONS = 102


def load_set(name):
    with storage.drive_guard():
        rows = [json.loads(l) for l in open(storage.path("manifests", f"{SETS[name]}.jsonl"))]
    if name == "precision":
        from .precision_check import questions
        keep = {(q["video_id"], q["qid"]) for q in questions(PRECISION_QUESTIONS)}
        rows = [r for r in rows if (r["video_id"], r["qid"]) in keep]
    return rows


def prompt_of(q):
    """The question as OpenJev renders a Choice question, options lettered in manifest order."""
    from openjev.systemone import ChoiceQuestion, render_choice
    return render_choice(None, ChoiceQuestion(type="choice", instructions=q["question"],
                                              criteria=dict(zip(LETTERS, q["options"]))))[0]


def result_path(set_name, tag):
    return storage.path("results", "evals", set_name, f"{tag}.jsonl")


def load_results(set_name, tag):
    with storage.drive_guard():
        return {(r["video_id"], r["qid"]): r for r in map(json.loads, open(result_path(set_name, tag)))}


def run(args):
    import mlx.core as mx
    from openjev.video import load_video
    from . import mlx_model as mm
    from .inputs import Gemma4Inputs

    storage.require_root()
    awake = storage.keep_awake()
    rows = load_set(args.set)
    out = result_path(args.set, args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    config = {"set": args.set, "tag": args.tag, "model": args.model, "mode": args.mode, "frames": args.frames,
              "frames_as": args.frames_as, "soft_tokens": args.soft_tokens}
    if args.select != "uniform":  # only then, so rows written before selection existed still resume
        config.update(select=args.select, candidates=args.candidates)
    done = set()
    if os.path.exists(out):
        prev = load_results(args.set, args.tag)
        clash = {k: (r[k], v) for r in list(prev.values())[:1] for k, v in config.items() if r.get(k) != v}
        if clash:
            raise SystemExit(f"{out} was written with different settings {clash}; use another --tag")
        done = set(prev)
    by_video = defaultdict(list)
    for r in rows:
        if (r["video_id"], r["qid"]) not in done:
            by_video[r["video_id"]].append(r)
    vids = list(by_video)
    if args.limit_videos:
        vids = vids[:args.limit_videos]
    print(f"{args.tag} on {args.set}: {len(rows)} questions, {len(done)} done, "
          f"{sum(len(by_video[v]) for v in vids)} to score over {len(vids)} videos", flush=True)
    if not vids:
        return
    enc = Gemma4Inputs(CONFIG)
    model = mm.load_model(local_model(args.model))
    warm = enc.encode("warm up", [__import__("PIL.Image").Image.new("RGB", (64, 64))], "images", 70)
    mx.eval(mm.letter_logits(model, warm.input_ids, mm.vision_features(model, warm.pixel_values, warm.position_ids),
                             "images", enc.letter_ids(3)))
    selector = None
    if args.select == "question":
        from .frame_select import SIGLIP, QuestionScorer
        selector = QuestionScorer(local_model(SIGLIP))

    def score(qs, frames, timestamps):
        """Questions that share these frames: (probs, input-encoding s, vision s, tokens).

        Prompt/pixel encoding is preprocessing: like video decoding it is kept out of latency.
        """
        t0 = time.perf_counter()
        encs = [enc.encode(prompt_of(q), frames, args.frames_as, args.soft_tokens, timestamps) for q in qs]
        t1 = time.perf_counter()
        feats = mm.vision_features(model, encs[0].pixel_values, encs[0].position_ids)
        mx.eval(feats)
        t2 = time.perf_counter()
        n_opts = {len(q["options"]) for q in qs}
        if len(n_opts) != 1:
            raise ValueError(f"{qs[0]['video_id']}: questions with different option counts")
        labels = enc.letter_ids(n_opts.pop())
        if args.mode == "encode_once":
            p = encs[0].prefix_len
            prefix = encs[0].input_ids[:p]
            if any(e.input_ids[:e.prefix_len] != prefix for e in encs):
                raise ValueError(f"{qs[0]['video_id']}: questions do not share the video prefix")
            logits = mm.letter_logits_grouped(model, prefix, feats, args.frames_as,
                                              [e.input_ids[p:] for e in encs], labels)
            probs = np.array(mx.softmax(logits, axis=-1)).astype(float)
        else:
            probs = np.stack([np.array(mx.softmax(mm.letter_logits(model, e.input_ids, feats, args.frames_as,
                                                                   labels))).astype(float) for e in encs])
        return probs, t1 - t0, t2 - t1, len(encs[0].input_ids)

    n_done, t_start = len(done), time.time()
    for vi, vid in enumerate(vids):
        storage.check_alive()
        qs = by_video[vid]
        t0 = time.perf_counter()
        n_decode = args.frames if args.select == "uniform" else args.candidates
        with storage.drive_guard():
            video = load_video(storage.path(qs[0]["video"]), n_frames=n_decode)
        t1 = time.perf_counter()
        # Which frames each question sees: all of them (uniform), one choice per video (change),
        # or one per question (question).
        if args.select == "uniform":
            picks = [tuple(range(len(video.frames)))] * len(qs)
        elif args.select == "change":
            picks = [tuple(frame_select.select_change(video.frames, args.frames))] * len(qs)
        else:
            img = selector.image_embeddings(video.frames)
            picks = [tuple(frame_select.select_question(
                img, selector.text_embeddings(frame_select.question_texts(q["question"], q["options"])),
                args.frames)) for q in qs]
        select_s = time.perf_counter() - t1
        groups = defaultdict(list)
        for i, pk in enumerate(picks):
            groups[pk].append(i)
        probs, enc_s, vision_s, tokens = [None] * len(qs), 0.0, 0.0, {}
        for pk, idx in groups.items():
            pr, es, vs, n_tok = score([qs[i] for i in idx], [video.frames[j] for j in pk],
                                  [video.timestamps[j] for j in pk])
            enc_s += es
            vision_s += vs
            for i, p in zip(idx, pr):
                probs[i], tokens[i] = p, n_tok
        t3 = time.perf_counter()
        model_s = t3 - t1 - enc_s
        with storage.drive_guard(), open(out, "a") as f:
            for i, (q, pr) in enumerate(zip(qs, probs)):
                row = {
                    "video_id": vid, "qid": q["qid"], "area": q["area"], "reasoning": q.get("reasoning", q.get("type")),
                    "answer_id": q["answer_id"], "probs": pr.tolist(), "pred": int(pr.argmax()),
                    "correct": int(pr.argmax()) == q["answer_id"], **config,
                    "n_video_questions": len(qs), "decode_s": (t1 - t0 + enc_s) / len(qs),
                    "latency_s": model_s / len(qs), "video_model_s": model_s,
                    "vision_s": vision_s, "tokens": tokens[i]}
                if args.select != "uniform":
                    row.update(select_s=select_s, frame_groups=len(groups), frame_idx=list(picks[i]),
                               frame_times=[round(video.timestamps[j], 3) for j in picks[i]])
                f.write(json.dumps(row) + "\n")
        n_done += len(qs)
        if (vi + 1) % 5 == 0 or vi + 1 == len(vids):
            el = time.time() - t_start
            left = sum(len(by_video[v]) for v in vids[vi + 1:])
            print(f"[{n_done}/{len(rows)}] {model_s / len(qs):.1f}s/question model, "
                  f"eta {el / (n_done - len(done)) * left / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    s = metrics(list(load_results(args.set, args.tag).values()))
    log_run(args.step, "eval", config, {"manifest": SETS[args.set]}, None, s)
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}))


# ------------------------------------------------------------------ metrics
def _ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(rs):
    if not rs:
        return {}
    correct = np.array([r["correct"] for r in rs], float)
    conf = np.array([max(r["probs"]) for r in rs])
    p_true = np.array([r["probs"][r["answer_id"]] for r in rs])
    brier = np.mean([sum((p - (i == r["answer_id"])) ** 2 for i, p in enumerate(r["probs"])) for r in rs])
    hi = conf >= 0.8
    out = {"n": len(rs), "accuracy": float(correct.mean()),
           "ci95": float(1.96 * math.sqrt(correct.mean() * (1 - correct.mean()) / len(rs))),
           "nll": float(-np.mean(np.log(np.maximum(p_true, 1e-12)))), "brier": float(brier),
           "ece": _ece(conf, correct), "mean_confidence": float(conf.mean()),
           "coverage_conf_ge_0.8": float(hi.mean()),
           "accuracy_conf_ge_0.8": float(correct[hi].mean()) if hi.any() else None,
           "latency_s": float(np.mean([r["latency_s"] for r in rs])),
           "decode_s": float(np.mean([r["decode_s"] for r in rs]))}
    for key in ("area", "reasoning"):
        groups = defaultdict(list)
        for r in rs:
            groups[r[key]].append(r["correct"])
        out[f"by_{key}"] = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(groups.items())}
    return out


def agree(args):
    a, b = load_results(args.set, args.a), load_results(args.set, args.b)
    keys = sorted(set(a) & set(b))
    pa, pb = np.array([a[k]["probs"] for k in keys]), np.array([b[k]["probs"] for k in keys])
    same = pa.argmax(1) == pb.argmax(1)
    d = np.abs(pa - pb)
    res = {"n": len(keys), "agree": float(same.mean()), "max_abs_prob_diff": float(d.max()),
           "mean_abs_prob_diff": float(d.mean()), "acc_a": float(np.mean([a[k]["correct"] for k in keys])),
           "acc_b": float(np.mean([b[k]["correct"] for k in keys])),
           "latency_a": float(np.mean([a[k]["latency_s"] for k in keys])),
           "latency_b": float(np.mean([b[k]["latency_s"] for k in keys])),
           "disagreements": [f"{k[0]}/{k[1]}" for k, s in zip(keys, same) if not s]}
    print(json.dumps(res, indent=1))
    log_run(args.step, "eval", {"what": "agreement", "set": args.set, "a": args.a, "b": args.b}, {}, None, res)
    if res["agree"] < args.min_agree:
        sys.exit(2)  # stop any command chained after this check
    return res


def _table(name, m):
    lines = [f"### {name}", "",
             f"Accuracy **{m['accuracy']:.1%}** ± {m['ci95']:.1%} (n={m['n']}); NLL {m['nll']:.3f}; "
             f"Brier {m['brier']:.3f}; ECE {m['ece']:.3f}; mean confidence {m['mean_confidence']:.2f}.",
             f"Answers with confidence ≥ 0.8: {m['coverage_conf_ge_0.8']:.0%} of questions, "
             + (f"{m['accuracy_conf_ge_0.8']:.1%} correct." if m['accuracy_conf_ge_0.8'] is not None else "none."),
             f"Model latency {m['latency_s']:.2f} s/question (video decode {m['decode_s']:.2f} s, not included).", ""]
    for key in ("area", "reasoning"):
        lines += [f"| {key} | n | accuracy |", "|---|---:|---:|"]
        lines += [f"| {k} | {v['n']} | {v['accuracy']:.1%} |" for k, v in m[f"by_{key}"].items()]
        lines.append("")
    return lines


def report(args):
    storage.require_root()
    lines = [f"# {args.step}: {args.tag}", ""]
    res = {}
    for s in args.sets.split(","):
        rs = list(load_results(s, args.tag).values())
        m = metrics(rs)
        res[s] = m
        cfg = {k: rs[0][k] for k in ("model", "mode", "frames", "frames_as", "soft_tokens")}
        lines += _table(f"{'Perception Test dev' if s == 'dev' else 'NExT-QA val (1,003-question subset)'} "
                        f"— {json.dumps(cfg)}", m)
    text = "\n".join(lines)
    d = storage.path("results", args.step)
    os.makedirs(d, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(d, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    log_run(args.step, "eval", {"what": "report", "tag": args.tag, "sets": args.sets}, {}, None,
            {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in res.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=SETS, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--model", default="gemma-4-E4B-it-mlx-8bit")
    r.add_argument("--mode", choices=("encode_once", "per_question"), default="encode_once")
    r.add_argument("--frames", type=int, default=16)
    # Default = the training format: Gemma 4's video layout, 16 frames x 70 tokens, mm:ss timestamps.
    r.add_argument("--frames-as", choices=("images", "video"), default="video")
    r.add_argument("--soft-tokens", type=int, default=70)
    r.add_argument("--select", choices=frame_select.SELECTORS, default="uniform",
                   help="frame selection (step 3); uniform = the baseline")
    r.add_argument("--candidates", type=int, default=32, help="frames decoded before selection")
    r.add_argument("--limit-videos", type=int, default=0, help="timing sample")
    r.add_argument("--step", default="step2")
    a = sub.add_parser("agree")
    a.add_argument("--set", choices=SETS, required=True)
    a.add_argument("--a", required=True)
    a.add_argument("--b", required=True)
    a.add_argument("--step", default="step2")
    a.add_argument("--min-agree", type=float, default=0.98, help="exit with status 2 below this")
    p = sub.add_parser("report")
    p.add_argument("--step", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--sets", default="dev,nextqa")
    args = ap.parse_args(argv)
    try:
        {"run": run, "agree": agree, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
