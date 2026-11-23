"""Stage 1 of docs/video-decision-model-plan.md: scores on the dev sets and benchmarks (MLX 8-bit).

    python -m video_train.vdm_eval run --set all --tag stageA_ep1 --adapter <checkpoint dir>   # resumable
    python -m video_train.vdm_eval run --set nextqa_dev --limit 50 --tag smoke --adapter <checkpoint dir>
    python -m video_train.vdm_eval report --tag stageA_ep1  # summary.md + history entry

Qwen3-VL-4B with a trained LoRA only (``--adapter``, required): a LoRA saved by
``adapters.save_adapter``, installed as served (``adapters.install_adapter``, forced on).

Sets: the in-domain dev sets and the never-trained-on benchmarks of ``vdm_data`` (test sets are
held back for the final run). 16 frames, 192 vision tokens per 2 frames, prompts from
``vdm_data.turns``. The vision encoder and the prompt up to the last video token run once per
video (KV cache); each question then runs only its own tokens.

Readouts: multiple choice = float32 option-letter probabilities; CLEVRER multi-answer = P(Yes)
vs P(No) per option; CLEVRER descriptive and Charades-STA = greedy answer (<= 24 tokens).
Metrics: accuracy (+-95% interval); CLEVRER per option and per question (all options right);
Charades-STA mean IoU and recall at IoU 0.3 / 0.5 / 0.7.

Rows go to ``vdm/results/stage1/<tag>/<set>.jsonl``; ``run.json`` there holds the settings (a
rerun under the same tag must use the same ones).
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
from .models import local_model, store_path
from .vdm_data import BENCH, DEV, YES, NO, kind, load_row_video, parse_span, rows, turns

MLX8 = "Qwen3-VL-4B-Instruct-mlx-8bit"
MAX_TOKENS = 24
SETS = {**DEV, **BENCH}
OUT = ("vdm", "results", "stage1")


def out(tag, name):
    p = storage.path(*OUT, tag, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def iou(a, b):
    if a is None:
        return 0.0
    a = sorted(a)
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def single_ids(tok, words):
    ids = [tok.encode(w, add_special_tokens=False) for w in words]
    if any(len(x) != 1 for x in ids):
        raise ValueError(f"{words} must be single tokens")
    return [x[0] for x in ids]


def finish(r, k, res):
    """Result fields from a readout: ``res`` is option probs (mc), P(Yes) per option, or text."""
    if k == "mc":
        return {"probs": res.tolist(), "pred": int(res.argmax()), "correct": int(res.argmax()) == r["answer_id"]}
    if k == "yesno":
        right = [(p > 0.5) == c for p, c in zip(res, r["correct"])]
        return {"p_yes": res, "option_correct": right, "correct": all(right)}
    if k == "word":
        return {"output": res, "correct": res.strip().strip(".").lower() == str(r["answer"]).lower()}
    span = parse_span(res)
    v = iou(span, r["span"])
    return {"output": res, "pred_span": span, "iou": v, "correct": v >= 0.5}


class Qwen:
    """Qwen3-VL-4B: the prompt up to the last video token is cached once per video."""

    name = MLX8

    def __init__(self, tokens, model=None, inp=None):
        from . import qwen3vl as q3
        self.q3 = q3
        self.inp = inp or q3.Inputs(store_path(MLX8), tokens)
        self.yes_no = single_ids(self.inp.tok, (YES, NO))
        self.model = model or q3.load_model(local_model(MLX8))

    def start(self, video, r0):
        self.video = video
        enc0 = self.inp.encode([(turns(r0)[0][0], None)], video, True)
        self.pre = self.q3.prefill(self.model, enc0, self.q3.vision_features(self.model, enc0))

    def score(self, r):
        import mlx.core as mx
        q3, m, k, ts = self.q3, self.model, kind(r), turns(r)

        def probs(text, ids):
            enc = self.inp.encode([(text, None)], self.video, True)
            return np.array(mx.softmax(q3.logits_at(m, q3.hidden_last(m, enc, self.pre), ids))).astype(float)

        if k == "mc":
            return finish(r, k, probs(ts[0][0], self.inp.letter_ids[:len(r["options"])]))
        if k == "yesno":
            return finish(r, k, [float(probs(t, self.yes_no)[0]) for t, _ in ts])
        return finish(r, k, q3.generate(m, self.inp, self.inp.encode([(ts[0][0], None)], self.video, True),
                                        self.pre, MAX_TOKENS))


def run(args):
    storage.require_root()
    tag = args.tag
    cfg = {"model": Qwen.name, "adapter": args.adapter, "frames": args.frames, "tokens": args.tokens}
    cfg_path = out(tag, "run.json")
    if os.path.exists(cfg_path):
        if (prev := json.load(open(cfg_path))) != cfg:
            sys.exit(f"{tag} was run with {prev}, not {cfg}; use another --tag")
    else:
        with storage.drive_guard(), open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)
    awake = storage.keep_awake()
    names = list(SETS) if args.set == "all" else args.set.split(",")
    from .adapters import adapter_scope, install_adapter
    b = Qwen(args.tokens)
    print(f"adapter on {install_adapter(b.model, 'video', args.adapter)} layers", flush=True)
    with adapter_scope("video"):
        for name in names:
            qs = rows(SETS[name])[:args.limit]
            path = out(tag, f"{name}.jsonl")
            done = {(r["video_id"], str(r["qid"])) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
            by = defaultdict(list)
            for r in qs:
                if (r["video_id"], str(r["qid"])) not in done:
                    by[r["video_id"]].append(r)
            print(f"{name}: {len(qs)} questions, {len(done)} done", flush=True)
            n = len(done)
            for vid, vqs in by.items():
                storage.check_alive()
                t0 = time.perf_counter()
                video = load_row_video(vqs[0], args.frames)
                t1 = time.perf_counter()
                b.start(video, vqs[0])
                t2 = time.perf_counter()
                right = 0
                for r in vqs:
                    t3 = time.perf_counter()
                    res = b.score(r)
                    row = {"video_id": vid, "qid": r["qid"], "type": r.get("type") or r.get("area"), **res,
                           "decode_s": (t1 - t0) / len(vqs), "video_s": (t2 - t1) / len(vqs),
                           "question_s": time.perf_counter() - t3}
                    with storage.drive_guard(), open(path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    n += 1
                    right += bool(res["correct"])
                print(f"[{name} {n}/{len(qs)}] {vid} {right}/{len(vqs)} right, "
                      f"{t2 - t1:.1f}s video + {row['question_s']:.2f}s/q", flush=True)
    if awake:
        awake.terminate()


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * np.sqrt(x.mean() * (1 - x.mean()) / len(x)) if len(x) else 0.0


def report(args):
    storage.require_root()
    tag = args.tag
    cfg_path = out(tag, "run.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"no run under {tag}")
    cfg = json.load(open(cfg_path))
    lines = [f"# Stage 1: {cfg['model']} + {cfg['adapter']} ({tag})", "",
             "| set | questions | videos | accuracy | other | s / question |", "|---|---:|---:|---:|---|---:|"]
    res = {}
    for name in SETS:
        p = out(tag, f"{name}.jsonl")
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p)]
        acc = [r["correct"] for r in rs]
        lat = np.mean([r["video_s"] + r["question_s"] for r in rs])
        other = ""
        if name.startswith("clevrer"):
            opt = [c for r in rs if "option_correct" in r for c in r["option_correct"]]
            word = [r["correct"] for r in rs if "output" in r]
            other = f"descriptive {np.mean(word):.1%} ({len(word)}), per option {np.mean(opt):.1%}"
        if name.startswith("charades_sta"):
            v = [r["iou"] for r in rs]
            other = (f"mIoU {np.mean(v):.3f}, R@0.3 {np.mean([x >= .3 for x in v]):.1%}, "
                     f"R@0.7 {np.mean([x >= .7 for x in v]):.1%}, unparsed {sum(r['pred_span'] is None for r in rs)}")
        res[name] = {"n": len(rs), "acc": float(np.mean(acc)), "ci": float(ci(acc)), "s_per_q": float(lat), "other": other}
        lines.append(f"| {name} | {len(rs)} | {len({r['video_id'] for r in rs})} | {np.mean(acc):.1%} ± {ci(acc):.1%} "
                     f"| {other} | {lat:.2f} |")
    lines += ["", "Charades-STA accuracy = recall at IoU 0.5. Seconds per question = model compute "
              "(video encoding + prefix shared over the video's questions); decoding apart."]
    text = "\n".join(lines) + "\n"
    with storage.drive_guard(), open(out(tag, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    log_run("vdm-1", "eval", {"what": "stage 1 scores", **cfg, "tag": tag}, {"sets": list(res)}, None, res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "report"):
        p = sub.add_parser(n)
        p.add_argument("--tag", required=True)
    r = sub.choices["run"]
    r.add_argument("--adapter", required=True, help="LoRA directory (adapters.save_adapter format)")
    r.add_argument("--set", required=True, help=f"all, or comma-separated from: {', '.join(SETS)}")
    r.add_argument("--frames", type=int, default=16)
    r.add_argument("--tokens", type=int, default=192, help="vision tokens per 2 frames")
    r.add_argument("--limit", type=int, default=None, help="first N questions of each set (smoke test)")
    args = ap.parse_args(argv)
    try:
        {"run": run, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
