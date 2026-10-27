"""Check that per-video grouping gives the same loss and LoRA gradients as per-question passes.

    python -m video_train.check_grouping --model gemma-4-E4B-it-mlx-8bit --videos 3

For each video: the mean letter cross-entropy over its questions and its gradient, computed
(a) one full sequence per question and (b) the prefix once plus all questions as a batch.
LoRA B is set to small random values first, so gradients reach every adapter matrix.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from openjev.systemone import ChoiceQuestion, render_choice
from openjev.video import load_video

from .inputs import LABELS, Gemma4Inputs
from .models import CONFIG, local_model
from .mlx_model import attach_lora, letter_ce, letter_logits, letter_logits_grouped, load_model, vision_features


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E4B-it-mlx-8bit", help="model name (staged from the drive)")
    ap.add_argument("--tokenizer", default=CONFIG)
    ap.add_argument("--subset", default="data/perception_test/subset.jsonl")
    ap.add_argument("--videos-dir", default="data/perception_test/videos")
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--modes", default="video:70,images:280")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    mx.random.seed(0)
    enc = Gemma4Inputs(args.tokenizer)
    model = attach_lora(load_model(local_model(args.model)), rank=8, alpha=16, checkpoint=True)
    rnd = [(k, (mx.random.normal(v.shape) * 0.01).astype(v.dtype) if k.endswith("lora_b") else v)
           for k, v in tree_flatten(model.trainable_parameters())]
    model.update(tree_unflatten(rnd))
    model.train()

    by_video = defaultdict(list)
    for r in map(json.loads, open(args.subset)):
        by_video[r["video_id"]].append(r)
    vids = [v for v, qs in by_video.items() if len(qs) >= 3][:args.videos]

    def per_question(mdl, items, feats, mode):
        losses = [letter_ce(letter_logits(mdl, ids, feats, mode, enc.label_ids), t) for ids, _, t in items]
        return mx.mean(mx.stack(losses))

    def grouped(mdl, items, feats, mode):
        prefix = items[0][0][:items[0][1]]
        logits = letter_logits_grouped(mdl, prefix, feats, mode, [ids[p:] for ids, p, _ in items], enc.label_ids)
        return letter_ce(logits, [t for _, _, t in items])

    results = []
    for spec in args.modes.split(","):
        mode, soft = spec.split(":")
        soft = int(soft)
        for vid in (vids if mode == "video" else vids[:1]):
            video = load_video(f"{args.videos_dir}/{vid}.mp4", n_frames=16)
            items, prefix = [], None
            for q in by_video[vid]:
                prompt, _ = render_choice(None, ChoiceQuestion(type="choice", instructions=q["question"],
                                                               criteria=dict(zip(LABELS, q["options"]))))
                e = enc.encode(prompt, video.frames, mode, soft, video.timestamps)
                if prefix is None:
                    prefix = e.input_ids[:e.prefix_len]
                    feats = vision_features(model, e.pixel_values, e.position_ids)
                    mx.eval(feats)
                assert e.input_ids[:e.prefix_len] == prefix, "questions of one video must share the prefix"
                items.append((e.input_ids, e.prefix_len, q["answer_id"]))
            out = {}
            for name, fn in (("per_question", per_question), ("grouped", grouped)):
                vg = nn.value_and_grad(model, lambda m: fn(m, items, feats, mode))
                t0 = time.perf_counter()
                loss, grads = vg(model)
                mx.eval(loss, grads)
                out[name] = (loss.item(), dict(tree_flatten(grads)), time.perf_counter() - t0)
            (l1, g1, t1), (l2, g2, t2) = out["per_question"], out["grouped"]
            num = sum(float(mx.sum((g1[k].astype(mx.float32) - g2[k].astype(mx.float32)) ** 2)) for k in g1)
            den = sum(float(mx.sum(g1[k].astype(mx.float32) ** 2)) for k in g1)
            dot = sum(float(mx.sum(g1[k].astype(mx.float32) * g2[k].astype(mx.float32))) for k in g1)
            n2 = sum(float(mx.sum(g2[k].astype(mx.float32) ** 2)) for k in g2)
            worst = max(float(mx.max(mx.abs(g1[k] - g2[k]))) / (float(mx.max(mx.abs(g1[k]))) + 1e-12) for k in g1)
            r = {"video": vid, "mode": f"{mode}:{soft}", "questions": len(items), "prefix_tokens": len(prefix),
                 "loss_per_question": l1, "loss_grouped": l2, "loss_abs_diff": abs(l1 - l2),
                 "grad_rel_l2_err": (num / den) ** 0.5, "grad_cosine": dot / (den * n2) ** 0.5,
                 "grad_worst_tensor_rel_maxdiff": worst, "time_per_question_s": t1, "time_grouped_s": t2}
            results.append(r)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
