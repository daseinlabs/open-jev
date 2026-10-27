"""Step 7: target selection. LoRA on the language model, trained on cached features, three targets.

    python -m video_train.distill prepare                      # question suffix ids, once
    python -m video_train.distill train --variant B --lr 1e-4  # one run: train, then score dev
    python -m video_train.distill grid                         # A, B, C at the shared learning rate
    python -m video_train.distill report                       # curves and dev comparison
    python -m video_train.distill train --variant B --lr 1e-4 --limit-videos 4 --tag smoke  # smoke test

Loss per question, over the option-letter probabilities (the readout OpenJev scores with):

    alpha * CE(student, correct answer) + (1 - alpha) * KL(teacher_T || student)

``teacher_T`` is d3-mini's step 5 probabilities raised to 1/T and renormalized (T = 1.5); the
student is not tempered. Variants: A alpha=1 (answer only), B alpha=0.5, C alpha=0 (teacher only,
every question kept, including the ones the teacher gets wrong).

Identical across runs: data (all 7,392 train questions), frames (step 6's cache: 16 frames, video
layout, 140 tokens/frame), seed (LoRA init and video order), passes, schedule, LoRA shape and one
shared learning rate (1e-4). Only alpha changes. The encoder and projector are frozen and never run.
A learning-rate search, if the curves call for one, is a later step on the winner only.

Diagnostics, to judge from the curves whether the shared rate suits each target:
- ``train_log.jsonl`` every ``--log-every`` steps: loss, CE, KL, batch accuracy, agreement with
  the teacher, mean confidence, gradient norm, update ratio (step size / LoRA weight size), and
  the LoRA B norm;
- ``dev_curve.jsonl`` every ``--dev-every`` steps: a fixed 100-video dev sample (videos d3-mini
  scored, so teacher agreement covers all of it): accuracy, NLL, ECE, confidence, teacher
  agreement, drift from zero-shot.

A step is ``--videos-per-step`` videos; each video's questions share one prefix pass
(``mlx_model.letter_logits_grouped``, gradients verified equal to per-question passes). The loss
is the mean over the step's questions. AdamW, linear warmup then cosine decay, gradient norm
clipped at 1.

Runs are resumable: the adapter and optimizer state are saved every ``--save-every`` steps to
``<DATA_ROOT>/checkpoints/step7/<tag>/`` and a rerun continues from there (the video order is
fixed by seed and pass, so it continues the same sequence). After training, dev is scored from
cached features into ``results/evals/dev/<tag>.jsonl`` (the evaluate.py format, resumable).

``report`` shows the learning curves and diagnostics, then reports, against
zero-shot video140 (step 3): accuracy overall and by area, NLL / ECE, paired McNemar, drift
(answers that differ from zero-shot), agreement with the teacher's top answer and accuracy on
the questions the teacher gets wrong (on the 712 dev questions d3-mini scored in step 4), and the
A-OKVQA check if it was run (``python -m video_train.aokvqa run --tag <tag> --adapter <dir>``).
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
from .cache_features import FORMAT, cache_dir, feature_path, load, suffix_ids
from .history import log_run

VARIANTS = {"A": 1.0, "B": 0.5, "C": 0.0}
LRS = (1e-4,)  # one shared rate; a learning-rate search, if any, comes after, on the winner
DEV_CURVE_VIDEOS = 100
TEACHER = "d3mini"
BASE = "video140"  # zero-shot dev rows in the training format (step 3)
STEP = "step7"


# ------------------------------------------------------------------ loss
def soften(p, T: float):
    """p^(1/T), renormalized, rows of a (n, k) array. T > 1 flattens; zeros stay zero."""
    p = np.asarray(p, dtype=np.float64)
    q = np.power(np.maximum(p, 0.0), 1.0 / T)
    return q / q.sum(axis=-1, keepdims=True)


def distill_loss(logits, answers, teacher, alpha: float):
    """Summed loss over questions, with its CE and KL parts (also sums).

    logits (n, k) float32; answers (n,) int; teacher (n, k) softened teacher probabilities.
    KL(teacher || student) = sum q (log q - log p); 0 log 0 = 0.
    """
    import mlx.core as mx
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ce = -mx.take_along_axis(logp, answers[:, None], axis=-1)[:, 0]
    q = teacher
    logq = mx.where(q > 0, mx.log(mx.maximum(q, 1e-30)), 0.0)
    kl = mx.sum(q * (logq - logp), axis=-1)
    return mx.sum(alpha * ce + (1 - alpha) * kl), mx.sum(ce), mx.sum(kl)


# ------------------------------------------------------------------ data
FORMAT_TAG = "video140"  # step 6's cache


def suffix_path():
    return os.path.join(cache_dir(FORMAT_TAG), "suffixes.json")


def _manifest(name):
    with storage.drive_guard():
        return [json.loads(l) for l in open(storage.path("manifests", f"{name}.jsonl"))]


def prepare(args):
    """Question ids after the video prefix, for train and dev (needs only the tokenizer)."""
    from .evaluate import prompt_of
    from .inputs import Gemma4Inputs
    from .models import CONFIG
    storage.require_root()
    enc = Gemma4Inputs(CONFIG)
    out, t0 = {}, time.time()
    for name in ("train", "dev"):
        for q in _manifest(name):
            out[f"{q['video_id']}/{q['qid']}"] = suffix_ids(enc, prompt_of(q), FORMAT["frames_as"],
                                                            FORMAT["soft_tokens"])
    os.makedirs(cache_dir(FORMAT_TAG), exist_ok=True)
    json.dump(out, open(suffix_path(), "w"))
    print(f"{len(out)} question suffixes in {time.time() - t0:.0f}s -> {suffix_path()}")


def load_data(name, T: float):
    """[(video_id, [question rows with 'suffix' and, for train, 'teacher' (softened)])], manifest order."""
    if not os.path.exists(suffix_path()):
        raise SystemExit("question suffixes missing: run `python -m video_train.distill prepare` first")
    suffixes = json.load(open(suffix_path()))
    teacher = {}
    if name == "train":
        from .evaluate import load_results
        teacher = load_results("train", TEACHER)
    by_video = defaultdict(list)
    for q in _manifest(name):
        k = (q["video_id"], q["qid"])
        q = dict(q, suffix=suffixes[f"{k[0]}/{k[1]}"])
        if name == "train":
            if k not in teacher:
                raise SystemExit(f"no teacher label for {k}: step 5 is incomplete")
            q["teacher"] = soften([teacher[k]["probs"]], T)[0]
            if len(q["teacher"]) != len(q["options"]):
                raise SystemExit(f"{k}: teacher has {len(q['teacher'])} probabilities for {len(q['options'])} options")
        by_video[q["video_id"]].append(q)
    for v, qs in by_video.items():
        if len({len(q["options"]) for q in qs}) != 1:
            raise SystemExit(f"{v}: questions with different option counts")
    return list(by_video.items())


def require_cached(data, name):
    missing = [v for v, _ in data if not os.path.exists(feature_path(FORMAT_TAG, v))]
    if missing:
        raise SystemExit(f"{len(missing)} {name} videos have no cached features (e.g. {missing[0]}): "
                         "finish step 6 (python -m video_train.cache_features run)")


# ------------------------------------------------------------------ training
def run_tag(variant, lr, args):
    return args.tag or f"s7_{variant}_lr{lr:.0e}"


def ckpt_dir(tag):
    return storage.path("checkpoints", STEP, tag)


def _order(n_videos, seed, epoch):
    idx = list(range(n_videos))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def train(args, variant=None, lr=None):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from . import mlx_model as mm
    from .adapters import save_adapter
    from .inputs import Gemma4Inputs
    from .models import CONFIG, local_model

    variant, lr = variant or args.variant, lr or args.lr
    alpha, tag = VARIANTS[variant], run_tag(variant, lr, args)
    storage.require_root()
    awake = storage.keep_awake()
    data = load_data("train", args.T)
    if args.limit_videos:
        data = data[:args.limit_videos]
    require_cached(data, "train")
    n_q = sum(len(qs) for _, qs in data)
    steps_per_epoch = math.ceil(len(data) / args.videos_per_step)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(total * args.warmup))
    config = {"variant": variant, "alpha": alpha, "T": args.T, "lr": lr, "epochs": args.epochs,
              "videos_per_step": args.videos_per_step, "warmup": args.warmup, "rank": args.rank,
              "lora_alpha": args.lora_alpha, "seed": args.seed, "clip": 1.0, "train_videos": len(data),
              "train_questions": n_q, "dev_every": args.dev_every,
              "dev_curve_videos": args.dev_curve_videos, **FORMAT}
    d = ckpt_dir(tag)
    os.makedirs(d, exist_ok=True)
    state_file = os.path.join(d, "state.json")
    state = json.load(open(state_file)) if os.path.exists(state_file) else {"step": 0, "config": config}
    if state["config"] != config:
        raise SystemExit(f"{d} was trained with {state['config']}, not {config}; use another --tag")
    if state["step"] >= total:
        print(f"{tag}: trained ({total} steps)", flush=True)
    else:
        mx.random.seed(args.seed)  # same LoRA initialization for every run
        model = mm.attach_lora(mm.load_model(local_model(FORMAT["model"])), rank=args.rank,
                               alpha=args.lora_alpha, checkpoint=True)
        schedule = optim.join_schedules([optim.linear_schedule(0.0, lr, warmup),
                                         optim.cosine_decay(lr, max(1, total - warmup))], [warmup])
        opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
        if state["step"]:
            model.load_weights(list(mx.load(os.path.join(d, "adapter.safetensors")).items()), strict=False)
            opt.state = tree_unflatten(list(mx.load(os.path.join(d, "optimizer.safetensors")).items()))
            print(f"{tag}: resuming at step {state['step']}/{total}", flush=True)
        model.train()
        label_ids = Gemma4Inputs(CONFIG).letter_ids(len(data[0][1][0]["options"]))

        def video_loss(m, feats, prefix, qs):
            logits = mm.letter_logits_grouped(m, prefix, feats, FORMAT["frames_as"], [q["suffix"] for q in qs],
                                              label_ids)
            return distill_loss(logits, mx.array([q["answer_id"] for q in qs]),
                                mx.array(np.stack([q["teacher"] for q in qs]), dtype=mx.float32), alpha), logits

        curve = dev_curve_data(args.T, args.dev_curve_videos) if args.dev_every else None
        log_f = open(os.path.join(d, "train_log.jsonl"), "a")
        t_start, s0, win = time.time(), state["step"], defaultdict(float)
        print(f"{tag}: alpha={alpha} lr={lr:g}, {len(data)} videos / {n_q} questions, {total} steps "
              f"({args.epochs} pass{'es' * (args.epochs > 1)} x {steps_per_epoch})", flush=True)
        for step in range(state["step"], total):
            storage.check_alive()
            epoch, k = divmod(step, steps_per_epoch)
            order = _order(len(data), args.seed, epoch)
            batch = [data[i] for i in order[k * args.videos_per_step:(k + 1) * args.videos_per_step]]
            acc = None
            for vid, qs in batch:
                feats, prefix, _ = load(FORMAT_TAG, vid)
                stats, grads = _value_parts(video_loss, model, feats, prefix, qs)
                acc = grads if acc is None else tree_map(mx.add, acc, grads)
                mx.eval(acc)
                for key, v in stats.items():
                    win[key] += v
            acc = tree_map(lambda g: g / sum(len(qs) for _, qs in batch), acc)
            acc, gnorm = optim.clip_grad_norm(acc, 1.0)
            before = dict(tree_flatten(model.trainable_parameters()))
            opt.update(model, acc)
            mx.eval(model.trainable_parameters(), opt.state)
            win["gnorm"] += float(gnorm)
            win["update_ratio"] += _update_ratio(before, dict(tree_flatten(model.trainable_parameters())))
            win["n"] += 1
            done = step + 1
            if done % args.log_every == 0 or done == total:
                q = win["q"]
                rec = {"step": done, "lr": float(schedule(done)), "loss": win["loss"] / q, "ce": win["ce"] / q,
                       "kl": win["kl"] / q, "acc": win["correct"] / q, "teacher_agree": win["agree"] / q,
                       "confidence": win["conf"] / q, "gnorm": win["gnorm"] / win["n"],
                       "update_ratio": win["update_ratio"] / win["n"],
                       "lora_b_norm": _norm([v for k, v in tree_flatten(model.trainable_parameters())
                                             if k.endswith("lora_b")])}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                el = time.time() - t_start
                print(f"[{done}/{total}] loss {rec['loss']:.4f} (ce {rec['ce']:.4f}, kl {rec['kl']:.4f}) "
                      f"acc {rec['acc']:.1%} teacher {rec['teacher_agree']:.1%} gnorm {rec['gnorm']:.2f} "
                      f"upd {rec['update_ratio']:.1e}, {el / (done - s0):.1f}s/step, "
                      f"eta {el / (done - s0) * (total - done) / 60:.0f} min", flush=True)
                win.clear()
            if args.dev_every and (done % args.dev_every == 0 or done == total):
                _dev_curve(model, curve, label_ids, done, d)
            if done % args.save_every == 0 or done == total:
                save_adapter(model, d, args.rank, args.lora_alpha, extra={"step7": config})
                mx.save_safetensors(os.path.join(d, "optimizer.safetensors"), dict(tree_flatten(opt.state)))
                state["step"] = done
                tmp = state_file + ".tmp"
                json.dump(state, open(tmp, "w"), indent=1)
                os.replace(tmp, state_file)
        log_f.close()
        log_run(STEP, "train", dict(config, tag=tag), {"manifest": "train", "teacher": TEACHER}, args.seed,
                {"steps": total, "train_s": time.time() - t_start})
        del model, opt
    if awake:
        awake.terminate()
    if not args.no_eval:
        evaluate_dev(tag, config, args)
    return tag


def _value_parts(video_loss, model, feats, prefix, qs):
    """Sums over the video's questions (loss, ce, kl, correct, teacher agreement, confidence, count)
    and the gradient of the loss. The logged parts come from the same forward pass: a second pass
    without gradients would double the cost."""
    import mlx.core as mx
    import mlx.nn as nn
    parts = {}

    def f(m, fe, pr, q):
        (loss, ce, kl), logits = video_loss(m, fe, pr, q)
        parts.update(ce=ce, kl=kl, logits=logits)
        return loss

    loss, grads = nn.value_and_grad(model, f)(model, feats, prefix, qs)
    mx.eval(loss, grads, parts["ce"], parts["kl"], parts["logits"])
    p = np.array(mx.softmax(parts["logits"], axis=-1)).astype(float)
    pred = p.argmax(1)
    stats = {"loss": loss.item(), "ce": parts["ce"].item(), "kl": parts["kl"].item(),
             "correct": float(sum(int(a == q["answer_id"]) for a, q in zip(pred, qs))),
             "agree": float(sum(int(a == int(np.argmax(q["teacher"]))) for a, q in zip(pred, qs))),
             "conf": float(p.max(1).sum()), "q": float(len(qs))}
    return stats, grads


def _norm(arrays):
    import mlx.core as mx
    return math.sqrt(sum(float(mx.sum(a.astype(mx.float32) ** 2)) for a in arrays))


def _update_ratio(before, after):
    """||step|| / ||weights|| over the LoRA parameters. Read it as a trend and across variants, not
    against a fixed threshold: it starts high (LoRA B starts at zero) and should fall as the weights
    grow and the schedule decays; one variant far above the others is taking much larger steps."""
    step = _norm([after[k] - before[k] for k in before])
    return step / max(_norm(list(before.values())), 1e-12)


# ------------------------------------------------------------------ learning curves
def dev_curve_data(T, n_videos=DEV_CURVE_VIDEOS):
    """A fixed sample of ``n_videos`` dev videos that d3-mini scored (step 4), so teacher
    agreement covers every question of the sample. Same sample for every run (seed 0)."""
    from .evaluate import load_results
    teacher = load_results("dev", TEACHER)
    scored = {v for v, _ in teacher}
    data = [(v, qs) for v, qs in load_data("dev", T) if v in scored]
    data = sorted(random.Random(0).sample(data, min(n_videos, len(data))))
    require_cached(data, "dev")
    return data


def curve_stats(rows, teacher, base):
    from .evaluate import metrics
    m = metrics([{"latency_s": 0.0, "decode_s": 0.0, **r} for r in rows])  # timing is not tracked here
    keys = [(r["video_id"], r["qid"]) for r in rows]
    return {"n": m["n"], "accuracy": m["accuracy"], "nll": m["nll"], "ece": m["ece"],
            "confidence": m["mean_confidence"],
            "teacher_agree": float(np.mean([r["pred"] == teacher[k]["pred"] for r, k in zip(rows, keys)
                                            if k in teacher])),
            "drift": float(np.mean([r["pred"] != base[k]["pred"] for r, k in zip(rows, keys)]))}


def _dev_curve(model, data, label_ids, step, d):
    """Score the dev sample in eval mode and append to ``dev_curve.jsonl``; back to training mode."""
    from .evaluate import load_results
    model.eval()
    rows = [r for vid, qs in data for r in _score_video(model, vid, qs, label_ids)[0]]
    model.train()
    rec = {"step": step, **curve_stats(rows, load_results("dev", TEACHER), load_results("dev", BASE))}
    with open(os.path.join(d, "dev_curve.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  dev sample @ {step}: acc {rec['accuracy']:.1%} nll {rec['nll']:.3f} ece {rec['ece']:.3f} "
          f"teacher {rec['teacher_agree']:.1%} drift {rec['drift']:.1%}", flush=True)


def _score_video(model, vid, qs, label_ids):
    """(rows without run config, load s, model s) for one video's questions from cached features."""
    import mlx.core as mx
    from . import mlx_model as mm
    t0 = time.perf_counter()
    feats, prefix, _ = load(FORMAT_TAG, vid)
    mx.eval(feats)
    t1 = time.perf_counter()
    logits = mm.letter_logits_grouped(model, prefix, feats, FORMAT["frames_as"], [q["suffix"] for q in qs],
                                      label_ids)
    probs = np.array(mx.softmax(logits, axis=-1)).astype(float)
    t2 = time.perf_counter()
    rows = [{"video_id": vid, "qid": q["qid"], "area": q["area"], "reasoning": q.get("reasoning"),
             "answer_id": q["answer_id"], "probs": p.tolist(), "pred": int(p.argmax()),
             "correct": int(p.argmax()) == q["answer_id"]} for q, p in zip(qs, probs)]
    return rows, t1 - t0, t2 - t1


# ------------------------------------------------------------------ evaluation
def evaluate_dev(tag, config, args):
    """Score dev from cached features with the trained adapter; rows in the evaluate.py format."""
    import mlx.core as mx
    from . import mlx_model as mm
    from .evaluate import load_results, metrics, result_path
    from .inputs import Gemma4Inputs
    from .models import CONFIG, local_model

    storage.require_root()
    out = result_path("dev", tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set(load_results("dev", tag)) if os.path.exists(out) else set()
    data = [(v, [q for q in qs if (v, q["qid"]) not in done]) for v, qs in load_data("dev", args.T)]
    data = [(v, qs) for v, qs in data if qs]
    if args.limit_dev_videos:
        data = data[:args.limit_dev_videos]
    require_cached(data, "dev")
    print(f"{tag} on dev: {len(done)} done, {sum(len(q) for _, q in data)} to score", flush=True)
    if data:
        model = mm.attach_lora(mm.load_model(local_model(FORMAT["model"])), rank=args.rank,
                               alpha=args.lora_alpha, checkpoint=False)
        model.load_weights(list(mx.load(os.path.join(ckpt_dir(tag), "adapter.safetensors")).items()),
                           strict=False)
        model.eval()
        enc = Gemma4Inputs(CONFIG)
        row_cfg = {"set": "dev", "tag": tag, "model": FORMAT["model"], "mode": "encode_once",
                   "frames": FORMAT["frames"], "frames_as": FORMAT["frames_as"],
                   "soft_tokens": FORMAT["soft_tokens"], "adapter": ckpt_dir(tag),
                   **{k: config[k] for k in ("variant", "alpha", "T", "lr")}}
        for vi, (vid, qs) in enumerate(data):
            rows, load_s, model_s = _score_video(model, vid, qs, enc.letter_ids(len(qs[0]["options"])))
            with storage.drive_guard(), open(out, "a") as f:
                for r in rows:
                    f.write(json.dumps({**r, **row_cfg, "n_video_questions": len(qs),
                                        "decode_s": load_s / len(qs), "latency_s": model_s / len(qs)}) + "\n")
            if (vi + 1) % 50 == 0 or vi + 1 == len(data):
                print(f"  dev [{vi + 1}/{len(data)} videos]", flush=True)
        del model
    s = metrics(list(load_results("dev", tag).values()))
    if not args.limit_dev_videos:
        log_run(STEP, "eval", row_cfg if data else {"tag": tag}, {"manifest": "dev"}, args.seed, s)
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}), flush=True)


def grid(args):
    lrs = [float(x) for x in args.lrs.split(",")]
    for variant in args.variants.split(","):
        for lr in lrs:
            train(args, variant, lr)


# ------------------------------------------------------------------ report
def _pct(x):
    return f"{x:.1%}" if x is not None else "-"


def _curve_lines(tags, teacher, base):
    """Dev-sample learning curves (step 0 = zero-shot) and the end-of-training diagnostics."""
    curves, logs = {}, {}
    for v, tag in tags.items():
        d = ckpt_dir(tag)
        for name, store in (("dev_curve.jsonl", curves), ("train_log.jsonl", logs)):
            path = os.path.join(d, name)
            if os.path.exists(path):
                store[v] = {r["step"]: r for r in map(json.loads, open(path))}  # a resumed run's repeat wins
    if not curves:
        return []
    cfg = json.load(open(os.path.join(ckpt_dir(next(iter(tags.values()))), "state.json")))["config"]
    sample = [(vid, q["qid"]) for vid, qs in dev_curve_data(cfg["T"], cfg["dev_curve_videos"]) for q in qs]
    zero = curve_stats([dict(base[k], video_id=k[0], qid=k[1]) for k in sample], teacher, base)
    steps = sorted({s for c in curves.values() for s in c})
    lines = [f"## Learning curves (dev sample: {zero['n']} questions, {cfg['dev_curve_videos']} videos)", "",
             "Accuracy / NLL / teacher agreement. Step 0 is zero-shot.", "",
             "| step | " + " | ".join(curves) + " |", "|---|" + "---|" * len(curves),
             "| 0 | " + " | ".join(f"{zero['accuracy']:.1%} / {zero['nll']:.3f} / {zero['teacher_agree']:.0%}"
                                   for _ in curves) + " |"]
    for s in steps:
        lines.append(f"| {s} | " + " | ".join(
            f"{c[s]['accuracy']:.1%} / {c[s]['nll']:.3f} / {c[s]['teacher_agree']:.0%}" if s in c else "-"
            for c in curves.values()) + " |")
    lines += ["", "Peak vs final on the dev sample (a peak well before the end, then a drop, points to a "
              "learning rate that is too high; still rising at the end, too low):", ""]
    for v, c in curves.items():
        peak = max(c.values(), key=lambda r: r["accuracy"])
        last = c[max(c)]
        lines.append(f"- {v}: peak {peak['accuracy']:.1%} at step {peak['step']}, final {last['accuracy']:.1%} "
                     f"at step {last['step']}")
    if logs:
        lines += ["", "Training diagnostics, first and last logged window "
                  "(update ratio = step size / LoRA weight size):", "",
                  "| variant | loss | train acc | teacher agree | grad norm | update ratio |", "|---|---|---|---|---|---|"]
        for v, lg in logs.items():
            a, b = lg[min(lg)], lg[max(lg)]
            f = lambda k, fmt: f"{format(a[k], fmt)} → {format(b[k], fmt)}" if k in a and k in b else "-"
            lines.append(f"| {v} | {f('loss', '.3f')} | {f('acc', '.1%')} | {f('teacher_agree', '.1%')} | "
                         f"{f('gnorm', '.2f')} | {f('update_ratio', '.1e')} |")
    return lines + [""]


def report(args):
    from .evaluate import load_results, metrics, result_path
    from .teacher import paired
    storage.require_root()
    base = load_results("dev", BASE)
    teacher = load_results("dev", TEACHER)
    runs = {}
    for variant in VARIANTS:
        for lr in [float(x) for x in args.lrs.split(",")]:
            tag = f"s7_{variant}_lr{lr:.0e}"
            if os.path.exists(result_path("dev", tag)):
                rows = load_results("dev", tag)
                if len(rows) == len(base):
                    runs.setdefault(variant, []).append((tag, lr, rows))
    if not runs:
        raise SystemExit("no complete step 7 dev results yet")
    mb = metrics(list(base.values()))
    lines = [f"# {STEP}: target selection", "",
             f"Perception Test dev ({mb['n']} questions), cached features, LoRA on the language model only. "
             f"Teacher columns on the {len(teacher)} dev questions d3-mini scored (step 4).", ""]
    best = {}
    for variant, rs in runs.items():
        # With several learning rates: best by accuracy, NLL breaks ties.
        best[variant] = max(rs, key=lambda r: (metrics(list(r[2].values()))["accuracy"],
                                               -metrics(list(r[2].values()))["nll"]))
    lines += _curve_lines({v: best[v][0] for v in sorted(best)}, teacher, base)
    lines += ["", "## Best learning rate per variant", "",
              "| | accuracy | net vs zero-shot (p) | NLL | ECE | conf ≥ 0.8: cov / acc | drift | "
              "teacher agreement | acc where teacher wrong |", "|---|---|---|---|---|---|---|---|---|"]
    t_keys = sorted(teacher)
    t_wrong = [k for k in t_keys if not teacher[k]["correct"]]

    def teacher_cols(rows):
        agree = np.mean([rows[k]["pred"] == teacher[k]["pred"] for k in t_keys])
        wrong = np.mean([rows[k]["correct"] for k in t_wrong]) if t_wrong else None
        return agree, wrong

    a0, w0 = teacher_cols(base)
    lines.append(f"| zero-shot ({BASE}) | {mb['accuracy']:.1%} ± {mb['ci95']:.1%} | - | {mb['nll']:.3f} | "
                 f"{mb['ece']:.3f} | {mb['coverage_conf_ge_0.8']:.0%} / {_pct(mb['accuracy_conf_ge_0.8'])} | - | "
                 f"{a0:.1%} | {_pct(w0)} |")
    summary, areas = {}, {"zero-shot": mb["by_area"]}
    for variant in sorted(best):
        tag, lr, rows = best[variant]
        m = metrics(list(rows.values()))
        g, lost, p = paired(base, rows)
        drift = np.mean([rows[k]["pred"] != base[k]["pred"] for k in base])
        agree, wrong = teacher_cols(rows)
        lines.append(f"| {variant} (α={VARIANTS[variant]}, lr {lr:g}) | {m['accuracy']:.1%} ± {m['ci95']:.1%} | "
                     f"{g - lost:+d} ({p:.2g}) | {m['nll']:.3f} | {m['ece']:.3f} | "
                     f"{m['coverage_conf_ge_0.8']:.0%} / {_pct(m['accuracy_conf_ge_0.8'])} | {drift:.1%} | "
                     f"{agree:.1%} | {_pct(wrong)} |")
        areas[variant] = m["by_area"]
        summary[variant] = {"tag": tag, "lr": lr, "accuracy": m["accuracy"], "nll": m["nll"], "ece": m["ece"],
                            "gained": g, "lost": lost, "p": p, "drift": float(drift),
                            "teacher_agreement": float(agree), "acc_teacher_wrong": wrong}
    lines += ["", "## By area", "", "| area | n | " + " | ".join(areas) + " |", "|---|---|" + "---|" * len(areas)]
    for area, v in mb["by_area"].items():
        lines.append(f"| {area} | {v['n']} | " + " | ".join(f"{a[area]['accuracy']:.1%}" for a in areas.values()) + " |")
    # Variants against each other: is the winner separable from the runner-up on dev?
    order = sorted(best, key=lambda v: -summary[v]["accuracy"])
    if len(order) > 1:
        lines += ["", "## Variants against each other (paired)", ""]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                g, lost, p = paired(best[b][2], best[a][2])
                lines.append(f"- {a} vs {b}: {a} gains {g}, loses {lost} (net {g - lost:+d}, p={p:.2g})")
    from . import aokvqa
    ok_lines = []
    for variant in sorted(best):
        tag = best[variant][0]
        try:
            r = aokvqa.paired(aokvqa._load("base", 500), aokvqa._load(tag, 500))
        except FileNotFoundError:
            continue
        ok_lines.append(f"- {variant}: {r['acc_before']:.1%} -> {r['acc_after']:.1%} "
                        f"({r['lost']} lost, {r['gained']} gained, p={r['mcnemar_p']:.3f})")
        summary[variant]["aokvqa"] = r
    lines += ["", "## A-OKVQA (500 questions, adapter on; information only)", ""]
    lines += ok_lines or [f"- not run: `python -m video_train.aokvqa run --tag <tag> --adapter "
                          f"{ckpt_dir('<tag>')}` for each winner"]
    text = "\n".join(lines) + "\n"
    out = storage.path("results", STEP)
    os.makedirs(out, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    log_run(STEP, "eval", {"what": "report"}, {"manifest": "dev"}, None, summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--variant", choices=VARIANTS, required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--tag", default=None, help="default s7_<variant>_lr<lr>")
    g = sub.add_parser("grid")
    g.add_argument("--variants", default="A,B,C")
    g.add_argument("--lrs", default=",".join(f"{x:g}" for x in LRS))
    g.set_defaults(tag=None)
    for p in (t, g):
        p.add_argument("--T", type=float, default=1.5, help="teacher temperature")
        p.add_argument("--epochs", type=int, default=1, help="passes over the train set")
        p.add_argument("--videos-per-step", type=int, default=8)
        p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
        p.add_argument("--rank", type=int, default=8)
        p.add_argument("--lora-alpha", type=float, default=16.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--log-every", type=int, default=10)
        p.add_argument("--save-every", type=int, default=25)
        p.add_argument("--limit-videos", type=int, default=0, help="smoke test: train on the first n videos")
        p.add_argument("--limit-dev-videos", type=int, default=0, help="smoke test: score n dev videos")
        p.add_argument("--no-eval", action="store_true", help="train only")
        p.add_argument("--dev-every", type=int, default=50,
                       help="score the dev sample every n steps (0: off)")
        p.add_argument("--dev-curve-videos", type=int, default=DEV_CURVE_VIDEOS,
                       help="dev sample size, from the videos d3-mini scored")
    r = sub.add_parser("report")
    r.add_argument("--lrs", default=",".join(f"{x:g}" for x in LRS))
    args = ap.parse_args(argv)
    try:
        {"prepare": prepare, "train": train, "grid": grid, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
