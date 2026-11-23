"""Stage 3 and Stage B of docs/video-decision-model-plan.md: LoRA on a mix (Qwen3-VL-4B, MLX 8-bit).

    python -m video_train.vdm_train train                                  # Stage A: one pass, ~58 h (resumable)
    python -m video_train.vdm_train train --tag stageB_ep1 --mix vdm/manifests/stageB_mix.jsonl \
        --lr 5e-5 --init-adapter "<DATA_ROOT>/vdm/checkpoints/stageA_ep1"  # Stage B: ~27 h
    python -m video_train.vdm_train train --tag smoke --limit-videos 16 --dev-videos 2 --save-every 1
    python -m video_train.vdm_eval run --set all --tag stageA_ep1 --adapter "<DATA_ROOT>/vdm/checkpoints/stageA_ep1"

Data: ``--mix`` (default ``vdm/manifests/train_mix.jsonl``), one row per video with its questions packed into one
multi-turn conversation (``qwen3vl.Inputs.encode``); 16 frames, 192 vision tokens per 2 frames
(the stage 1 format). One forward and backward per video; the encoder stays frozen.

Loss: mean over the step's assistant turns of each turn's mean token cross-entropy (answer +
``<|im_end|>``), so every answer counts once whatever its length. A step is ``--videos-per-step``
videos (gradients summed, divided by the step's turns). AdamW, linear warmup then cosine decay,
gradient norm clipped at 1; LoRA rank 8 / alpha 16 on all LLM linear layers (the s7/s8 recipe).
Pass ``e`` visits the videos in a fixed order (seed, pass), so a rerun continues the same sequence.
The next video is decoded from the drive while the current one trains.

Logs, in ``<DATA_ROOT>/vdm/checkpoints/<tag>/``:
- ``train_log.jsonl`` every ``--log-every`` steps: loss (overall and per source), gradient norm,
  learning rate, s/step;
- ``dev_curve.jsonl`` every ``--dev-every`` steps: a fixed sample of each dev set
  (``--dev-videos`` videos per set, all their questions), scored as ``vdm_eval`` scores it:
  accuracy per set and mean NLL of the right option on multiple choice.

``--init-adapter``: start from a trained LoRA (same rank and alpha) instead of a fresh one; the
optimizer state and schedule start fresh.

Resumable: the adapter, optimizer state and step are saved every ``--save-every`` steps; a rerun
under the same tag continues (and refuses a different config). The adapter is in the
``adapters.save_adapter`` format, so ``vdm_eval --adapter`` and the server load it as is.
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import storage
from .history import log_run
from .models import local_model, store_path
from .vdm_data import DEV, rows, kind
from .vdm_eval import MLX8, Qwen

MIX = "vdm/manifests/train_mix.jsonl"


def ckpt_dir(tag):
    return storage.path("vdm", "checkpoints", tag)


def order(n, seed, epoch):
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def dev_sample(n_videos, seed=0):
    """{set: [rows]}: ``n_videos`` videos per dev set (fixed by seed), all their questions."""
    out = {}
    for name, rel in DEV.items():
        by = defaultdict(list)
        for r in rows(rel):
            by[r["video_id"]].append(r)
        keep = set(random.Random(f"{seed}-{name}").sample(sorted(by), min(n_videos, len(by))))
        out[name] = [r for v in sorted(keep) for r in by[v]]
    return out


def dev_curve(b, sample, frames, step, path):
    from openjev.video import load_video
    b.model.eval()
    rec, t0 = {"step": step}, time.time()
    for name, rs in sample.items():
        by = defaultdict(list)
        for r in rs:
            by[r["video_id"]].append(r)
        right, nll = [], []
        for vqs in by.values():
            with storage.drive_guard():
                video = load_video(storage.path(vqs[0]["video"]), n_frames=frames)
            b.start(video, vqs[0])
            for r in vqs:
                res = b.score(r)
                right.append(bool(res["correct"]))
                if kind(r) == "mc":
                    nll.append(-math.log(max(res["probs"][r["answer_id"]], 1e-12)))
        rec[name] = {"n": len(right), "acc": float(np.mean(right)), "nll": float(np.mean(nll)) if nll else None}
    rec["s"] = time.time() - t0
    with storage.drive_guard(), open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("dev " + ", ".join(f"{k} {v['acc']:.1%}" for k, v in rec.items() if isinstance(v, dict))
          + f" ({rec['s'] / 60:.0f} min)", flush=True)
    b.model.train()


def train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from openjev.video import load_video
    from . import qwen3vl as q3
    from .adapters import save_adapter

    storage.require_root()
    awake = storage.keep_awake()
    data = rows(args.mix)
    if args.limit_videos:
        data = data[:args.limit_videos]
    steps_per_epoch = math.ceil(len(data) / args.videos_per_step)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(total * args.warmup))
    config = {"model": MLX8, "mix": args.mix, **({"init_adapter": args.init_adapter} if args.init_adapter else {}), "videos": len(data), "turns": sum(len(r["turns"]) for r in data),
              "epochs": args.epochs, "videos_per_step": args.videos_per_step, "lr": args.lr, "warmup": args.warmup,
              "rank": args.rank, "lora_alpha": args.lora_alpha, "clip": 1.0, "seed": args.seed,
              "frames": args.frames, "tokens": args.tokens, "loss": "mean over turns of per-turn token CE",
              "dev_videos": args.dev_videos, "dev_every": args.dev_every}
    d = ckpt_dir(args.tag)
    os.makedirs(d, exist_ok=True)
    state_file = os.path.join(d, "state.json")
    state = json.load(open(state_file)) if os.path.exists(state_file) else {"step": 0, "config": config}
    if state["config"] != config:
        sys.exit(f"{d} was trained with {state['config']}, not {config}; use another --tag")
    if state["step"] >= total:
        print(f"{args.tag}: trained ({total} steps)")
        return

    mx.random.seed(args.seed)  # same LoRA initialization for every run
    inp = q3.Inputs(store_path(MLX8), args.tokens)
    model = q3.attach_lora(q3.load_model(local_model(MLX8)), rank=args.rank, alpha=args.lora_alpha)
    schedule = optim.join_schedules([optim.linear_schedule(0.0, args.lr, warmup),
                                     optim.cosine_decay(args.lr, max(1, total - warmup))], [warmup])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    if state["step"]:
        model.load_weights(list(mx.load(os.path.join(d, "adapter.safetensors")).items()), strict=False)
        opt.state = tree_unflatten(list(mx.load(os.path.join(d, "optimizer.safetensors")).items()))
        print(f"{args.tag}: resuming at step {state['step']}/{total}", flush=True)
    elif args.init_adapter:
        cfg = json.load(open(os.path.join(args.init_adapter, "adapter_config.json")))
        if (cfg["rank"], cfg["alpha"]) != (args.rank, args.lora_alpha):
            sys.exit(f"{args.init_adapter} is rank {cfg['rank']} / alpha {cfg['alpha']}, "
                     f"not {args.rank} / {args.lora_alpha}")
        w = mx.load(os.path.join(args.init_adapter, "adapter.safetensors"))
        lora = {k for k, _ in tree_flatten(model.trainable_parameters()) if k.endswith(("lora_a", "lora_b"))}
        if set(w) != lora:
            sys.exit(f"{args.init_adapter}: LoRA layers differ from this model's ({len(w)} vs {len(lora)} tensors)")
        model.load_weights(list(w.items()), strict=False)
        print(f"{args.tag}: starting from {args.init_adapter}", flush=True)
    b = Qwen(args.tokens, model=model, inp=inp)
    sample = dev_sample(args.dev_videos) if args.dev_every else None
    curve = os.path.join(d, "dev_curve.jsonl")
    model.train()

    loss_grad = nn.value_and_grad(model, lambda m, enc, vis: q3.turn_ce(m, inp, enc, vis)[0])

    def decode(r):
        with storage.drive_guard():
            return load_video(storage.path(r["video"]), n_frames=args.frames)

    seq = [(s, data[i]) for s in range(state["step"], total)
           for i in order(len(data), args.seed, s // steps_per_epoch)[
               (s % steps_per_epoch) * args.videos_per_step:(s % steps_per_epoch + 1) * args.videos_per_step]]
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(decode, seq[0][1])
    log_f = open(os.path.join(d, "train_log.jsonl"), "a")
    print(f"{args.tag}: {len(data)} videos / {config['turns']} turns, {total} steps "
          f"({args.epochs} pass{'es' * (args.epochs > 1)} x {steps_per_epoch}), warmup {warmup}", flush=True)
    t_start, s0, win, i = time.time(), state["step"], defaultdict(float), 0
    for step in range(state["step"], total):
        storage.check_alive()
        acc, n_turns, t_step = None, 0, time.time()
        while i < len(seq) and seq[i][0] == step:
            r = seq[i][1]
            video = nxt.result()
            i += 1
            if i < len(seq):
                nxt = pool.submit(decode, seq[i][1])
            enc = inp.encode([tuple(t) for t in r["turns"]], video, False)
            vis = q3.vision_features(model, enc)
            loss, grads = loss_grad(model, enc, vis)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(loss, acc)
            n = len(r["turns"])
            n_turns += n
            win["loss"] += loss.item()
            win["turns"] += n
            win[f"loss_{r['source']}"] += loss.item()
            win[f"turns_{r['source']}"] += n
        acc, gnorm = optim.clip_grad_norm(tree_map(lambda g: g / n_turns, acc), 1.0)
        opt.update(model, acc)
        mx.eval(model.trainable_parameters(), opt.state)
        win["gnorm"] += float(gnorm)
        win["steps"] += 1
        win["s"] += time.time() - t_step
        done = step + 1
        if done % args.log_every == 0 or done == total:
            srcs = sorted(k[5:] for k in win if k.startswith("loss_"))
            rec = {"step": done, "lr": float(schedule(done)), "loss": win["loss"] / win["turns"],
                   **{f"loss_{s}": win[f"loss_{s}"] / win[f"turns_{s}"] for s in srcs},
                   "gnorm": win["gnorm"] / win["steps"], "s_per_step": win["s"] / win["steps"],
                   "peak_gb": mx.get_peak_memory() / 1e9}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            el = time.time() - t_start
            print(f"[{done}/{total}] loss {rec['loss']:.4f} gnorm {rec['gnorm']:.2f} lr {rec['lr']:.1e} "
                  f"{rec['s_per_step']:.0f}s/step, peak {rec['peak_gb']:.1f} GB, "
                  f"eta {el / (done - s0) * (total - done) / 3600:.1f} h", flush=True)
            win.clear()
        if done % args.save_every == 0 or done == total:
            save_adapter(model, d, args.rank, args.lora_alpha, extra={"vdm_stage3": config, "step": done})
            mx.save_safetensors(os.path.join(d, "optimizer.safetensors"), dict(tree_flatten(opt.state)))
            state["step"] = done
            with open(state_file + ".tmp", "w") as f:
                json.dump(state, f, indent=1)
            os.replace(state_file + ".tmp", state_file)
        if sample and (done % args.dev_every == 0 or done == total):
            dev_curve(b, sample, args.frames, done, curve)
    pool.shutdown()
    log_f.close()
    log_run("vdm-3", "train", dict(config, tag=args.tag), {"manifest": args.mix}, args.seed,
            {"steps": total, "train_s": time.time() - t_start})
    print(f"done: python -m video_train.vdm_eval run --set all --tag {args.tag} --adapter \"{d}\"")
    if awake:
        awake.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="stageA_ep1")
    t.add_argument("--mix", default=MIX, help="training manifest (relative to DATA_ROOT)")
    t.add_argument("--init-adapter", default="", help="start from this trained LoRA directory")
    t.add_argument("--epochs", type=int, default=1, help="passes over the mix")
    t.add_argument("--videos-per-step", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--lora-alpha", type=float, default=16.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--frames", type=int, default=16)
    t.add_argument("--tokens", type=int, default=192, help="vision tokens per 2 frames")
    t.add_argument("--log-every", type=int, default=5)
    t.add_argument("--save-every", type=int, default=25)
    t.add_argument("--dev-every", type=int, default=150, help="dev sample every n steps (0: off)")
    t.add_argument("--dev-videos", type=int, default=15, help="dev sample: videos per dev set")
    t.add_argument("--limit-videos", type=int, default=0, help="smoke test: the first n videos of the mix")
    args = ap.parse_args(argv)
    try:
        train(args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
