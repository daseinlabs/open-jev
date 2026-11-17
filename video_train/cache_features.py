"""Step 6: cache Gemma 4's frozen vision features (encoder + projector) for every video.

    python -m video_train.cache_features run                       # train + dev, the training format
    python -m video_train.cache_features run --sets dev --limit-videos 3   # smoke test
    python -m video_train.cache_features check --videos 20         # cached vs fresh, against video140 rows
    python -m video_train.cache_features status

Format = step 3's choice: 16 uniform frames, Gemma 4's video layout, 140 tokens/frame, ``mm:ss``
timestamps, on the MLX 8-bit model. The encoder and projector stay frozen through step 7/8, so a
video's features never change: encoding each one once replaces ~one encoding per video per pass,
per variant, per learning rate.

One file per video, ``<LOCAL_CACHE>/features/<tag>/<video_id>.safetensors``:
- ``features``: (1, frames x tokens, hidden), the projector output exactly as
  ``mlx_model.vision_features`` returns it (same dtype; nothing is rounded);
- ``prefix_ids``: the token ids up to and including the last frame's end marker (BOS, timestamps,
  frame markers, soft tokens). A question's own ids follow it; :func:`suffix_ids` builds them
  from the prompt alone, without the video.
- ``timestamps``: the frame times in seconds.

The cache lives on the internal disk (random reads during training; the HDD does ~18 ms per
seek). It is rebuildable, so nothing is kept on the drive. Resumable: a video is skipped once its
file exists; files are written to a temporary name and renamed, so an interrupted write leaves
nothing behind. ``check`` scores dev questions from cached features plus :func:`suffix_ids`
(no video decoding) and compares them with the video140 dev rows of step 3, which encoded the
video fresh.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from . import storage
from .history import log_run
from .models import CONFIG, local_model

FORMAT = {"model": "gemma-4-E4B-it-mlx-8bit", "frames": 16, "frames_as": "video", "soft_tokens": 140}
SETS = {"train": "train", "dev": "dev", "nextqa": "nextqa_val1000"}
REFERENCE = "video140"  # step 3's dev rows in this format, encoded fresh
SPARE_BYTES = 10e9      # keep this much free on the Mac


def cache_dir(tag: str) -> str:
    return os.path.join(storage.LOCAL_CACHE, "features", tag)


def feature_path(tag: str, video_id: str) -> str:
    return os.path.join(cache_dir(tag), f"{video_id}.safetensors")


def load(tag: str, video_id: str):
    """(features (1, n, hidden) mx.array, prefix_ids list[int], timestamps list[float])."""
    import mlx.core as mx
    a = mx.load(feature_path(tag, video_id))
    return a["features"], np.array(a["prefix_ids"]).tolist(), np.array(a["timestamps"]).tolist()


def suffix_ids(enc, prompt: str, frames_as: str = FORMAT["frames_as"],
               soft_tokens: int = FORMAT["soft_tokens"]) -> list[int]:
    """A question's ids after the video prefix. The prefix ends with a special end-of-image token, so
    the text after it tokenizes the same whatever the frames are: one dummy frame stands in."""
    from PIL import Image
    e = enc.encode(prompt, [Image.new("RGB", (64, 64))], frames_as, soft_tokens, [0.0])
    return e.input_ids[e.prefix_len:]


def _videos(sets):
    """{video_id: manifest video path} over ``sets``, in manifest order."""
    out = {}
    for s in sets:
        with storage.drive_guard():
            for line in open(storage.path("manifests", f"{SETS[s]}.jsonl")):
                r = json.loads(line)
                out.setdefault(r["video_id"], r["video"])
    return out


def _check_meta(tag, fmt):
    d = cache_dir(tag)
    os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "format.json")
    if os.path.exists(meta):
        old = json.load(open(meta))
        if old != fmt:
            raise SystemExit(f"{d} holds features in format {old}, not {fmt}; use another --tag")
    else:
        json.dump(fmt, open(meta, "w"), indent=1)


def _encode_video(model, enc, path, fmt):
    import mlx.core as mx
    from openjev.video import load_video
    from . import mlx_model as mm
    t0 = time.perf_counter()
    with storage.drive_guard():
        video = load_video(path, n_frames=fmt["frames"])
    e = enc.encode("", video.frames, fmt["frames_as"], fmt["soft_tokens"], video.timestamps)
    t1 = time.perf_counter()
    feats = mm.vision_features(model, e.pixel_values, e.position_ids)
    mx.eval(feats)
    return feats, e.input_ids[:e.prefix_len], list(video.timestamps), t1 - t0, time.perf_counter() - t1


def run(args):
    import mlx.core as mx
    from . import mlx_model as mm
    from .inputs import Gemma4Inputs

    storage.require_root()
    awake = storage.keep_awake()
    fmt = dict(FORMAT, model=args.model)
    _check_meta(args.tag, fmt)
    sets = args.sets.split(",")
    vids = _videos(sets)
    todo = [v for v in vids if not os.path.exists(feature_path(args.tag, v))]
    cached = len(vids) - len(todo)
    if args.limit_videos:
        todo = todo[:args.limit_videos]
    print(f"{args.tag}: {len(vids)} videos in {sets}, {cached} cached, {len(todo)} to encode "
          f"-> {cache_dir(args.tag)}", flush=True)
    if not todo:
        return
    enc = Gemma4Inputs(CONFIG)
    model = mm.load_model(local_model(args.model))
    decode_s, vision_s, sizes, t_start = [], [], [], time.time()
    for i, vid in enumerate(todo):
        storage.check_alive()
        feats, prefix, ts, d_s, v_s = _encode_video(model, enc, storage.path(vids[vid]), fmt)
        dst = feature_path(args.tag, vid)
        tmp = dst[:-len(".safetensors")] + ".tmp.safetensors"
        mx.save_safetensors(tmp, {"features": feats, "prefix_ids": mx.array(prefix, dtype=mx.int32),
                                  "timestamps": mx.array(ts, dtype=mx.float32)},
                            metadata={"video_id": vid, **{k: str(v) for k, v in fmt.items()}})
        os.replace(tmp, dst)
        decode_s.append(d_s)
        vision_s.append(v_s)
        sizes.append(os.path.getsize(dst))
        if i == 0:  # stop early rather than fill the Mac's disk
            need = sizes[0] * len(todo)
            free = shutil.disk_usage(cache_dir(args.tag)).free
            print(f"features {tuple(feats.shape)} {feats.dtype}, {sizes[0] / 1e6:.1f} MB/video, "
                  f"~{need / 1e9:.1f} GB for {len(todo)} videos ({free / 1e9:.0f} GB free)", flush=True)
            if need > free - SPARE_BYTES:
                raise SystemExit(f"not enough space: need ~{need / 1e9:.1f} GB plus {SPARE_BYTES / 1e9:.0f} GB "
                                 f"spare, {free / 1e9:.1f} GB free. Free space (e.g. python -m "
                                 "video_train.models release d3-mini) and rerun; it resumes.")
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.time() - t_start
            print(f"[{i + 1}/{len(todo)}] vision {np.mean(vision_s[-25:]):.2f}s, decode {np.mean(decode_s[-25:]):.2f}s "
                  f"per video, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    res = {"videos_encoded": len(todo), "videos_total": len(vids), "mb_per_video": float(np.mean(sizes) / 1e6),
           "gb_written": float(sum(sizes) / 1e9), "vision_s": float(np.mean(vision_s)),
           "decode_s": float(np.mean(decode_s))}
    log_run(args.step, "cache", dict(fmt, tag=args.tag), {"sets": sets}, None, res)
    print(json.dumps(res))


def check(args):
    """Cached features + prefix from the file + suffix from the prompt alone, against fresh encoding."""
    import mlx.core as mx
    from . import mlx_model as mm
    from .evaluate import load_results, load_set, prompt_of
    from .inputs import Gemma4Inputs

    storage.require_root()
    fmt = dict(FORMAT, model=args.model)
    ref = load_results("dev", REFERENCE)
    cfg = next(iter(ref.values()))
    if any(cfg.get(k) != fmt[k] for k in ("model", "frames", "frames_as", "soft_tokens")):
        raise SystemExit(f"{REFERENCE} rows were not made in format {fmt}")
    by_video = {}
    for q in load_set("dev"):
        if (q["video_id"], q["qid"]) in ref and os.path.exists(feature_path(args.tag, q["video_id"])):
            by_video.setdefault(q["video_id"], []).append(q)
    vids = list(by_video)[:args.videos]
    if not vids:
        raise SystemExit(f"no cached dev videos under {cache_dir(args.tag)}; run `run --sets dev` first")
    enc = Gemma4Inputs(CONFIG)
    model = mm.load_model(local_model(args.model))
    same, diffs, feat_diff, suffix_bad, n = 0, [], 0.0, 0, 0
    for vid in vids:
        qs = by_video[vid]
        feats, prefix, _, = load(args.tag, vid)
        fresh, fresh_prefix, _, _, _ = _encode_video(model, enc, storage.path(qs[0]["video"]), fmt)
        feat_diff = max(feat_diff, float(mx.max(mx.abs(feats.astype(mx.float32) - fresh.astype(mx.float32)))))
        if prefix != fresh_prefix:
            raise SystemExit(f"{vid}: cached prefix ids differ from a fresh encoding")
        suffixes = [suffix_ids(enc, prompt_of(q), fmt["frames_as"], fmt["soft_tokens"]) for q in qs]
        # The suffix built without the video must equal the tail of the real encoding.
        from openjev.video import load_video
        with storage.drive_guard():
            video = load_video(storage.path(qs[0]["video"]), n_frames=fmt["frames"])
        for q, s in zip(qs, suffixes):
            e = enc.encode(prompt_of(q), video.frames, fmt["frames_as"], fmt["soft_tokens"], video.timestamps)
            suffix_bad += e.input_ids[e.prefix_len:] != s
        n_opts = {len(q["options"]) for q in qs}
        if len(n_opts) != 1:
            raise SystemExit(f"{vid}: questions with different option counts")
        logits = mm.letter_logits_grouped(model, prefix, feats, fmt["frames_as"], suffixes,
                                          enc.letter_ids(n_opts.pop()))
        probs = np.array(mx.softmax(logits, axis=-1)).astype(float)
        for q, p in zip(qs, probs):
            r = np.array(ref[(vid, q["qid"])]["probs"])
            same += int(p.argmax() == r.argmax())
            diffs.append(float(np.abs(p - r).max()))
            n += 1
    res = {"videos": len(vids), "questions": n, "agree_with_" + REFERENCE: same / n,
           "max_abs_prob_diff": max(diffs), "mean_abs_prob_diff": float(np.mean(diffs)),
           "max_abs_feature_diff_vs_fresh": feat_diff, "suffix_mismatches": suffix_bad}
    print(json.dumps(res, indent=1))
    log_run(args.step, "check", dict(fmt, tag=args.tag, what="cached features"), {"set": "dev"}, None, res)
    if res["agree_with_" + REFERENCE] < args.min_agree or suffix_bad:
        sys.exit(2)


def status(args):
    d = cache_dir(args.tag)
    files = [f for f in os.listdir(d) if f.endswith(".safetensors") and ".tmp." not in f] if os.path.isdir(d) else []
    try:
        storage.require_root()
        total = {s: len(_videos([s])) for s in SETS}
        have = {f[:-len(".safetensors")] for f in files}
        per_set = {s: f"{len(have & set(_videos([s])))}/{n}" for s, n in total.items()}
    except storage.DriveMissing:
        per_set = "(external drive not available)"
    size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    print(f"{d}: {len(files)} videos, {size / 1e9:.1f} GB; per set: {per_set}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sets", default="train,dev", help=f"comma-separated, from {list(SETS)}")
    r.add_argument("--limit-videos", type=int, default=0, help="smoke test")
    c = sub.add_parser("check")
    c.add_argument("--videos", type=int, default=20, help="dev videos to compare")
    c.add_argument("--min-agree", type=float, default=0.98, help="exit with status 2 below this")
    s = sub.add_parser("status")
    for p in (r, c, s):
        p.add_argument("--tag", default="video140")
        p.add_argument("--model", default=FORMAT["model"])
        p.add_argument("--step", default="step6")
    args = ap.parse_args(argv)
    try:
        {"run": run, "check": check, "status": status}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
