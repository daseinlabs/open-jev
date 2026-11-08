"""Step 9 v2: the encoder probe with a 4x4 grid of regions per frame instead of one vector.

    python -m video_train.probe_grid run                       # encode, train, report (resumable)
    python -m video_train.probe_grid encode|train|report       # one stage (--limit-videos N: smoke test)

v1 (``video_train.probe``) gave each frame one vector; for Gemma that averaged 120-136 tokens
that each describe a different region, and the probe could not read Gemma's features at all.
v2 keeps 16 regions per frame for every encoder, made by one operation for all of them: average
the encoder's own patch tokens over each cell of a 4x4 grid (``grid_pool``).

- Gemma: its token grid from the step 6 cache. Tokens are stored row by row, each covering 3x3
  patches; the grid follows the video's aspect ratio (8x15 for 16:9, 10x13 for 4:3), read from
  the patch positions Gemma's own input code gives a frame of that size (``gemma_grid``).
- SigLIP 2: 14x14 patches. DINOv2: 16x16 patches, CLS dropped. V-JEPA 2: 16x16 patches per
  two-frame slot (8 slots). Frames are squashed to the square input, so a cell covers the same
  part of the picture for every encoder.
A video is 256 vectors (V-JEPA 2: 128), each with its time and cell.

Probe: video vectors (+ cell and time embeddings, + a per-encoder embedding) and the question
vector go through two transformer layers, so regions are read in light of the question; each
option attends over the result and an MLP scores it. Text features are v1's (base Gemma, no
adapter). Arms, seeds, held-out split and the decision rule are v1's (``probe.verdict``). New
control: ``gemma`` scored on dev with each question given another dev video's features.

Features: ``<LOCAL_CACHE>/probe_grid/<source>/<video_id>.npz`` (~1.3 MB per video for Gemma,
float16). The probe holds every video's features in memory (~4.5 GB for the largest arm):
close other apps. Outputs: ``results/step9/v2/``; report ``results/step9/v2/summary.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import probe, storage
from .history import log_run

STEP = "step9"
GRID = 4
ARMS, CANDIDATES, SEEDS = probe.ARMS, probe.CANDIDATES, probe.SEEDS
HP = {"width": 256, "heads": 4, "layers": 2, "dropout": 0.2, "lr": 3e-4, "weight_decay": 0.05, "batch": 64,
      "epochs": 40, "patience": 5, "holdout": probe.HP["holdout"]}
CONTROL = "gemma_shuffled"  # gemma's probes, each dev question given another dev video's features


def feature_file(source, video_id):
    return os.path.join(storage.LOCAL_CACHE, "probe_grid", source, f"{video_id}.npz")


def out_dir(smoke=False):
    return storage.path("results", STEP, "v2_smoke" if smoke else "v2")


def probe_file(arm, seed, smoke=False):
    return os.path.join(out_dir(smoke), "probe", f"{arm}_seed{seed}.json")


# ------------------------------------------------------------------ features
def grid_pool(tokens, rows, cols, grid=GRID):
    """(rows x cols, d) row-major patch tokens -> (grid x grid, d): the mean of each cell.

    Cell edges follow torch's adaptive average pooling: cell i spans [floor(i R / g), ceil((i+1) R / g)).
    """
    t = np.asarray(tokens, dtype=np.float32).reshape(rows, cols, -1)
    out = np.empty((grid, grid, t.shape[-1]), dtype=np.float32)
    for i in range(grid):
        r0, r1 = (i * rows) // grid, -((-(i + 1) * rows) // grid)
        for j in range(grid):
            c0, c1 = (j * cols) // grid, -((-(j + 1) * cols) // grid)
            out[i, j] = t[r0:r1, c0:c1].mean(axis=(0, 1))
    return out.reshape(grid * grid, -1)


def gemma_grid(enc, width, height):
    """(rows, cols) of Gemma's soft-token grid for a frame of this size, in the step 6 format."""
    from PIL import Image
    from .cache_features import FORMAT
    e = enc.encode("", [Image.new("RGB", (width, height))], FORMAT["frames_as"], FORMAT["soft_tokens"], [0.0])
    pos = e.position_ids[0]
    real = pos[(pos >= 0).all(-1)]
    k = 3  # Gemma 4 pools 3x3 patches into one soft token
    return int(real[:, 1].max() + 1) // k, int(real[:, 0].max() + 1) // k


def gemma_cells(features, frames, rows, cols):
    """Cached (1, frames x rows x cols, hidden) Gemma features -> (frames x 16, hidden)."""
    f = np.asarray(features, dtype=np.float32).reshape(-1, np.shape(features)[-1])
    if f.shape[0] != frames * rows * cols:
        raise ValueError(f"{f.shape[0]} tokens, expected {frames} frames x {rows} x {cols}")
    return np.concatenate([grid_pool(x, rows, cols) for x in f.reshape(frames, rows * cols, -1)])


class GridEncoder(probe.FrameEncoder):
    """v1's frozen encoders, returning a 4x4 grid of patch averages per frame (or two-frame slot)."""

    def __call__(self, frames, times):
        torch = self.torch
        with torch.no_grad():
            if self.kind == "video":
                x = self.proc(videos=[np.stack([np.asarray(f) for f in frames])], return_tensors="pt",
                              **self.video_kw)
                h = self.model(pixel_values_videos=x["pixel_values_videos"].to(self.device),
                               skip_predictor=True).last_hidden_state[0]
                slots = len(frames) // self.model.config.tubelet_size
                t = np.asarray(times, dtype=np.float32).reshape(slots, -1).mean(axis=1)
                per = h.reshape(slots, -1, h.shape[-1])  # patch embedding order: time, then rows, then columns
            else:
                x = self.proc(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
                per = self.model(pixel_values=x).last_hidden_state
                if self.cls:
                    per = per[:, 1:]
                t = np.asarray(times, dtype=np.float32)
            per = per.float().cpu().numpy()
        side = int(round(per.shape[1] ** 0.5))
        if side * side != per.shape[1]:
            raise ValueError(f"{self.name}: {per.shape[1]} patches per frame is not a square grid")
        cells = np.concatenate([grid_pool(p, side, side) for p in per])
        return cells, np.repeat(t, GRID * GRID)


def encode(args):
    """Decode each video once; write Gemma's cells (from the step 6 cache) and each encoder's cells."""
    import mlx.core as mx
    from openjev.video import load_video
    from .cache_features import FORMAT, feature_path
    from .inputs import Gemma4Inputs
    from .models import CONFIG
    sources = ["gemma", *args.encoders.split(",")]
    keep, todo = probe._first_videos(args.limit_videos), []
    for s in probe.SETS:
        by = probe._by_video(probe._questions(s))
        todo += [(v, by[v][0]["video"]) for v in keep[s] if any(not os.path.exists(feature_file(n, v)) for n in sources)]
    print(f"encode {sources}: {len(todo)} videos to do", flush=True)
    if not todo:
        return
    awake = storage.keep_awake()
    enc = Gemma4Inputs(CONFIG)
    grids = {}
    encs = {n: GridEncoder(n) for n in sources[1:]}
    secs, decode_s, t_start = {n: [] for n in sources}, [], time.time()
    for i, (vid, path) in enumerate(todo):
        storage.check_alive()
        t0 = time.perf_counter()
        with storage.drive_guard():
            video = load_video(storage.path(path), n_frames=FORMAT["frames"])
        decode_s.append(time.perf_counter() - t0)
        if not os.path.exists(feature_file("gemma", vid)):
            t0 = time.perf_counter()
            size = (video.width, video.height)
            if size not in grids:
                grids[size] = gemma_grid(enc, *size)
            a = mx.load(feature_path(probe.GEMMA_CACHE, vid))
            cells = gemma_cells(np.array(a["features"].astype(mx.float32)), FORMAT["frames"], *grids[size])
            times = np.repeat(np.array(a["timestamps"], dtype=np.float32), GRID * GRID)
            if not np.allclose(np.array(a["timestamps"]), video.timestamps, atol=1e-3):
                raise SystemExit(f"{vid}: cached frame times differ from a fresh decode")
            probe._save_npz(feature_file("gemma", vid), tokens=cells.astype(np.float16), times=times)
            secs["gemma"].append(time.perf_counter() - t0)
        for n, e in encs.items():
            if os.path.exists(feature_file(n, vid)):
                continue
            t0 = time.perf_counter()
            cells, times = e(video.frames, video.timestamps)
            secs[n].append(time.perf_counter() - t0)
            probe._save_npz(feature_file(n, vid), tokens=cells.astype(np.float16), times=times)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            el = time.time() - t_start
            per = ", ".join(f"{n} {np.mean(s[-50:]):.2f}s" for n, s in secs.items() if s)
            print(f"[{i + 1}/{len(todo)}] {per}, decode {np.mean(decode_s[-50:]):.2f}s per video, "
                  f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    log_run(STEP, "smoke" if args.limit_videos else "cache", {"what": "probe grid features", "grid": GRID,
                                                               "sources": sources},
            {"sets": list(probe.SETS)}, None, {"gemma_grids": {f"{w}x{h}": g for (w, h), g in grids.items()},
                                               "decode_s_per_video": float(np.mean(decode_s))})


# ------------------------------------------------------------------ probe
def time_features(t):
    """Times in [0, 1], (..., n) -> (..., n, 16) sines and cosines at 8 frequencies."""
    import torch
    f = 2 ** torch.arange(8, dtype=t.dtype, device=t.device) * torch.pi
    return torch.cat([torch.sin(t[..., None] * f), torch.cos(t[..., None] * f)], dim=-1)


def build_probe(dims, hp=HP):
    """Question + video cells through a small transformer; each option attends over it and is scored."""
    import torch
    from torch import nn
    w, text_dim = hp["width"], dims.get("_text", 2560)

    class GridProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.sources = [s for s in dims if s != "_text"]
            self.last = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, w))
            self.opt = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, w))
            self.vis = nn.ModuleDict({s: nn.Sequential(nn.LayerNorm(dims[s]), nn.Dropout(hp["dropout"]),
                                                       nn.Linear(dims[s], w)) for s in self.sources})
            self.kind = nn.ParameterDict({s: nn.Parameter(torch.zeros(w)) for s in self.sources})
            self.cell = nn.Parameter(torch.zeros(GRID * GRID, w))
            self.time = nn.Linear(16, w)
            layer = nn.TransformerEncoderLayer(w, hp["heads"], 2 * w, hp["dropout"], batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, hp["layers"], enable_nested_tensor=False)
            self.attn = nn.MultiheadAttention(w, hp["heads"], dropout=hp["dropout"], batch_first=True)
            self.drop = nn.Dropout(hp["dropout"])
            self.head = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Dropout(hp["dropout"]), nn.Linear(w, 1))

        def forward(self, last, options, video=None):
            """last (B, D), options (B, K, D), video {source: (cells (B, n, d), times (B, n) in [0, 1])} -> (B, K)."""
            qv = self.last(last)[:, None]                                           # (B, 1, w)
            seq = [qv]
            for s in self.sources:
                cells, t = video[s]
                n = cells.shape[1]
                pos = self.cell.repeat(n // (GRID * GRID), 1)[None]                 # cells are frame-major
                seq.append(self.vis[s](cells) + self.kind[s] + pos + self.time(time_features(t)))
            z = self.mix(torch.cat(seq, dim=1))
            q = self.drop(nn.functional.gelu(qv + self.opt(options)))              # (B, K, w)
            a, _ = self.attn(q, z, z, need_weights=False)
            return self.head(torch.cat([q, a, q * a], dim=-1)).squeeze(-1)

    return GridProbe()


def load_split(set_name, sources, limit_videos=0):
    """Like v1's, from the grid features; float16 in memory, converted per batch."""
    rows, by_vid = [], probe._by_video(probe._questions(set_name))
    vids = list(by_vid)[:limit_videos or None]
    files = lambda v: [probe.feature_file("text", v), *(feature_file(s, v) for s in sources)]
    if not limit_videos:
        missing = [f for v in vids for f in files(v) if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"{set_name}: {len(missing)} feature files missing, e.g. {missing[0]}; run `encode`")
    feats = {s: {} for s in sources}
    for vid in vids:
        if limit_videos and not all(os.path.exists(f) for f in files(vid)):
            continue
        tf = dict(np.load(probe.feature_file("text", vid)))  # read each member once
        idx = {int(q): i for i, q in enumerate(tf["qid"])}
        for s in sources:
            f = np.load(feature_file(s, vid))
            feats[s][vid] = (f["tokens"], f["times"])
        for q in by_vid[vid]:
            i = idx[q["qid"]]
            rows.append({"q": q, "last": tf["last"][i], "options": tf["options"][i]})
    return rows, feats


def _batch(rows, feats, sources, idx, device, swap=None):
    """Tensors for rows ``idx``; ``swap`` maps a video id to the one whose features stand in for it."""
    import torch
    sel = [rows[i] for i in idx]
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
    last, opts = f32(np.stack([r["last"] for r in sel])), f32(np.stack([r["options"] for r in sel]))
    video = {}
    for s in sources:
        src = [(swap or {}).get(r["q"]["video_id"], r["q"]["video_id"]) for r in sel]
        tok = [feats[s][v] for v in src]
        if len({t[0].shape[0] for t in tok}) != 1:
            raise ValueError(f"{s}: videos with different numbers of cells")
        # a stand-in video's times are scaled by its own duration
        dur = np.array([max(DURATIONS.get(v, r["q"]["duration_s"]), 1e-3) for v, r in zip(src, sel)],
                       dtype=np.float32)[:, None]
        video[s] = (f32(np.stack([t[0] for t in tok])), f32(np.clip(np.stack([t[1] for t in tok]) / dur, 0, 1)))
    y = torch.tensor([r["q"]["answer_id"] for r in sel], device=device)
    return last, opts, video, y


DURATIONS = {}  # video_id -> duration_s, filled by train()


def train_arm(arm, seed, train_rows, dev_rows, feats, hp=HP, log=print, swap=None):
    """Train one probe; return dev probabilities (and, with ``swap``, the shuffled-video ones) and the curve."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sources = probe.sources_of(arm)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dims = {"_text": train_rows[0]["last"].shape[0], **{s: next(iter(feats[s].values()))[0].shape[1] for s in sources}}
    model = build_probe(dims, hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    held = probe.holdout_videos({r["q"]["video_id"] for r in train_rows}, hp["holdout"])
    fit = [i for i, r in enumerate(train_rows) if r["q"]["video_id"] not in held]
    val = [i for i, r in enumerate(train_rows) if r["q"]["video_id"] in held] or fit  # empty only in a smoke test

    def run(rows, idx, swap_map=None):
        model.eval()
        tot, probs = 0.0, []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                last, opts, video, y = _batch(rows, feats, sources, idx[s:s + 128], device, swap_map)
                lg = model(last, opts, video)
                tot += float(torch.nn.functional.cross_entropy(lg, y, reduction="sum"))
                probs.append(torch.softmax(lg, -1).float().cpu().numpy())
        return tot / max(len(idx), 1), np.concatenate(probs)

    best, best_state, bad, curve = float("inf"), None, 0, []
    for ep in range(hp["epochs"]):
        model.train()
        order = rng.permutation(fit)
        for s in range(0, len(order), hp["batch"]):
            last, opts, video, y = _batch(train_rows, feats, sources, order[s:s + hp["batch"]], device)
            loss = torch.nn.functional.cross_entropy(model(last, opts, video), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v, _ = run(train_rows, val)
        curve.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    every = list(range(len(dev_rows)))
    _, probs = run(dev_rows, every)
    shuffled = run(dev_rows, every, swap)[1] if swap else None
    log(f"{arm} seed {seed}: {len(curve)} epochs, best held-out NLL {best:.3f} at epoch {int(np.argmin(curve)) + 1}",
        flush=True)
    return probs, shuffled, curve


def shuffle_map(video_ids, seed=0):
    """Each dev video -> a different dev video (a fixed derangement)."""
    v = sorted(video_ids)
    perm = np.random.default_rng(seed).permutation(len(v))
    return {v[i]: v[perm[(k + 1) % len(v)]] for k, i in enumerate(perm)}


def train(args):
    storage.require_root()
    arms = args.arms.split(",") if args.arms else ARMS
    smoke = bool(args.limit_videos)
    hp = dict(HP, epochs=2) if smoke else HP
    for arm in arms:
        todo = [s for s in SEEDS if not os.path.exists(probe_file(arm, s, smoke))]
        if not todo:
            continue
        sources = probe.sources_of(arm)
        tr, ftr = load_split("train", sources, args.limit_videos)
        dv, fdv = load_split("dev", sources, args.limit_videos)
        feats = {s: {**ftr[s], **fdv[s]} for s in sources}
        DURATIONS.update({r["q"]["video_id"]: r["q"]["duration_s"] for r in tr + dv})
        swap = shuffle_map({r["q"]["video_id"] for r in dv}) if arm == "gemma" else None
        print(f"{arm}: {len(tr)} train / {len(dv)} dev questions", flush=True)
        for seed in todo:
            t0 = time.time()
            probs, shuffled, curve = train_arm(arm, seed, tr, dv, feats, hp, swap=swap)
            row = lambda r, p: {"video_id": r["q"]["video_id"], "qid": r["q"]["qid"], "probs": p.tolist()}
            out = {"arm": arm, "seed": seed, "hp": hp, "curve": curve, "train_s": time.time() - t0,
                   "rows": [row(r, p) for r, p in zip(dv, probs)]}
            if shuffled is not None:
                out["shuffled_rows"] = [row(r, p) for r, p in zip(dv, shuffled)]
            os.makedirs(os.path.dirname(probe_file(arm, seed, smoke)), exist_ok=True)
            with storage.drive_guard(), open(probe_file(arm, seed, smoke), "w") as f:
                json.dump(out, f)
            if not smoke:
                acc = float(np.mean([int(np.argmax(p)) == r["q"]["answer_id"] for r, p in zip(dv, probs)]))
                log_run(STEP, "train", {"what": "probe v2", "arm": arm, **hp}, {"train": len(tr), "dev": len(dv)},
                        seed, {"dev_accuracy": acc, "epochs": len(curve), "best_holdout_nll": min(curve)})


# ------------------------------------------------------------------ report
def arm_rows(arm, questions, smoke=False, key="rows"):
    """({(video_id, qid): eval-style row} with seed-mean probabilities, per-seed accuracies), or (None, None)."""
    runs = []
    for s in SEEDS:
        p = probe_file(arm, s, smoke)
        if not os.path.exists(p):
            return None, None
        with storage.drive_guard():
            runs.append({(r["video_id"], r["qid"]): r["probs"] for r in json.load(open(p))[key]})
    seed_acc = [float(np.mean([int(np.argmax(p)) == questions[k]["answer_id"] for k, p in r.items()])) for r in runs]
    out = {}
    for k in runs[0]:
        q = questions[k]
        probs = np.mean([r[k] for r in runs], axis=0)
        out[k] = {"probs": probs.tolist(), "answer_id": q["answer_id"], "correct": int(np.argmax(probs)) == q["answer_id"],
                  "area": q["area"], "reasoning": q["reasoning"], "latency_s": 0.0, "decode_s": 0.0}
    return out, seed_acc


def report(args):
    from .evaluate import metrics
    from .teacher import paired
    storage.require_root()
    smoke = args.smoke
    questions = {(q["video_id"], q["qid"]): q for q in probe._questions("dev")}
    got = {a: arm_rows(a, questions, smoke) for a in ARMS}
    rows, seeds = {a: g[0] for a, g in got.items()}, {a: g[1] for a, g in got.items()}
    rows[CONTROL], seeds[CONTROL] = arm_rows("gemma", questions, smoke, "shuffled_rows")
    if all(rows[a] is None for a in ARMS):
        raise SystemExit("no complete step 9 v2 probe results yet (python -m video_train.probe_grid run)")
    state, win, tests = probe.verdict({a: rows[a] for a in ARMS})
    lines = [f"# {STEP} v2: probing vision encoders, 4x4 regions per frame", "",
             "Same probe and text features in every arm; only the video features change. Every encoder gives a 4x4 grid "
             "of patch averages per frame. Trained on Perception Test train (3 seeds; rows are seed-mean probabilities), "
             "scored once on dev. Probe accuracy measures what each encoder's features make available, not what a Gemma "
             "model with that encoder would score.", "",
             "| arm | dev accuracy | seeds (min-max) | NLL | net vs text (p) | net vs gemma (p) |", "|---|---|---|---|---|---|"]
    order = ["text", "gemma", CONTROL, *ARMS[2:]]
    for a in order:
        r = rows[a]
        if r is None:
            lines.append(f"| {a} | not run | | | | |")
            continue
        m = metrics(list(r.values()))

        def net(ref):
            if rows.get(ref) is None or ref == a:
                return "-"
            g, lost, p = paired(rows[ref], r)
            return f"{g - lost:+d} ({p:.2g})"
        lines.append(f"| {a} | {m['accuracy']:.1%} ± {m['ci95']:.1%} | {min(seeds[a]):.1%}-{max(seeds[a]):.1%} | "
                     f"{m['nll']:.3f} | {net('text')} | {net('gemma')} |")
    lines += ["", f"`{CONTROL}`: the `gemma` probes, each dev question given another dev video's features. Near `text` "
                  "means the probe's gain from Gemma's features comes from the video itself."]
    done = [a for a in order if rows[a] is not None]
    lines += ["", "| area | n | " + " | ".join(done) + " |", "|---|---|" + "---|" * len(done)]
    for ar in sorted({q["area"] for q in questions.values()}):
        n = sum(v["area"] == ar for v in rows[done[0]].values())
        if n:
            accs = [np.mean([v["correct"] for v in rows[a].values() if v["area"] == ar]) for a in done]
            lines.append(f"| {ar} | {n} | " + " | ".join(f"{x:.1%}" for x in accs) + " |")
    lines += ["", "## Verdict", ""]
    lines += [f"- {name}: gains {g}, loses {lost} (net {g - lost:+d}, p={p:.2g})" for name, (g, lost, p) in tests.items()]
    bar = probe.P_SANITY / len(CANDIDATES)
    lines += ["", {
        "incomplete": "**Incomplete:** some arms have not run yet.",
        "inconclusive": "**Inconclusive again:** with 4x4 regions the probe still does not beat text-only with Gemma's "
                        "own features. As decided before this run, the probe route is closed: step 10 is decided "
                        "without it.",
        "fail": f"**No encoder adds to Gemma's own features** (bar: p < {bar:.3f}).",
        "pass": f"**{win} adds to Gemma's own features** (bar: p < {bar:.3f}).",
    }[state]]
    os.makedirs(out_dir(smoke), exist_ok=True)
    dst = os.path.join(out_dir(smoke), "summary.md")
    with storage.drive_guard(), open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-len(tests) - 3:]))
    print(f"\nwrote {dst}")
    if not smoke:
        log_run(STEP, "eval", {"what": "probe v2 report"}, {"set": "dev"}, None, {"state": state, "winner": win})


def run(args):
    storage.require_root()
    missing = [n for n in CANDIDATES if not probe._on_drive_or_mac(n)]
    if missing:
        raise SystemExit(f"encoders not downloaded: {missing}; run `python -m video_train.probe fetch` first")
    probe.text(args)  # v1's text features; already on disk unless the cache was cleared
    args.encoders = ",".join(CANDIDATES)
    encode(args)
    args.arms = None
    train(args)
    args.smoke = bool(args.limit_videos)
    report(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"encode": encode, "train": train, "report": report, "run": run}
    p = {c: sub.add_parser(c) for c in cmds}
    p["encode"].add_argument("--encoders", default=",".join(CANDIDATES))
    for c in ("encode", "train", "run"):
        p[c].add_argument("--limit-videos", type=int, default=0, help="smoke test: first N videos per set")
    p["train"].add_argument("--arms", default=None, help=f"comma-separated, from {ARMS}")
    p["report"].add_argument("--smoke", action="store_true", help="report the --limit-videos probe outputs")
    args = ap.parse_args(argv)
    try:
        storage.require_root()
        cmds[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
