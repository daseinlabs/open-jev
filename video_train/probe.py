"""Step 9: probe other vision encoders against a text-only baseline, before any step 10 work.

    python -m video_train.probe fetch                    # download dinov2 + vjepa2 to the drive (~1.6 GB, once)
    python -m video_train.probe run                      # every stage below in order, resumable
    python -m video_train.probe text|pool|encode|train   # one stage (--limit-videos N: smoke test)
    python -m video_train.probe report                   # results/step9/summary.md and the verdict

Question: does another frozen encoder see something in Perception Test videos that Gemma 4's own
encoder does not? Answering it with Gemma itself would mean a step 10 run per encoder; a small
probe trained on frozen features answers it for all of them in a few hours.

Every arm is the same probe on the same text features; only the video features change:
- ``text``: no video. The floor: what the question and options alone give.
- ``gemma``: Gemma 4's own encoder + projector, from step 6's cache (no re-encoding).
- ``siglip2`` (SigLIP 2 base, 86M), ``dinov2`` (DINOv2 base, 86M, self-supervised), ``vjepa2``
  (V-JEPA 2 ViT-L, 300M, a video encoder: it sees motion across frames, the others see stills).
- ``gemma+<x>``: Gemma's features and x's together. This is step 10's "second encoder" question.

Inputs. Text: base Gemma 4 E4B (no adapter: it was trained with video) reads each question as
scored, without the video prefix; features are the final hidden state at the last prompt token
and the mean final hidden state over each option's tokens. Video: the same 16 uniform frames as
the step 6 cache, one vector per frame (Gemma: mean of its 140 tokens; SigLIP 2 / DINOv2: pooled
output + mean of patch tokens; V-JEPA 2: mean over space for each of its 8 two-frame slots).
Frames are resized to the encoder's square input without cropping. One vector per frame keeps
the probe small; it under-measures fine spatial detail, equally for every arm.

Probe (fixed before any result; nothing is tuned on dev): each option's text becomes a query
that attends over the video vectors (one self-attention layer over time first), then an MLP
scores the option; softmax over options, cross-entropy. Width 256, dropout 0.2, AdamW lr 1e-3,
weight decay 0.05, batch 64, up to 40 epochs, stopped on the loss of 10% of train videos held
out (by video). Trained on Perception Test train, 3 seeds per arm; dev rows are the seed-mean
probabilities. Dev is scored once per arm.

Decision (step 1's paired rule, McNemar, more gained than lost):
1. ``gemma`` must beat ``text`` (p < 0.05). If not, the probe cannot read video features at
   all and step 9 says nothing: inconclusive.
2. An encoder x passes if ``gemma+x`` beats ``gemma`` with p < 0.05 / (number of candidates):
   three candidates are tested, so each needs stronger evidence. The passing x with the largest
   net gain goes to step 10. If none passes, Gemma's own features already hold what these
   encoders see, and step 10 is skipped (straight to step 11).
``x`` alone against ``gemma`` and the per-area table are reported, not gated. NExT-QA is not
used: the probe is a diagnostic, not a model; step 10's model gets the NExT-QA check. The test
split is never read.

Files: features in ``<LOCAL_CACHE>/probe/<source>/<video_id>.npz`` (rebuildable, ~0.1 MB per
video and source); probe outputs in ``results/step9/probe/<arm>_seed<k>.json``; report
``results/step9/summary.md``. Every stage skips work already on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import storage
from .history import log_run
from .perception_data import load_manifest

STEP = "step9"
GEMMA_CACHE = "video140"  # step 6's features
ENCODERS = {  # name -> (Hugging Face repo, folder under models/, kind, square input side)
    "siglip2": ("google/siglip2-base-patch16-224", "siglip2-base-patch16-224", "image", 224),
    "dinov2": ("facebook/dinov2-base", "dinov2-base", "image", 224),  # its config says 518; 224 matches SigLIP 2
    "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", "vjepa2-vitl-fpc64-256", "video", 256),
}
CANDIDATES = list(ENCODERS)
ARMS = ["text", "gemma", *CANDIDATES, *(f"gemma+{x}" for x in CANDIDATES)]
SETS = ("train", "dev")
SEEDS = (0, 1, 2)
P_SANITY = 0.05
HP = {"width": 256, "heads": 4, "dropout": 0.2, "lr": 1e-3, "weight_decay": 0.05, "batch": 64,
      "epochs": 40, "patience": 5, "holdout": 0.1}


def feature_dir(source):
    return os.path.join(storage.LOCAL_CACHE, "probe", source)


def feature_file(source, video_id):
    return os.path.join(feature_dir(source), f"{video_id}.npz")


def out_dir():
    return storage.path("results", STEP)


def _save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _questions(set_name):
    return load_manifest(set_name)  # evaluate.load_set knows only the eval sets


def _first_videos(limit):
    """{set: [video_id, ...]}: every video, or the first ``limit`` of each set (smoke test)."""
    return {s: list(_by_video(_questions(s)))[:limit or None] for s in SETS}


def _by_video(rows):
    out = {}
    for r in rows:
        out.setdefault(r["video_id"], []).append(r)
    return out


def sources_of(arm):
    return [] if arm == "text" else arm.split("+")


# ------------------------------------------------------------------ text features
def option_spans(prompt, n_options, offsets):
    """Token indices (into ``offsets``) of each option's rendered text, in letter order."""
    from .inputs import LETTERS
    spans, pos = [], prompt.index("Options:\n")
    for letter in LETTERS[:n_options]:
        tag = f"- {letter}: "
        start = prompt.index(tag, pos) + len(tag)
        end = prompt.index("\n", start)
        idx = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not idx:
            raise ValueError(f"option {letter} has no tokens in {prompt!r}")
        spans.append(idx)
        pos = end
    return spans


def text(args):
    """Base Gemma on each question without the video: last-token and per-option hidden states."""
    import mlx.core as mx
    from . import mlx_model as mm
    from .evaluate import prompt_of
    from .inputs import Gemma4Inputs
    from .models import CONFIG, local_model
    from .cache_features import FORMAT

    keep, todo = _first_videos(args.limit_videos), []
    for s in SETS:
        by = _by_video(_questions(s))
        todo += [(v, by[v]) for v in keep[s] if not os.path.exists(feature_file("text", v))]
    print(f"text: {len(todo)} videos to do -> {feature_dir('text')}", flush=True)
    if not todo:
        return
    enc = Gemma4Inputs(CONFIG)
    model = mm.load_model(local_model(FORMAT["model"]))
    t0, n = time.time(), 0
    for i, (vid, qs) in enumerate(todo):
        last, opts, blind = [], [], []
        for q in qs:
            p = prompt_of(q)
            t = enc.tok(p, add_special_tokens=False, return_offsets_mapping=True)
            ids = [enc.tok.bos_token_id] + t["input_ids"]  # the scoring format, minus the video prefix
            h = mm._hidden(model, mx.array(ids)[None], None, FORMAT["frames_as"], keep=0)[0].astype(mx.float32)
            spans = option_spans(p, len(q["options"]), t["offset_mapping"])
            o = mx.stack([h[mx.array([j + 1 for j in sp])].mean(axis=0) for sp in spans])  # +1: BOS
            lg = mm._letter_readout(model, h[-1], enc.letter_ids(len(q["options"])))
            mx.eval(o, lg)
            last.append(np.array(h[-1]))
            opts.append(np.array(o))
            blind.append(np.array(mx.softmax(lg)))
        _save_npz(feature_file("text", vid), qid=np.array([q["qid"] for q in qs]),
                  last=np.stack(last).astype(np.float16), options=np.stack(opts).astype(np.float16),
                  blind=np.stack(blind).astype(np.float32))
        n += len(qs)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"[{i + 1}/{len(todo)}] {el / n:.3f}s/question, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min",
                  flush=True)


# ------------------------------------------------------------------ video features
def pool_gemma(features, frames):
    """(frames x tokens, hidden) or (1, frames x tokens, hidden) -> (frames, hidden) per-frame means."""
    f = np.asarray(features, dtype=np.float32).reshape(-1, np.shape(features)[-1])
    if f.shape[0] % frames:
        raise ValueError(f"{f.shape[0]} tokens do not split into {frames} frames")
    return f.reshape(frames, -1, f.shape[-1]).mean(axis=1)


def pool(args):
    """Step 6's cached Gemma features -> one vector per frame."""
    from .cache_features import FORMAT, feature_path
    todo = [v for vs in _first_videos(args.limit_videos).values() for v in vs
            if not os.path.exists(feature_file("gemma", v))]
    print(f"pool: {len(todo)} videos to do -> {feature_dir('gemma')}", flush=True)
    import mlx.core as mx
    for i, vid in enumerate(todo):
        src = feature_path(GEMMA_CACHE, vid)
        if not os.path.exists(src):
            raise SystemExit(f"{src} missing: rebuild step 6's cache (python -m video_train.cache_features run)")
        a = mx.load(src)
        tokens = pool_gemma(np.array(a["features"].astype(mx.float32)), FORMAT["frames"])
        _save_npz(feature_file("gemma", vid), tokens=tokens.astype(np.float16),
                  times=np.array(a["timestamps"], dtype=np.float32))
        if (i + 1) % 250 == 0:
            print(f"[{i + 1}/{len(todo)}]", flush=True)


class FrameEncoder:
    """A frozen torch encoder (MPS when available): 16 frames -> (n, dim) vectors and their times."""

    def __init__(self, name):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor
        from .models import local_model
        _, folder, self.kind, side = ENCODERS[name]
        self.torch, self.name = torch, name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        path = local_model(folder)
        m = AutoModel.from_pretrained(path, dtype=torch.float32)
        self.model = (m.vision_model if hasattr(m, "vision_model") else m).to(self.device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        size = {"height": side, "width": side}
        self.video_kw = {}
        if self.kind == "video":
            self.proc = AutoVideoProcessor.from_pretrained(path, do_center_crop=False, size=size)
            # V-JEPA 2's processor drops `size` given here (keeps shortest_edge 292, aspect kept);
            # it only squashes to the square input when `size` is passed at call time.
            self.video_kw = {"size": size}
        elif name == "dinov2":
            self.proc = AutoImageProcessor.from_pretrained(path, do_center_crop=False, size=size)
        else:  # SigLIP 2 already squashes to its square input
            self.proc = AutoImageProcessor.from_pretrained(path)
        self.cls = name == "dinov2"  # token 0 is CLS; SigLIP 2 has none

    def __call__(self, frames, times):
        torch = self.torch
        with torch.no_grad():
            if self.kind == "video":
                x = self.proc(videos=[np.stack([np.asarray(f) for f in frames])], return_tensors="pt", **self.video_kw)
                h = self.model(pixel_values_videos=x["pixel_values_videos"].to(self.device),
                               skip_predictor=True).last_hidden_state[0]
                slots = len(frames) // self.model.config.tubelet_size
                v = h.reshape(slots, -1, h.shape[-1]).mean(dim=1)
                t = np.asarray(times, dtype=np.float32).reshape(slots, -1).mean(axis=1)
            else:
                x = self.proc(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
                out = self.model(pixel_values=x)
                patches = out.last_hidden_state[:, 1:] if self.cls else out.last_hidden_state
                v = torch.cat([out.pooler_output, patches.mean(dim=1)], dim=-1)
                t = np.asarray(times, dtype=np.float32)
            if self.device == "mps":
                torch.mps.synchronize()
        return v.float().cpu().numpy(), t


def encode(args):
    """Decode each video's 16 frames once and run every requested encoder that lacks its file."""
    from openjev.video import load_video
    from .cache_features import FORMAT
    names = args.encoders.split(",")
    keep, todo = _first_videos(args.limit_videos), []
    for s in SETS:
        by = _by_video(_questions(s))
        todo += [(v, by[v][0]["video"]) for v in keep[s] if any(not os.path.exists(feature_file(n, v)) for n in names)]
    print(f"encode {names}: {len(todo)} videos to do", flush=True)
    if not todo:
        return
    awake = storage.keep_awake()
    encs = {n: FrameEncoder(n) for n in names}
    secs = {n: [] for n in names}
    decode_s, t_start = [], time.time()
    for i, (vid, path) in enumerate(todo):
        storage.check_alive()
        t0 = time.perf_counter()
        with storage.drive_guard():
            video = load_video(storage.path(path), n_frames=FORMAT["frames"])
        decode_s.append(time.perf_counter() - t0)
        for n, e in encs.items():
            if os.path.exists(feature_file(n, vid)):
                continue
            t0 = time.perf_counter()
            v, t = e(video.frames, video.timestamps)
            secs[n].append(time.perf_counter() - t0)
            _save_npz(feature_file(n, vid), tokens=v.astype(np.float16), times=t)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            el = time.time() - t_start
            per = ", ".join(f"{n} {np.mean(s[-50:]):.2f}s" for n, s in secs.items() if s)
            print(f"[{i + 1}/{len(todo)}] {per}, decode {np.mean(decode_s[-50:]):.2f}s per video, "
                  f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    res = {n: {"params_m": encs[n].params / 1e6, "encode_s_per_video": float(np.mean(s)) if s else None,
               "videos": len(s)} for n, s in secs.items()}
    res["decode_s_per_video"] = float(np.mean(decode_s))
    log_run(STEP, "smoke" if args.limit_videos else "cache", {"what": "probe features", "encoders": names, "frames": FORMAT["frames"]},
            {"sets": list(SETS)}, None, res)
    print(json.dumps(res))


def fetch(args):
    """Download the encoders not yet on the drive into ``<DATA_ROOT>/models/<folder>``."""
    from huggingface_hub import snapshot_download
    from .models import _has_weights, store_path
    storage.require_root()
    for name in args.encoders.split(","):
        repo, folder = ENCODERS[name][:2]
        dst = store_path(folder)
        if _has_weights(dst):
            print(f"{name}: already at {dst}")
            continue
        print(f"{name}: downloading {repo} -> {dst}", flush=True)
        with storage.drive_guard():
            snapshot_download(repo, local_dir=dst, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


# ------------------------------------------------------------------ probe
def build_probe(dims, hp=HP):
    """Option-scoring probe over the text features and, per source in ``dims``, video vectors."""
    import torch
    from torch import nn
    w, text_dim = hp["width"], dims.get("_text", 2560)

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.sources = [s for s in dims if s != "_text"]
            self.last = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, w))
            self.opt = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, w))
            self.vis = nn.ModuleDict({s: nn.Sequential(nn.LayerNorm(dims[s]), nn.Dropout(hp["dropout"]),
                                                       nn.Linear(dims[s], w)) for s in self.sources})
            self.kind = nn.ParameterDict({s: nn.Parameter(torch.zeros(w)) for s in self.sources})
            self.time = nn.Linear(16, w)
            if self.sources:
                self.temporal = nn.TransformerEncoderLayer(w, hp["heads"], 2 * w, hp["dropout"], batch_first=True)
                self.attn = nn.MultiheadAttention(w, hp["heads"], dropout=hp["dropout"], batch_first=True)
            self.drop = nn.Dropout(hp["dropout"])
            self.head = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Dropout(hp["dropout"]), nn.Linear(w, 1))

        @staticmethod
        def time_features(t):  # t in [0, 1], (..., n) -> (..., n, 16)
            f = 2 ** torch.arange(8, dtype=t.dtype, device=t.device) * torch.pi
            return torch.cat([torch.sin(t[..., None] * f), torch.cos(t[..., None] * f)], dim=-1)

        def forward(self, last, options, video=None):
            """last (B, D), options (B, K, D), video {source: (tokens (B, n, d), times (B, n))} -> (B, K)."""
            q = self.drop(nn.functional.gelu(self.last(last)[:, None] + self.opt(options)))  # (B, K, w)
            if self.sources:
                z = torch.cat([self.vis[s](video[s][0]) + self.kind[s] + self.time(self.time_features(video[s][1]))
                               for s in self.sources], dim=1)
                z = self.temporal(z)
                a, _ = self.attn(q, z, z, need_weights=False)
            else:
                a = torch.zeros_like(q)
            return self.head(torch.cat([q, a, q * a], dim=-1)).squeeze(-1)

    return Probe()


def holdout_videos(video_ids, frac=HP["holdout"]):
    """The train videos held out for early stopping: a fixed, seed-independent ~``frac`` of them."""
    import hashlib
    return {v for v in video_ids if int(hashlib.sha256(v.encode()).hexdigest(), 16) % 1000 < frac * 1000}


def load_split(set_name, sources, limit_videos=0):
    """Questions with their features, as arrays. Videos missing any feature are an error unless smoke."""
    rows, by_vid = [], _by_video(_questions(set_name))
    vids = list(by_vid)[:limit_videos or None]
    if not limit_videos:
        missing = [(s, v) for v in vids for s in ["text", *sources] if not os.path.exists(feature_file(s, v))]
        if missing:
            raise SystemExit(f"{set_name}: {len(missing)} feature files missing, e.g. {missing[0]}; run the earlier stages")
    feats = {s: {} for s in sources}
    for vid in vids:
        if limit_videos and not all(os.path.exists(feature_file(s, vid)) for s in ["text", *sources]):
            continue
        tf = dict(np.load(feature_file("text", vid)))  # read each member once
        idx = {int(q): i for i, q in enumerate(tf["qid"])}
        for s in sources:
            f = np.load(feature_file(s, vid))
            feats[s][vid] = (f["tokens"].astype(np.float32), f["times"])
        for q in by_vid[vid]:
            i = idx[q["qid"]]
            rows.append({"q": q, "last": tf["last"][i].astype(np.float32), "options": tf["options"][i].astype(np.float32),
                         "blind": tf["blind"][i]})
    return rows, feats


def _batch(rows, feats, sources, idx, device):
    import torch
    sel = [rows[i] for i in idx]
    last = torch.tensor(np.stack([r["last"] for r in sel]), device=device)
    opts = torch.tensor(np.stack([r["options"] for r in sel]), device=device)
    video = {}
    for s in sources:
        tok = [feats[s][r["q"]["video_id"]] for r in sel]
        n = max(t[0].shape[0] for t in tok)
        if any(t[0].shape[0] != n for t in tok):
            raise ValueError(f"{s}: videos with different numbers of vectors")
        dur = np.array([max(r["q"]["duration_s"], 1e-3) for r in sel], dtype=np.float32)[:, None]
        video[s] = (torch.tensor(np.stack([t[0] for t in tok]), device=device),
                    torch.tensor(np.clip(np.stack([t[1] for t in tok]) / dur, 0, 1), device=device))
    y = torch.tensor([r["q"]["answer_id"] for r in sel], device=device)
    return last, opts, video, y


def train_arm(arm, seed, train_rows, dev_rows, feats, hp=HP, log=print):
    """Train one probe; return its dev probabilities and its early-stopping curve."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sources = sources_of(arm)
    device = "cpu"  # small model, variable shapes: CPU is fast enough and deterministic
    dims = {"_text": train_rows[0]["last"].shape[0], **{s: next(iter(feats[s].values()))[0].shape[1] for s in sources}}
    model = build_probe(dims, hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    held = holdout_videos({r["q"]["video_id"] for r in train_rows}, hp["holdout"])
    fit = [i for i, r in enumerate(train_rows) if r["q"]["video_id"] not in held]
    val = [i for i, r in enumerate(train_rows) if r["q"]["video_id"] in held]
    if not val:  # only in a smoke test, with a handful of videos
        val = fit

    def nll(rows, idx):
        model.eval()
        tot, probs = 0.0, []
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                last, opts, video, y = _batch(rows, feats, sources, idx[s:s + 256], device)
                lg = model(last, opts, video)
                tot += float(torch.nn.functional.cross_entropy(lg, y, reduction="sum"))
                probs.append(torch.softmax(lg, -1).numpy())
        return tot / max(len(idx), 1), np.concatenate(probs) if probs else np.zeros((0, 3))

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
        v, _ = nll(train_rows, val)
        curve.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    _, probs = nll(dev_rows, list(range(len(dev_rows))))
    log(f"{arm} seed {seed}: {len(curve)} epochs, best held-out NLL {best:.3f} at epoch {int(np.argmin(curve)) + 1}")
    return probs, curve


def probe_file(arm, seed, smoke=False):
    return os.path.join(out_dir(), "probe_smoke" if smoke else "probe", f"{arm}_seed{seed}.json")


def train(args):
    storage.require_root()
    arms = args.arms.split(",") if args.arms else ARMS
    smoke = bool(args.limit_videos)
    hp = dict(HP, epochs=2) if smoke else HP
    for arm in arms:
        todo = [s for s in SEEDS if not os.path.exists(probe_file(arm, s, smoke))]
        if not todo:
            continue
        sources = sources_of(arm)
        tr, ftr = load_split("train", sources, args.limit_videos)
        dv, fdv = load_split("dev", sources, args.limit_videos)
        feats = {s: {**ftr[s], **fdv[s]} for s in sources}
        print(f"{arm}: {len(tr)} train / {len(dv)} dev questions", flush=True)
        for seed in todo:
            t0 = time.time()
            probs, curve = train_arm(arm, seed, tr, dv, feats, hp)
            out = {"arm": arm, "seed": seed, "hp": hp, "curve": curve, "train_s": time.time() - t0,
                   "rows": [{"video_id": r["q"]["video_id"], "qid": r["q"]["qid"], "probs": p.tolist()}
                            for r, p in zip(dv, probs)]}
            os.makedirs(os.path.dirname(probe_file(arm, seed, smoke)), exist_ok=True)
            with storage.drive_guard(), open(probe_file(arm, seed, smoke), "w") as f:
                json.dump(out, f)
            if not smoke:
                acc = float(np.mean([int(np.argmax(p)) == r["q"]["answer_id"] for r, p in zip(dv, probs)]))
                log_run(STEP, "train", {"arm": arm, **hp}, {"train": len(tr), "dev": len(dv)}, seed,
                        {"dev_accuracy": acc, "epochs": len(curve), "best_holdout_nll": min(curve)})


# ------------------------------------------------------------------ report
def arm_rows(arm, questions, smoke=False):
    """{(video_id, qid): eval-style row} with seed-mean probabilities, or None if a seed is missing."""
    runs = []
    for s in SEEDS:
        p = probe_file(arm, s, smoke)
        if not os.path.exists(p):
            return None
        with storage.drive_guard():
            runs.append({(r["video_id"], r["qid"]): r["probs"] for r in json.load(open(p))["rows"]})
    seed_acc = [float(np.mean([int(np.argmax(p)) == questions[k]["answer_id"] for k, p in r.items()])) for r in runs]
    out = {}
    for k in runs[0]:
        q = questions[k]
        probs = np.mean([r[k] for r in runs], axis=0)
        out[k] = {"probs": probs.tolist(), "answer_id": q["answer_id"], "correct": int(np.argmax(probs)) == q["answer_id"],
                  "area": q["area"], "reasoning": q["reasoning"], "latency_s": 0.0, "decode_s": 0.0}
    seed_accs[arm] = seed_acc
    return out


seed_accs = {}  # arm -> per-seed dev accuracy, filled by arm_rows


def verdict(rows):
    """(state, winner, tests). state: 'incomplete' | 'inconclusive' | 'pass' | 'fail'."""
    from .teacher import paired
    if rows.get("text") is None or rows.get("gemma") is None:
        return "incomplete", None, {}
    tests = {"gemma vs text": paired(rows["text"], rows["gemma"])}
    for x in CANDIDATES:
        if rows.get(x) is not None:
            tests[f"{x} vs gemma"] = paired(rows["gemma"], rows[x])
        if rows.get(f"gemma+{x}") is not None:
            tests[f"gemma+{x} vs gemma"] = paired(rows["gemma"], rows[f"gemma+{x}"])
    g, lost, p = tests["gemma vs text"]
    if not (g > lost and p < P_SANITY):
        return "inconclusive", None, tests
    if any(rows.get(f"gemma+{x}") is None for x in CANDIDATES):
        return "incomplete", None, tests
    bar = P_SANITY / len(CANDIDATES)
    passed = {x: tests[f"gemma+{x} vs gemma"] for x in CANDIDATES
              if tests[f"gemma+{x} vs gemma"][0] > tests[f"gemma+{x} vs gemma"][1] and tests[f"gemma+{x} vs gemma"][2] < bar}
    if not passed:
        return "fail", None, tests
    return "pass", max(passed, key=lambda x: (passed[x][0] - passed[x][1], -passed[x][2])), tests


def report(args):
    from .evaluate import metrics
    storage.require_root()
    smoke = args.smoke
    questions = {(q["video_id"], q["qid"]): q for q in _questions("dev")}
    rows = {a: arm_rows(a, questions, smoke) for a in ARMS}
    if all(v is None for v in rows.values()):
        raise SystemExit("no complete step 9 probe results yet (python -m video_train.probe run)")
    blind = {}
    for vid in {k[0] for k in questions}:
        if os.path.exists(feature_file("text", vid)):
            tf = np.load(feature_file("text", vid))
            for qid, p in zip(tf["qid"], tf["blind"]):
                q = questions[(vid, int(qid))]
                blind[(vid, int(qid))] = {"probs": p.tolist(), "answer_id": q["answer_id"], "area": q["area"],
                                          "reasoning": q["reasoning"], "correct": int(np.argmax(p)) == q["answer_id"],
                                          "latency_s": 0.0, "decode_s": 0.0}
    state, win, tests = verdict(rows)
    lines = [f"# {STEP}: probing vision encoders", "",
             "Same probe and text features in every arm; only the video features change. Trained on Perception "
             "Test train (3 seeds; rows are seed-mean probabilities), scored once on dev. Probe accuracy measures what "
             "each encoder's features make available, not what a Gemma model with that encoder would score.", "",
             "| arm | dev accuracy | seeds (min-max) | NLL | net vs text (p) | net vs gemma (p) |", "|---|---|---|---|---|---|"]
    from .teacher import paired
    for a in ARMS:
        r = rows[a]
        if r is None:
            lines.append(f"| {a} | not run | | | | |")
            continue
        m = metrics(list(r.values()))
        sa = seed_accs[a]

        def net(ref):
            if rows.get(ref) is None or ref == a:
                return "-"
            g, lost, p = paired(rows[ref], r)
            return f"{g - lost:+d} ({p:.2g})"
        lines.append(f"| {a} | {m['accuracy']:.1%} ± {m['ci95']:.1%} | {min(sa):.1%}-{max(sa):.1%} | {m['nll']:.3f} | "
                     f"{net('text')} | {net('gemma')} |")
    if blind:
        mb = metrics(list(blind.values()))
        lines += ["", f"Reference: base Gemma on dev without the video {mb['accuracy']:.1%} "
                      f"({len(blind)} questions); step 8's s8_A_ep2 with the video 73.5%."]
    done = [a for a in ARMS if rows[a] is not None]
    areas = sorted({q["area"] for q in questions.values()})
    lines += ["", "| area | n | " + " | ".join(done) + " |", "|---|---|" + "---|" * len(done)]
    for ar in areas:
        n = sum(v["area"] == ar for v in rows[done[0]].values())
        if n:
            accs = [np.mean([v["correct"] for v in rows[a].values() if v["area"] == ar]) for a in done]
            lines.append(f"| {ar} | {n} | " + " | ".join(f"{x:.1%}" for x in accs) + " |")
    enc = _encoder_costs()
    if enc:
        lines += ["", "Encoder cost on this Mac (16 frames, model compute only; frozen, fp32, MPS):", "",
                  "| encoder | parameters | encode s / video |", "|---|---|---|"]
        lines += [f"| {n} | {c['params_m']:.0f}M | {c['encode_s_per_video']:.2f} |" for n, c in enc.items()]
    lines += ["", "## Verdict", ""]
    for name, (g, lost, p) in tests.items():
        lines.append(f"- {name}: gains {g}, loses {lost} (net {g - lost:+d}, p={p:.2g})")
    lines.append("")
    bar = P_SANITY / len(CANDIDATES)
    lines.append({
        "incomplete": "**Incomplete:** some arms have not run yet.",
        "inconclusive": "**Inconclusive:** the probe does not beat text-only with Gemma's own video features, so it "
                        "cannot tell whether other encoders would help. Step 10 is not decided by this step.",
        "fail": f"**No encoder adds to Gemma's own features** (bar: p < {bar:.3f}). Skip step 10; go to step 11.",
        "pass": f"**{win} adds to Gemma's own features** (bar: p < {bar:.3f}). Step 10 builds on {win}.",
    }[state])
    os.makedirs(out_dir(), exist_ok=True)
    dst = os.path.join(out_dir(), "summary_smoke.md" if smoke else "summary.md")
    with storage.drive_guard(), open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-len(tests) - 3:]))
    print(f"\nwrote {dst}")
    if not smoke:
        log_run(STEP, "eval", {"what": "report"}, {"set": "dev"}, None, {"state": state, "winner": win})


def _encoder_costs():
    """The latest logged cost per encoder, from history.jsonl."""
    out = {}
    try:
        with storage.drive_guard():
            for line in open(storage.path("history.jsonl")):
                r = json.loads(line)
                if r.get("step") == STEP and r.get("config", {}).get("what") == "probe features":
                    for n, c in r["results"].items():
                        if isinstance(c, dict) and c.get("encode_s_per_video") is not None and c["videos"] >= 50:
                            out[n] = c
    except FileNotFoundError:
        pass
    return out


def run(args):
    storage.require_root()
    missing = [n for n in CANDIDATES if not _on_drive_or_mac(n)]
    if missing:
        raise SystemExit(f"encoders not downloaded: {missing}; run `python -m video_train.probe fetch` first")
    pool(args)
    text(args)
    args.encoders = ",".join(CANDIDATES)
    encode(args)
    args.arms = None
    train(args)
    args.smoke = bool(args.limit_videos)
    report(args)


def _on_drive_or_mac(name):
    from .models import LOCAL, _has_weights, store_path
    folder = ENCODERS[name][1]
    return _has_weights(os.path.join(LOCAL, folder)) or _has_weights(store_path(folder))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"fetch": fetch, "text": text, "pool": pool, "encode": encode, "train": train, "report": report, "run": run}
    p = {c: sub.add_parser(c) for c in cmds}
    for c in ("fetch", "encode"):
        p[c].add_argument("--encoders", default=",".join(CANDIDATES))
    for c in ("text", "pool", "encode", "train", "run"):
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
