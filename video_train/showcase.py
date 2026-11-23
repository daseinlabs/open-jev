"""Live showcase reel: the clip plays on the left, the question on the right, the model answers.

    python -m video_train.showcase --adapter "<DATA_ROOT>/vdm/checkpoints/stageB_ep1"
    python -m video_train.showcase --adapter <dir> --set nextqa_dev --n 4 --seed 1 --window

Nothing is precomputed. Each clip plays at real speed, and the model only gets frames that the
playback has already passed: as each pair of the 16 sampled frames goes by, the vision encoder
runs on that pair and the language model extends its KV cache with it. Qwen3-VL encodes each
2-frame group on its own, so this matches encoding the whole clip at once. When the clip ends,
only the question's own tokens are left to run, and the answer appears as soon as they finish.
The on-screen "answered N s after the clip ended" is wall-clock time from the last frame to the
answer, including any encoding still queued.

The output MP4 records the screen as it ran (``--window`` also shows it live). Clips are held-out
dev questions (multiple choice, one per video), drawn with ``--seed`` and kept only if
``--min-seconds`` to ``--max-seconds`` long; one more clip, never shown, warms the model up
first. The console prints, per clip, how far the live option probabilities are from that
adapter's stored ``vdm_eval`` row (measured: under 0.01).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import storage
from .vdm_data import DEV, kind, load_row_video, rows, turns

W, H, FPS = 1280, 720, 25
VID = (40, 110, 780, 440)                 # x, y, w, h of the video area
PANEL_X = 850
BG, FG, DIM, ACCENT, GOOD, BAD = (16, 18, 24), (240, 240, 245), (130, 135, 150), (90, 160, 255), (70, 200, 120), (235, 90, 80)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
GROUPS = 8                                # 16 frames, 2 per group


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


F_TITLE, F_Q, F_OPT, F_SMALL = font(26, True), font(24, True), font(21), font(17)


# ------------------------------------------------------------------ model side
class LiveReader:
    """One question on one clip: frame groups are fed as they play, then the question runs."""

    def __init__(self, qwen, row, video):
        self.q, self.row, self.video = qwen, row, video
        self.text = turns(row)[0][0]
        self.cache, self.ids, self.enc = None, [], None

    def add_group(self, k):
        """Encode frames 2k, 2k+1 (both already played) and extend the KV cache with them."""
        import mlx.core as mx
        from mlx_vlm.models.cache import KVCache
        from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter
        from openjev.video import SampledVideo
        q3, m, v = self.q.q3, self.q.model, self.video
        part = SampledVideo(v.frames[:2 * k + 2], v.timestamps[:2 * k + 2], v.duration_s, v.width, v.height, "", 0)
        enc = self.q.inp.encode([(self.text, None)], part, True)
        ids = enc.input_ids[0].tolist()
        vt = m.config.video_token_index
        n = max(i for i, t in enumerate(ids) if t == vt) + 1
        if ids[:len(self.ids)] != self.ids:
            raise ValueError("partial prompt is not an extension of the cached one")
        t, gh, gw = (int(x) for x in enc.grid_thw[0])
        per = gh * gw
        if t != k + 1 or enc.pixel_values.shape[0] != t * per:
            raise ValueError(f"unexpected video grid {enc.grid_thw.tolist()} for group {k}")
        one = q3.Encoded(None, enc.pixel_values[k * per:], np.array([[1, gh, gw]]), "")
        feats, deep = q3.vision_features(m, one)
        if self.cache is None:
            self.cache = [KVCache() for _ in m.language_model.model.layers]
        lm, a = m.language_model, len(self.ids)
        new = mx.array([ids[a:n]])
        vmask = new == vt
        emb = masked_scatter(lm.model.embed_tokens(new), mx.broadcast_to(vmask[..., None], (*new.shape, feats.shape[-1])), feats)
        pos, _ = lm.get_rope_index(mx.array(enc.input_ids), None, mx.array(enc.grid_thw), None)
        lm.model(new, inputs_embeds=emb, cache=self.cache, position_ids=pos[..., a:n], visual_pos_masks=vmask,
                 deepstack_visual_embeds=deep)
        mx.eval([c.keys for c in self.cache])
        self.ids, self.enc = ids[:n], enc

    def answer(self):
        """Option probabilities from the cached clip (the last group's prompt is the full one)."""
        import mlx.core as mx
        q3, m = self.q.q3, self.q.model
        h = q3.hidden_last(m, self.enc, q3.Prefix(self.ids, self.cache))
        ids = self.q.inp.letter_ids[:len(self.row["options"])]
        return np.array(mx.softmax(q3.logits_at(m, h, ids))).astype(float)


def worker(qwen, jobs, adapter_name):
    from .adapters import adapter_scope
    with adapter_scope(adapter_name):     # context variable: must be entered in this thread
        while (job := jobs.get()) is not None:
            job()


# ------------------------------------------------------------------ drawing
def wrap(draw, text, f, width):
    lines, cur = [], ""
    for word in text.split():
        nxt = f"{cur} {word}".strip()
        if draw.textlength(nxt, font=f) <= width or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)


def card(lines):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - 30 * len(lines)
    for text, f, color in lines:
        d.text((W // 2, y), text, font=f, fill=color, anchor="mm")
        y += 60
    return img


def frame(st, clip_img, header):
    """One output frame for clip state ``st`` (dict) with the current video image."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 40), header, font=F_TITLE, fill=FG)
    d.text((40, 76), st["source"], font=F_SMALL, fill=DIM)
    x, y, w, h = VID
    v = fit(clip_img, w, h)
    img.paste(v, (x + (w - v.width) // 2, y + (h - v.height) // 2))
    # what the model has read so far
    yb = y + h + 20
    d.text((x, yb), "model input", font=F_SMALL, fill=DIM)
    for g in range(GROUPS):
        cx = x + 120 + g * 34
        fill = ACCENT if g < st["groups_done"] else (60, 64, 76) if g >= st["groups_sent"] else (55, 90, 140)
        d.rounded_rectangle((cx, yb + 2, cx + 26, yb + 20), 4, fill=fill)
    d.text((x + 120 + GROUPS * 34 + 10, yb), f"{2 * st['groups_done']}/16 frames encoded", font=F_SMALL, fill=DIM)
    # question and options
    px, pw, py = PANEL_X, W - PANEL_X - 40, 110
    for line in wrap(d, st["question"], F_Q, pw):
        d.text((px, py), line, font=F_Q, fill=FG)
        py += 32
    py += 18
    probs, truth = st.get("probs"), st["answer_id"]
    pred = int(np.argmax(probs)) if probs is not None else None
    for i, opt in enumerate(st["options"]):
        lines = wrap(d, f"{'ABCDEFGH'[i]}.  {opt}", F_OPT, pw - 70)
        bh = 14 + 27 * len(lines)
        box = (px, py, px + pw, py + bh)
        if probs is not None:
            d.rounded_rectangle(box, 8, fill=(30, 34, 44))
            bar = int(pw * probs[i])
            if bar > 0:
                d.rounded_rectangle((px, py, px + max(bar, 10), py + bh), 8,
                                    fill=(GOOD if i == truth else BAD) if i == pred else (45, 52, 68))
            d.text((px + pw - 12, py + 8), f"{probs[i]:.0%}", font=F_OPT, fill=FG, anchor="ra")
        else:
            d.rounded_rectangle(box, 8, outline=(60, 64, 76), width=2)
        for j, line in enumerate(lines):
            d.text((px + 12, py + 8 + 27 * j), line, font=F_OPT, fill=FG)
        py += bh + 10
    py += 8
    if probs is not None:
        ok = pred == truth
        d.text((px, py), f"model: {'ABCDEFGH'[pred]}   {'correct' if ok else 'wrong, answer is ' + 'ABCDEFGH'[truth]}",
               font=F_Q, fill=GOOD if ok else BAD)
        d.text((px, py + 36), f"answered {st['latency']:.2f} s after the clip ended", font=F_SMALL, fill=DIM)
    elif st["ended"]:
        d.text((px, py), "deciding...", font=F_Q, fill=ACCENT)
    return img


# ------------------------------------------------------------------ playback
def play_frames(path):
    """(time, PIL image) for every frame, decoded lazily as playback reaches it."""
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        start = float(s.start_time * s.time_base) if s.start_time is not None and s.time_base else 0.0
        for f in c.decode(s):
            yield (f.time or 0.0) - start, f.to_image()


class Recorder:
    def __init__(self, out, window):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                                  stdin=subprocess.PIPE)
        self.window, self.n, self.t0 = window, 0, None

    def show(self, img):
        """Write ``img`` for every output tick up to now, so the file runs in wall-clock time."""
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        due = int((now - self.t0) * FPS) + 1
        buf = np.asarray(img).tobytes()
        while self.n < due:
            self.p.stdin.write(buf)
            self.n += 1
        if self.window:
            import cv2
            cv2.imshow("open-jev showcase", np.asarray(img)[:, :, ::-1])
            cv2.waitKey(1)
        nxt = self.t0 + self.n / FPS
        time.sleep(max(0.0, nxt - time.perf_counter()))

    def hold(self, img, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.show(img)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def run_clip(rec, jobs, qwen, row, header):
    video = load_row_video(row, 16)        # file read + decode only; frames reach the model below
    reader = LiveReader(qwen, row, video)
    st = {"question": row["question"], "options": row["options"], "answer_id": row["answer_id"],
          "source": f"{row['set']} · {row['video_id']} · held-out", "groups_sent": 0, "groups_done": 0,
          "ended": False, "probs": None}

    def group(k):
        reader.add_group(k)
        st["groups_done"] = k + 1

    def answer():
        probs = reader.answer()
        st["latency"] = time.perf_counter() - st["end_time"]
        st["probs"] = probs                # last: the drawing thread checks probs first

    due = [video.timestamps[2 * k + 1] for k in range(GROUPS)]  # second frame of each group
    frames = play_frames(storage.path(row["video"]))
    t_img, img = next(frames)
    nxt = next(frames, None)
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        while nxt is not None and nxt[0] <= t:
            img, nxt = nxt[1], next(frames, None)
        while st["groups_sent"] < GROUPS and due[st["groups_sent"]] <= t:
            jobs.put(lambda k=st["groups_sent"]: group(k))
            st["groups_sent"] += 1
        if t >= video.duration_s:
            break
        rec.show(frame(st, img, header))
    st["ended"], st["end_time"] = True, time.perf_counter()
    while st["groups_sent"] < GROUPS:      # timestamps all fall before the end; kept for safety
        jobs.put(lambda k=st["groups_sent"]: group(k))
        st["groups_sent"] += 1
    jobs.put(answer)
    while st["probs"] is None:
        rec.show(frame(st, img, header))
    rec.hold(frame(st, img, header), 4.0)
    return st


def pick(args):
    import av
    rs = [r for r in rows(DEV[args.set]) if kind(r) == "mc"]
    seen, by = set(), []
    for r in rs:
        if r["video_id"] not in seen:
            seen.add(r["video_id"])
            by.append(r)
    random.Random(args.seed).shuffle(by)
    out = []
    for r in by:
        with storage.drive_guard(), av.open(storage.path(r["video"])) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration and s.time_base else (c.duration or 0) / 1e6
        if args.min_seconds <= dur <= args.max_seconds:
            out.append(r)
        if len(out) == args.n + 1:
            return out
    return out


def stored(tag, name):
    p = storage.path("vdm", "results", "stage1", tag, f"{name}.jsonl")
    return {(r["video_id"], str(r["qid"])): r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="LoRA directory (adapters.save_adapter format)")
    ap.add_argument("--set", default="trafficqa_dev", help=f"one of: {', '.join(DEV)}")
    ap.add_argument("--n", type=int, default=6, help="clips in the reel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-seconds", type=float, default=5.0,
                    help="skip shorter clips: each frame pair takes ~0.6 s, so below ~5 s encoding falls behind playback")
    ap.add_argument("--max-seconds", type=float, default=12.0, help="skip longer clips")
    ap.add_argument("--tokens", type=int, default=192, help="vision tokens per 2 frames")
    ap.add_argument("--out", default=None, help="MP4 path (default <DATA_ROOT>/vdm/showcase/<tag>_<set>_s<seed>.mp4)")
    ap.add_argument("--window", action="store_true", help="also show the reel live in a window")
    args = ap.parse_args(argv)
    storage.require_root()
    tag = os.path.basename(os.path.normpath(args.adapter))
    out = args.out or storage.path("vdm", "showcase", f"{tag}_{args.set}_s{args.seed}.mp4")
    clips = pick(args)
    warm, clips = clips[-1], clips[:-1]    # one extra clip, used only to warm up MLX
    print(f"{len(clips)} clips from {args.set} (seed {args.seed}, {args.min_seconds:.0f}-{args.max_seconds:.0f} s)", flush=True)

    from .adapters import install_adapter
    from .vdm_eval import Qwen
    qwen = Qwen(args.tokens)
    install_adapter(qwen.model, "video", args.adapter)
    jobs = queue.Queue()
    th = threading.Thread(target=worker, args=(qwen, jobs, "video"), daemon=True)
    th.start()
    ref = stored(tag, args.set)
    done = threading.Event()

    def warm_up():
        reader = LiveReader(qwen, warm, load_row_video(warm, 16))
        for k in range(GROUPS):
            reader.add_group(k)
        reader.answer()
        done.set()

    jobs.put(warm_up)                      # first calls compile kernels; keep that out of the reel
    done.wait()

    rec = Recorder(out, args.window)
    header = "open-jev video decision model"
    rec.hold(card([(header, font(44, True), FG),
                   (f"Qwen3-VL-4B + LoRA ({tag}), MLX 8-bit, running live on a MacBook", F_OPT, DIM),
                   ("each clip plays in real time; the model reads frames as they play and answers when it ends", F_OPT, DIM)]), 4.0)
    right = 0
    for i, r in enumerate(clips):
        st = run_clip(rec, jobs, qwen, r, f"{header}  ·  clip {i + 1}/{len(clips)}")
        ok = int(np.argmax(st["probs"])) == r["answer_id"]
        right += ok
        old = ref.get((r["video_id"], str(r["qid"])))
        diff = f", max |p - eval| {np.abs(st['probs'] - np.array(old['probs'])).max():.4f}" if old else ""
        print(f"[{i + 1}/{len(clips)}] {r['video_id']} {'right' if ok else 'wrong'}, "
              f"answer {st['latency']:.2f} s after the clip{diff}", flush=True)
    rec.hold(card([(f"{right} of {len(clips)} correct on these clips", font(40, True), FG),
                   (f"clips drawn at random (seed {args.seed}) from held-out {args.set}", F_OPT, DIM)]), 4.0)
    jobs.put(None)
    rec.close()
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
