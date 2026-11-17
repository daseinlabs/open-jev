"""Stage 5: landmark tasks from 4 head cameras (4 videos in one prompt); the game is never trained on.

    python -m video_train.vdm_cams --tag cams_stageA_ep1 --adapter <stageA_ep1 dir>
    python -m video_train.vdm_cams --tag cams_oracle --oracle      # the map answers: the rule's ceiling

Input: 4 videos, one per camera on the walker's head (``Renderer.cameras``: front, left, back,
right, each 90° wide, so no gaps), the last 4 s at 2 fps each, in one prompt with each camera's
label before its video (left, back, right, front: see ``PROMPT_ORDER``), after a blank warm-up
video that takes the under-seen first slot. Each view is a normal camera view, not an unrolled
360° strip.
``--tokens`` per 2 frames of one camera (default 192: the 640x400 frames' sharpness).

Each second, on that input (encoded once, then one short pass per question):
1. ``cams``: "Can you see the {name} in the {front/left/back/right} camera now?"  4 yes/no; the
   camera with the highest P(yes) above 0.5, or none
2. if front, ``point``: the {name}'s bounding box in the last frame of the front camera (Qwen's
   grounding JSON, 0-1000); its centre gives the exact angle, so the side: within ±22.5° the
   middle, else left / right. ``words`` (always asked, logged for comparison; decides only when
   the box cannot be read): "In the front camera, where is the {name} now?"  left side / middle /
   right side
3. if front middle, ``near``: no question; the box's angular width and the landmark's known width
   (``VIEW_WIDTH``) give its distance; within the goal's distance (footprint + 4 m) -> stop

Rule: none -> walk forward (explore); left / right camera -> turn 45° that way; back -> turn
around; front, left or right side -> turn 45° that way; front middle -> forward, or stop if next
to it. A 45° turn puts a target 22.5-45° off in the middle: no zig-zag. A forward step that moved
less than 0.5 m makes the next forward a 45° turn toward the landmark's side (left if unknown).

Episodes: the first 30 dev landmark episodes that need no river crossing. Every answer is stored with the map's
truth. Rows: ``vdm/results/stage5/<tag>/cams_<split>.jsonl``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time

import numpy as np

from . import storage
from .history import log_run
from .vdm_data import game_clip
from .vdm_game import MAX_STEPS, angle_to, expert_steps, landmark_episodes, summary

CAMS = ("front", "left", "back", "right")  # Renderer.cameras order: heading + 0, +90, 180, -90 (positive = left)
FRONT, LEFT, BACK, RIGHT = range(4)
# The first video in the prompt is under-seen (2026-11-16 probe, base model: the same front view,
# P(yes) 0.2-0.5 as the first video, 1.0 as the third or last), so the front camera goes last and a
# blank warm-up video takes the first slot (cams-v1, no warm-up: the left camera, then first, was
# seen in 9 of 193 steps).
PROMPT_ORDER = (LEFT, BACK, RIGHT, FRONT)
SIDES = ("on the left side", "in the middle", "on the right side")
MIDDLE_DEG = 22.5  # half of one 45° turn
RULE = "cams-v2"
# Landmark widths seen from the side (m), from play.py's meshes: the box's angular width gives the
# distance, and "next to it" is the goal's distance (expert.goal_mask: footprint + 4 m). Replaces
# cams-v1's "Are you right next to it?" (P(yes) ~0.5 whatever the truth: 12 of 22 episodes stopped early).
VIEW_WIDTH = {"red house": 7.0, "stone tower": 5.0, "windmill": 4.4, "big tree": 8.0, "blue tent": 3.4}
NEAR_M = 4.0
BLOCKED_M = 0.5  # a forward step that moved less than this: blocked, turn instead
CLIP = 8  # 4 s at 2 fps
POINT_TOKENS = 64

INTRO = ("These are 4 videos from 4 cameras on your head while you walk through a 3D game world, after a blank "
         "warm-up video. Each covers the last 4 seconds; the last frame is now.\n")
WARMUP = "Warm-up video (blank, not a camera):"
LABELS = ("Front camera (what is straight ahead of you):", "Left camera (to your left):",
          "Back camera (behind you):", "Right camera (to your right):")
BOX = re.compile(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


def questions(name):
    return {"cams": [f"Can you see the {name} in the {c} camera now?" for c in CAMS],
            "point": f"Locate the {name} in the last frame of the front camera. Report its bbox coordinates in JSON format.",
            "words": f"In the front camera, where is the {name} now?"}


def side_of(angle):
    """0 left, 1 middle, 2 right (``SIDES``) for an angle in the front camera, positive = left."""
    return 0 if angle > MIDDLE_DEG else 2 if angle < -MIDDLE_DEG else 1


def cx_to_angle(cx):
    """Degrees from straight ahead (positive = left) of a box centre ``cx`` (0-1000) in the front camera."""
    return math.degrees(math.atan((1 - cx / 500) * math.tan(math.radians(45))))


def parse_cx(text):
    """The horizontal centre of the first bbox in the model's answer, or None."""
    m = BOX.search(text)
    if not m:
        return None
    x1, _, x2, _ = map(float, m.groups())
    return (x1 + x2) / 2


def truth(w, task, x, z, h):
    """cam: the camera the landmark's centre falls in if it is in sight (terrain,
    distance; trees ignored), else None; side: its side in the front camera; near: at the goal."""
    from openworld import expert as E
    l = next(l for l in w.landmarks if l.name == task.target)
    seen = E.in_view(w, x, z, E.heading_to(x, z, l.x, l.z), task.target)
    a = angle_to(x, z, h, l.x, l.z)
    return {"cam": int(round(a / 90.0)) % 4 if seen else None, "side": side_of(a), "angle": round(a, 1),
            "near": task.done(w, x, z)}


def decide(a):
    """(move, reason) from the answers: cam, side, near; blocked (the last forward step did not move)
    turns 45° instead of walking forward, toward the landmark's side of the front camera (``angle``)."""
    from openworld import expert as E
    m, why = _decide(a)
    if m == E.FORWARD and a.get("blocked"):
        return (E.RIGHT, "blocked, turn right") if (a.get("angle") or 0) < 0 else (E.LEFT, "blocked, turn left")
    return m, why


def _decide(a):
    from openworld import expert as E
    c = a["cam"]
    if c is None:
        return E.FORWARD, "not seen, explore"
    if c == BACK:
        return E.AROUND, "behind"
    if c in (LEFT, RIGHT):
        return (E.LEFT, "left camera") if c == LEFT else (E.RIGHT, "right camera")
    if a["side"] == 1:
        return (E.STOP, "next to it") if a.get("near") else (E.FORWARD, "ahead")
    return (E.LEFT, "front, left side") if a["side"] == 0 else (E.RIGHT, "front, right side")


def videos(clips):
    """The prompt's input: the intro and a blank warm-up video, then (label, video) per camera in ``PROMPT_ORDER``."""
    from PIL import Image
    f = clips[FRONT][0]
    blank = [Image.new("RGB", f.size, (128, 128, 128))] * len(clips[FRONT])
    return [(INTRO + WARMUP, game_clip(blank, 2))] + [(LABELS[k], game_clip(clips[k], 2)) for k in PROMPT_ORDER]


def box_distance(text, target):
    """Metres to ``target`` from the angular width of the first bbox in the front camera, or None."""
    m = BOX.search(text)
    if not m:
        return None
    x1, _, x2, _ = map(float, m.groups())
    width = math.radians(abs(cx_to_angle(x1) - cx_to_angle(x2)))
    return VIEW_WIDTH[target] / 2 / math.tan(width / 2) if width > 0 else None


def ask(b, name, t, clips):
    """The model's answers on one input (encoded once) and a log of each: probabilities and truth."""
    from . import qwen3vl as q3
    q, a, log = questions(name), {}, {}
    row = lambda text, ans: {"source": "game", "question": text, "answer": ans}
    b.start(videos(clips), row(q["cams"][0], t["cam"] == FRONT))
    p = [b.score(row(text, t["cam"] == k))["p_yes"] for k, text in enumerate(q["cams"])]
    a["cam"] = int(np.argmax(p)) if max(p) > 0.5 else None
    log["cams"] = {"p": [round(v, 4) for v in p], "truth": t["cam"]}
    if a["cam"] == FRONT:
        out = q3.generate(b.model, b.inp, b.inp.encode([(q["point"], None)], b.video, True), b.pre, POINT_TOKENS)
        cx = parse_cx(out)
        ang = None if cx is None else cx_to_angle(cx)
        probs = b.score({"source": "game", "question": q["words"], "options": list(SIDES), "answer_id": t["side"]})["probs"]
        words = int(np.argmax(probs))
        a["side"], a["angle"] = (words if ang is None else side_of(ang)), ang
        log["point"] = {"output": out, "cx": cx, "angle": None if ang is None else round(ang, 1),
                        "truth": t["angle"]}
        log["words"] = {"p": [round(v, 4) for v in probs], "truth": t["side"]}
        if a["side"] == 1:
            from openworld.world import FOOTPRINT
            dist = box_distance(out, name)
            a["near"] = dist is not None and dist <= FOOTPRINT[name] + NEAR_M
            log["near"] = {"dist": None if dist is None else round(dist, 1), "truth": t["near"]}
    return a, log


def run(args):
    from openworld import expert as E
    from openworld.render import Renderer
    storage.require_root()
    d = storage.path("vdm", "results", "stage5", args.tag)
    os.makedirs(d, exist_ok=True)
    cfg = {"adapter": args.adapter, "split": args.split, "episodes": args.episodes, "tokens": args.tokens,
           "rule": RULE, "oracle": args.oracle}
    cfg_path = os.path.join(d, f"cams_{args.split}.json")
    if os.path.exists(cfg_path) and (prev := json.load(open(cfg_path))) != cfg:
        sys.exit(f"{args.tag} ran with {prev}, not {cfg}; use another --tag")
    with storage.drive_guard(), open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    path = os.path.join(d, f"cams_{args.split}.jsonl")
    done = {json.loads(l)["seed"] for l in open(path)} if os.path.exists(path) else set()
    awake = storage.keep_awake()
    b = r = None
    scope = contextlib.nullcontext()
    if not args.oracle:
        from .adapters import adapter_scope, install_adapter
        from .vdm_eval import Qwen
        b, r = Qwen(args.tokens), Renderer()
        print(f"adapter on {install_adapter(b.model, 'video', args.adapter)} layers", flush=True)
        scope = adapter_scope("video")
    with scope:
        for seed, (w, pose, task) in landmark_episodes(args.split, args.episodes):
            if seed in done:
                continue
            storage.check_alive()
            best = expert_steps(w, pose, task)
            limit = min(MAX_STEPS, 2 * best + 20)
            (x, z, h), t = pose, 0.0
            frames = []  # per moment, the 4 camera images
            if r:
                r.load(w)
                frames = [r.cameras(x, z, h, t)]
            steps, reached, stopped, blocked, t0 = [], False, False, False, time.time()
            for _ in range(limit):
                tr = truth(w, task, x, z, h)
                if args.oracle:
                    a, log = {k: tr[k] for k in ("cam", "side", "near", "angle")}, {}
                else:
                    clip = [frames[0]] * (CLIP - len(frames)) + frames[-CLIP:]
                    a, log = ask(b, task.target, tr, [[m[k] for m in clip] for k in range(4)])
                a["blocked"] = blocked
                m, why = decide(a)
                x0, z0 = x, z
                steps.append({"expert": E.expert(w, task, x, z, h), "act": m, "why": why, "q": log})
                if m == E.STOP:
                    stopped = True
                    break
                for x, z, h in E.move(w, x, z, h, m):
                    t += 1 / E.FPS
                    if r:
                        frames.append(r.cameras(x, z, h, t))
                frames = frames[-CLIP:]
                blocked = m == E.FORWARD and math.hypot(x - x0, z - z0) < BLOCKED_M
                reached |= task.done(w, x, z)
            success = stopped and task.done(w, x, z)
            moved = [s for s in steps if s["act"] != E.STOP and s["expert"] != E.STOP]
            rec = {"seed": seed, "kind": task.kind, "instruction": task.text, "target": task.target,
                   "success": success, "reached": reached or success, "stopped_early": stopped and not success,
                   "timed_out": not stopped, "steps": len(steps), "expert_steps": best,
                   "spl": success * best / max(best, len(steps) - 1),
                   "agree": float(np.mean([s["act"] == s["expert"] for s in moved])) if moved else None,
                   "s_per_step": (time.time() - t0) / len(steps), "trace": steps}
            with storage.drive_guard(), open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            how = "SUCCESS" if success else ("stopped early" if stopped else "timed out")
            print(f"seed {seed} {task.text!r}: {how} after {len(steps)} steps (expert {best}), "
                  f"{rec['s_per_step']:.1f}s/step", flush=True)
    if awake:
        awake.terminate()
    summary(args.tag, args.split, path)
    if not args.oracle:
        accuracy(args.tag, args.split, path)


def accuracy(tag, split, path):
    """Which camera (confusion, rows = truth), the front side by box and by words, and near."""
    from openworld.world import FOOTPRINT
    rs = [json.loads(l) for l in open(path)]
    qs = [s["q"] for r in rs for s in r["trace"]]
    names = CAMS + ("none",)
    C = np.zeros((5, 5), int)
    for q in qs:
        p, tc = q["cams"]["p"], q["cams"]["truth"]
        C[4 if tc is None else tc, int(np.argmax(p)) if max(p) > 0.5 else 4] += 1
    res = {"cams": {"n": int(C.sum()), "acc": float(np.trace(C) / C.sum()), "confusion": C.tolist()}}
    lines = [f"# Per question, against the map: {tag} ({split})", "",
             f"which camera: {res['cams']['acc']:.0%} of {C.sum()} (rows = truth, columns = the model's answer)", "",
             "| true \\ answer | " + " | ".join(names) + " |", "|---|" + "---:|" * 5]
    lines += [f"| {names[i]} | " + " | ".join(str(v) for v in C[i]) + " |" for i in range(5)]
    # The side only where the landmark really is in the front camera.
    fr = [q for q in qs if "point" in q and q["cams"]["truth"] == FRONT]
    if fr:
        ok = [q for q in fr if q["point"]["angle"] is not None]
        err = [abs(q["point"]["angle"] - q["point"]["truth"]) for q in ok]
        res["point"] = {"n": len(fr), "parsed": len(ok) / len(fr),
                        "side_acc": float(np.mean([side_of(q["point"]["angle"]) == q["words"]["truth"] for q in ok])) if ok else None,
                        "median_err_deg": float(np.median(err)) if err else None}
        res["words"] = {"n": len(fr), "side_acc": float(np.mean([int(np.argmax(q["words"]["p"])) == q["words"]["truth"]
                                                                for q in fr]))}
        pct = lambda v: "-" if v is None else f"{v:.0%}"
        pt = res["point"]
        med = "-" if pt["median_err_deg"] is None else f"{pt['median_err_deg']:.1f}°"
        lines += ["", f"front side, where the landmark is in the front camera (n {len(fr)}):", "",
                  "| method | side accuracy | other |", "|---|---:|---|",
                  f"| box | {pct(pt['side_acc'])} | box read {pct(pt['parsed'])}, median angle error {med} |",
                  f"| words | {pct(res['words']['side_acc'])} | |"]
    nr = [(q["near"], FOOTPRINT[r["target"]] + NEAR_M) for r in rs for s in r["trace"] for q in [s["q"]] if "near" in q]
    if nr:
        said = lambda x, lim: x["dist"] is not None and x["dist"] <= lim
        res["near"] = {"n": len(nr), "acc": float(np.mean([said(x, lim) == x["truth"] for x, lim in nr]))}
        lines += ["", f"near: {res['near']['acc']:.0%} of {len(nr)}"]
    text = "\n".join(lines) + "\n"
    with storage.drive_guard(), open(os.path.splitext(path)[0] + "_questions.md", "w") as f:
        f.write(text)
    print(text)
    log_run("vdm-5", "eval", {"what": "game 4-camera questions", "tag": tag, "split": split}, None, None, res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA directory (required unless --oracle)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=192, help="vision tokens per 2 frames of one camera")
    ap.add_argument("--oracle", action="store_true", help="answers from the map, no model: tests the rule")
    args = ap.parse_args(argv)
    if not args.oracle and not args.adapter:
        ap.error("--adapter is required (only trained models are run)")
    try:
        run(args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
