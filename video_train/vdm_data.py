"""Stages 1-2 of docs/video-decision-model-plan.md: benchmarks, training mix, videos, overlap check.

    python -m video_train.vdm_data bench          # never-trained-on benchmark manifests (from parquet)
    python -m video_train.vdm_data fetch-bench    # their videos (Video-MME short 3.1 GB, TempCompass 0.4 GB)
    python -m video_train.vdm_data mix            # stage 2 training mix -> vdm/manifests/train_mix.jsonl
    python -m video_train.vdm_data overlap        # no eval video in the mix (exit 2 if any)
    python -m video_train.vdm_data extract        # videos of every eval set and the mix -> vdm/videos/

Prompt formats (``turns``) live here so evaluation and training see the same text:

- multiple choice (Perception Test, NExT-QA, STAR, LLaVA-Video mc, Video-MME, TempCompass):
  question, lettered options, "Answer with the option's letter ..."; answer = the letter.
  STAR asks about a clip of a Charades video: the clip is given in the text
  ("From 7.7 to 15.7 seconds: ...") and the whole video is shown, so a video is encoded once.
- CLEVRER descriptive: "... Answer with a single word or number."; answer = the word.
  CLEVRER multi-answer: one yes/no turn per option; answer "Yes" / "No".
- Charades-STA: "When does this happen ...? Answer as 'start - end' in seconds."; answer "7.7 - 15.7".
- LLaVA-Video open-ended: the question as asked; answer = the reference answer.

Benchmarks (never trained on): Video-MME short (300 videos < 2 min, 900 questions, no
subtitles) and TempCompass multiple choice (410 videos, 1,580 questions). The long-video
benchmarks (MLVU, LVBench) are left out: 16 frames over 3 min - 2 h says little.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict

from . import storage
from .perception_data import load_manifest
from .qwen3vl import LETTERS, mc_text
from .vdm_splits import OPT, raw, sha256

SEED = 0
MAN = ("vdm", "manifests")
VIDEOMME_REV = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
TEMPCOMPASS_REV = "c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722"
TEMPCOMPASS_ZIP_SHA = "fb7c8a2b742a3d10ae1bd1a6183fb602edb346a4c7fd91c631dba46b753e0f0b"  # HF etag
LLAVA = "30_60_s_youtube_v0_1"

# Every evaluation set: name -> manifest (relative to DATA_ROOT). Test sets are only for the final run.
DEV = {"perception_dev": "manifests/dev.jsonl", "nextqa_dev": "manifests/nextqa_val1000.jsonl",
       **{f"{d}_dev": f"vdm/manifests/{d}_dev.jsonl" for d in ("star", "charades_sta", "clevrer", "llava_video")}}
BENCH = {"videomme_short": "vdm/manifests/videomme_short.jsonl", "tempcompass_mc": "vdm/manifests/tempcompass_mc.jsonl"}
TEST = {"perception_test": "manifests/test.jsonl",
        **{f"{d}_test": f"vdm/manifests/{d}_test.jsonl" for d in ("nextqa", "star", "charades_sta", "clevrer", "llava_video")}}
EVAL = {**DEV, **BENCH, **TEST}

# Stage 2 mix (plan, 2026-11-09): questions per source and the most taken from one video.
MIX = {"perception_test": (None, None), "nextqa": (7500, 3), "star": (7000, 4), "clevrer": (6000, 4),
       "charades_sta": (3000, 3), "llava_video": (6000, 4)}


def rows(rel):
    with storage.drive_guard():
        return [json.loads(l) for l in open(storage.path(rel))]


def write(rel, rs):
    p = storage.path(rel)
    with storage.drive_guard(), open(p + ".tmp", "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------------ prompt formats
YES, NO = "Yes", "No"
GROUND = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")


def kind(r) -> str:
    """mc | yesno | word | span | open | yn (one yes/no question: the game)."""
    if r["source"] == "game":
        return "mc" if "options" in r else "yn"
    if r["source"] == "clevrer":
        return "yesno" if "options" in r else "word"
    if r["source"] == "charades_sta":
        return "span"
    return "open" if r.get("type") == "oe" else "mc"


def turns(r) -> list[tuple[str, str]]:
    """(user text, reference answer) per turn of one question (CLEVRER multi-answer: one per option)."""
    k = kind(r)
    if k == "mc":
        q = r["question"]
        if r["source"] == "star":
            q = f"From {r['clip'][0]:.1f} to {r['clip'][1]:.1f} seconds: {q}"
        return [(mc_text(q, r["options"]), LETTERS[r["answer_id"]])]
    if k == "yesno":
        return [(f"{r['question']}\nOption: {o}\nIs this option correct? Answer yes or no.", YES if c else NO)
                for o, c in zip(r["options"], r["correct"])]
    if k == "yn":
        return [(f"{r['question']}\nAnswer yes or no.", YES if r["answer"] else NO)]
    if k == "word":
        return [(f"{r['question']}\nAnswer with a single word or number.", r["answer"])]
    if k == "span":
        return [(f'When does this happen in the video: "{r["question"].rstrip(".")}"?\n'
                 "Answer with the start and end time in seconds, as 'start - end'.",
                 f"{r['span'][0]:.1f} - {r['span'][1]:.1f}")]
    return [(r["question"], r["answer"])]


def load_row_video(r, n_frames):
    """The row's video file, sampled to ``n_frames``."""
    from openjev.video import load_video
    with storage.drive_guard():
        return load_video(storage.path(r["video"]), n_frames=n_frames)


def game_clip(frames, fps):
    """Consecutive game frames as a sampled video (timestamps at the frame midpoints)."""
    from openjev.video import SampledVideo
    n = len(frames)
    return SampledVideo(frames, [(i + 0.5) / fps for i in range(n)], n / fps, frames[0].width, frames[0].height, "", 0)


def parse_span(text):
    m = GROUND.search(text)
    return (float(m.group(1)), float(m.group(2))) if m else None


# ------------------------------------------------------------------ benchmarks
def bench(args):
    import pandas as pd
    storage.require_root()
    vm = pd.read_parquet(raw("videomme", "test.parquet"))
    out = []
    for r in vm[vm.duration == "short"].itertuples():
        opts = [OPT.match(o).group(2) for o in r.options]
        out.append({"set": "videomme_short", "source": "videomme", "video_id": r.videoID, "qid": r.question_id,
                    "type": r.task_type, "question": r.question, "options": opts, "answer_id": LETTERS.index(r.answer),
                    "video": f"vdm/videos/videomme/{r.videoID}.mp4"})
    write(BENCH["videomme_short"], out)
    tc, out2 = pd.read_parquet(raw("tempcompass", "multi-choice.parquet")), []
    for i, r in enumerate(tc.itertuples()):
        lines = r.question.split("\n")
        opts = [OPT.match(l.strip()) for l in lines[1:]]  # one option has a leading space
        out2.append({"set": "tempcompass_mc", "source": "tempcompass", "video_id": r.video_id, "qid": i,
                     "type": r.dim, "question": lines[0], "options": [m.group(2) for m in opts],
                     "answer_id": LETTERS.index(r.answer[0]), "video": f"vdm/videos/tempcompass/{r.video_id}.mp4"})
    write(BENCH["tempcompass_mc"], out2)
    for n, o in (("videomme_short", out), ("tempcompass_mc", out2)):
        print(f"{n}: {len(o)} questions, {len({r['video_id'] for r in o})} videos")


class _HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (reads one zip member without the rest)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _put(member_reader, dest):
    """Copy to ``dest`` via a temporary name, so an interrupted copy is redone next time."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with storage.drive_guard(), member_reader as src, open(dest + ".part", "wb") as f:
        while b := src.read(1 << 22):
            f.write(b)
    os.replace(dest + ".part", dest)


def fetch_bench(args):
    """Video-MME: only the short videos, read out of the 20 remote zips. TempCompass: whole zip."""
    storage.require_root()
    want = {r["video_id"] for r in rows(BENCH["videomme_short"])}
    dest = lambda v: storage.path("vdm", "videos", "videomme", f"{v}.mp4")
    todo = {v for v in want if not os.path.exists(dest(v))}
    print(f"Video-MME short: {len(want) - len(todo)}/{len(want)} already here", flush=True)
    base = f"https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/{VIDEOMME_REV}"
    for i in range(1, 21):
        if not todo:
            break
        z = zipfile.ZipFile(io.BufferedReader(_HttpFile(f"{base}/videos_chunked_{i:02d}.zip"), 1 << 23))
        for m in z.infolist():
            v = os.path.basename(m.filename)[:-4]
            if v in todo:
                _put(z.open(m), dest(v))
                todo.discard(v)
                print(f"chunk {i:02d}: {v} ({m.file_size / 1e6:.0f} MB), {len(todo)} left", flush=True)
    if todo:
        sys.exit(f"{len(todo)} Video-MME short videos not found in any chunk: {sorted(todo)[:5]}")
    zp = raw("tempcompass", "tempcompass_videos.zip")
    if os.path.exists(zp) and sha256(zp) != TEMPCOMPASS_ZIP_SHA:
        os.remove(zp)  # truncated earlier download
    if not os.path.exists(zp):
        url = f"https://huggingface.co/datasets/lmms-lab/TempCompass/resolve/{TEMPCOMPASS_REV}/tempcompass_videos.zip"
        _put(urllib.request.urlopen(url, timeout=120), zp)
        if sha256(zp) != TEMPCOMPASS_ZIP_SHA:  # a dropped connection can end the read early
            os.remove(zp)
            sys.exit("TempCompass zip download was incomplete (checksum mismatch); rerun fetch-bench")
    need = {r["video_id"] for r in rows(BENCH["tempcompass_mc"])}
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            v, ext = os.path.splitext(os.path.basename(m.filename))
            d = storage.path("vdm", "videos", "tempcompass", f"{v}.mp4")
            if v in need and ext == ".mp4" and not os.path.exists(d):
                _put(z.open(m), d)
    print("benchmark videos complete")


# ------------------------------------------------------------------ training mix
def norm_id(v) -> str:
    v = os.path.basename(str(v))
    v = v[:-4] if v.endswith(".mp4") else v
    return v[4:] if v.startswith("ytb_") else v


def eval_ids() -> set[str]:
    missing = [rel for rel in EVAL.values() if not os.path.exists(storage.path(rel))]
    if missing:
        sys.exit(f"evaluation manifests missing (run `bench` first?): {missing}")
    return {norm_id(r["video_id"]) for rel in EVAL.values() for r in rows(rel)}


def nextqa_train():
    vmap = json.load(open(storage.path("nextqa", "annotations", "map_vid_vidorID.json")))
    out = []
    for r in csv.DictReader(open(raw("nextqa", "train.csv"))):
        out.append({"source": "nextqa", "video_id": r["video"], "qid": int(r["qid"]), "type": r["type"],
                    "question": r["question"], "options": [r[f"a{i}"] for i in range(5)],
                    "answer_id": int(r["answer"]), "video": f"vdm/videos/nextqa/{r['video']}.mp4",
                    "zip_member": f"NExTVideo/{vmap[r['video']]}.mp4"})
    return out


def llava_oe(keep):
    out = []
    for r in json.load(open(raw("llava_video", "30_60_s_youtube_oe_v0_1_qa_processed.json"))):
        fname = r["video"].split("/")[-1]
        if fname not in keep:
            continue
        conv = r["conversations"]
        for k in range(0, len(conv) - 1, 2):
            out.append({"source": "llava_video", "video_id": r["id"], "qid": f"oe{k // 2}", "type": "oe",
                        "question": conv[k]["value"].replace("<image>\n", "").strip(),
                        "answer": conv[k + 1]["value"].strip(), "video": f"vdm/videos/llava_video/{fname}"})
    return out


def sample(rs, n, cap, rng, balance=None):
    """Whole videos in random order, <= ``cap`` questions each, until ``n`` questions.

    ``balance``: field whose values are taken in turn within a video (e.g. one per CLEVRER type).
    """
    by = defaultdict(list)
    for r in rs:
        by[r["video_id"]].append(r)
    vids = sorted(by)
    rng.shuffle(vids)
    out = []
    for v in vids:
        if len(out) >= n:
            break
        qs = sorted(by[v], key=lambda r: str(r["qid"]))
        rng.shuffle(qs)
        if balance:
            groups = defaultdict(list)
            for q in qs:
                groups[q[balance]].append(q)
            qs = [groups[g][i] for i in range(max(map(len, groups.values()))) for g in sorted(groups)
                  if i < len(groups[g])]
        out += qs[:min(cap, n - len(out))]
    return out


def mix(args):
    storage.require_root()
    if os.path.exists(storage.path(*MAN, "train_mix.jsonl")) and not args.force:
        sys.exit("vdm/manifests/train_mix.jsonl exists; pass --force to rebuild it")
    held = eval_ids()
    rng = random.Random(SEED)
    pools = {"perception_test": load_manifest("train"), "nextqa": nextqa_train(),
             **{d: rows(f"vdm/manifests/{d}_pool.jsonl") for d in ("star", "clevrer", "charades_sta")}}
    in_chunks = set()
    for n in (1, 2, 3):
        in_chunks |= {l.strip().split("/")[-1] for l in open(raw("llava_video", f"chunk{n}_listing.txt")) if l.strip().endswith(".mp4")}
    mc = [r for r in rows("vdm/manifests/llava_video_pool.jsonl") if r["video"].split("/")[-1] in in_chunks]
    pools["llava_video"] = mc + llava_oe(in_chunks)
    picked, meta = [], {"seed": SEED, "sources": {}}
    for src, (n, cap) in MIX.items():
        pool = [r for r in pools[src] if norm_id(r["video_id"]) not in held]
        dropped = len({r["video_id"] for r in pools[src]}) - len({r["video_id"] for r in pool})
        if n is None:
            got = pool
        else:  # CLEVRER: one question per type in turn; LLaVA-Video: mc and open-ended in turn
            got = sample(pool, n, cap, random.Random(f"{SEED}-{src}"),
                         balance="type" if src in ("clevrer", "llava_video") else None)
        picked += got
        meta["sources"][src] = {"questions": len(got), "videos": len({r["video_id"] for r in got}),
                                "pool_videos_dropped_as_eval": dropped}
    by = defaultdict(list)
    for r in picked:
        by[(r["source"], r["video_id"])].append(r)
    items = []
    for (src, vid), qs in sorted(by.items()):
        rng.shuffle(qs)
        t = [list(x) for q in qs for x in turns(q)]
        items.append({"source": src, "video_id": vid, "video": qs[0]["video"], "questions": len(qs), "turns": t,
                      **({"zip_member": qs[0]["zip_member"]} if "zip_member" in qs[0] else {})})
    rng.shuffle(items)
    write("vdm/manifests/train_mix.jsonl", items)
    meta["total"] = {"questions": len(picked), "videos": len(items), "turns": sum(len(i["turns"]) for i in items)}
    with storage.drive_guard(), open(storage.path(*MAN, "train_mix_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    overlap(args)


def overlap(args):
    """By normalised video ID (YouTube IDs with and without ``ytb_``), every eval set against the mix."""
    storage.require_root()
    train = {norm_id(r["video_id"]): r["source"] for r in rows("vdm/manifests/train_mix.jsonl")}
    bad = 0
    for name, rel in EVAL.items():
        if not os.path.exists(storage.path(rel)):
            print(f"{name:16s} MISSING manifest {rel}")
            bad += 1
            continue
        ids = {norm_id(r["video_id"]) for r in rows(rel)}
        hit = ids & train.keys()
        bad += bool(hit)
        print(f"{name:16s} {len(ids):5d} videos  overlap {len(hit)}" + (f"  e.g. {sorted(hit)[:3]}" if hit else ""))
    print("OK: no evaluation video in the training mix" if not bad else "FAIL")
    if bad:
        sys.exit(2)


# ------------------------------------------------------------------ videos
def extract(args):
    """Copy each needed video out of its archive (skips files already there; resumable)."""
    storage.require_root()
    need = defaultdict(dict)  # source -> {archive member basename or member: dest}
    sets = [rows(rel) for name, rel in EVAL.items() if name not in BENCH]
    for r in [r for s in sets for r in s] + rows("vdm/manifests/train_mix.jsonl"):
        if not r["video"].startswith("vdm/videos/") or os.path.exists(storage.path(r["video"])):
            continue
        src = r["video"].split("/")[2]
        key = r.get("zip_member") or os.path.basename(r["video"])
        need[src][key] = storage.path(r["video"])
    print("to extract:", {s: len(v) for s, v in need.items()}, flush=True)
    zips = {"charades": ("charades/Charades_v1_480.zip", os.path.basename),
            "clevrer": ("clevrer/video_train.zip", os.path.basename),
            "nextqa": ("nextqa/NExTVideo_official.zip", lambda m: m)}
    for src, (zp, key) in zips.items():
        if not need.get(src):
            continue
        with zipfile.ZipFile(raw(*zp.split("/"))) as z:
            for m in z.infolist():
                if (d := need[src].pop(key(m.filename), None)):
                    _put(z.open(m), d)
        print(f"{src}: done", flush=True)
    if need.get("llava_video"):
        for n in (1, 2, 3):
            with tarfile.open(raw("llava_video", f"{LLAVA}_videos_{n}.tar.gz"), "r|gz") as t:
                for m in t:
                    if m.isfile() and (d := need["llava_video"].pop(os.path.basename(m.name), None)):
                        _put(t.extractfile(m), d)
            print(f"llava_video chunk {n}: done", flush=True)
    left = {s: len(v) for s, v in need.items() if v}
    if left:
        sys.exit(f"not found in their archives: {left}")
    print("all videos extracted")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("bench", "fetch-bench", "overlap", "extract"):
        sub.add_parser(n)
    sub.add_parser("mix").add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        {"bench": bench, "fetch-bench": fetch_bench, "mix": mix, "overlap": overlap, "extract": extract}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
