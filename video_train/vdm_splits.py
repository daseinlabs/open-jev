"""Stage 1 of docs/video-decision-model-plan.md: fix the in-domain dev and test sets.

    python -m video_train.vdm_splits make      # writes vdm/manifests/, refuses to overwrite
    python -m video_train.vdm_splits check     # disjointness and counts of what is on the drive

Held out by video (never by question), seed 0, before any training data is sampled. Per
dataset: ``<ds>_dev.jsonl`` and ``<ds>_test.jsonl``, plus ``<ds>_pool.jsonl`` (every other
train question, with held-out videos removed): stage 2 samples training data from the pool
only. At most ``PER_VIDEO`` questions per held-out video, so a set spans many videos.

- Perception Test: the previous plan's ``manifests/dev.jsonl`` (2,003) and ``test.jsonl``
  (2,000) stay as they are (not copied).
- NExT-QA: dev = ``manifests/nextqa_val1000.jsonl`` (as before); test = 900 questions (<= 2 per video) from
  the other NExT-QA val videos. Training uses NExT-QA train (different videos by design).
- STAR and Charades-STA share Charades videos: held-out Charades videos are removed from
  both pools.
- LLaVA-Video (30-60 s YouTube, multiple choice only): dev and test from the videos in
  chunk 1 (``tar`` listing passed with ``--llava-listing``), the only complete chunk.

Rows keep the raw annotation (question, options or answer, clip times); turning them into
prompts is stage 2's conversion. ``video`` is the path under DATA_ROOT where stage 2
extracts the file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

from . import storage
from .perception_data import load_manifest

SEED, PER_VIDEO = 0, 2
SIZES = {"nextqa": 900, "star": 500, "charades_sta": 500, "clevrer": 500, "llava_video": 300}
RAW = ("vdm", "raw")
OUT = ("vdm", "manifests")


def raw(*p):
    return storage.path(*RAW, *p)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def star():
    out = []
    for r in json.load(open(raw("star", "STAR_train.json"))):
        choices = [c["choice"] for c in r["choices"]]
        out.append({"source": "star", "video_id": r["video_id"], "qid": r["question_id"],
                    "type": r["question_id"].split("_")[0], "question": r["question"], "options": choices,
                    "answer_id": choices.index(r["answer"]), "clip": [r["start"], r["end"]],
                    "video": f"vdm/videos/charades/{r['video_id']}.mp4"})
    return out


def charades_sta():
    out = []
    for i, line in enumerate(open(raw("charades_sta", "Charades_sta_train.txt"))):
        head, sentence = line.rstrip("\n").split("##")
        vid, s, e = head.split()
        out.append({"source": "charades_sta", "video_id": vid, "qid": i, "type": "grounding",
                    "question": sentence.strip(), "span": [float(s), float(e)],
                    "video": f"vdm/videos/charades/{vid}.mp4"})
    return out


def clevrer():
    out = []
    for v in json.load(open(raw("clevrer", "train.json"))):
        vid = v["video_filename"][:-4]
        for q in v["questions"]:
            r = {"source": "clevrer", "video_id": vid, "qid": q["question_id"], "type": q["question_type"],
                 "question": q["question"], "video": f"vdm/videos/clevrer/{v['video_filename']}"}
            if "choices" in q:  # multi-answer: each choice is correct or wrong
                r["options"] = [c["choice"] for c in q["choices"]]
                r["correct"] = [c["answer"] == "correct" for c in q["choices"]]
            else:
                r["answer"] = q["answer"]
            out.append(r)
    return out


OPT = re.compile(r"^([A-H])\.\s*(.*)$")


def llava_video(listing):
    in_chunk1 = {l.strip().split("/")[-1] for l in open(listing) if l.strip().endswith(".mp4")}
    out = []
    for r in json.load(open(raw("llava_video", "30_60_s_youtube_mc_v0_1_qa_processed.json"))):
        fname = r["video"].split("/")[-1]
        conv = r["conversations"]
        for k in range(0, len(conv) - 1, 2):
            lines = [l for l in conv[k]["value"].replace("<image>\n", "").split("\n") if l.strip()]
            opts = [OPT.match(l) for l in lines]
            options = [m.group(2) for m in opts if m]
            letter = conv[k + 1]["value"].strip()[:1]
            if not options or letter not in "ABCDEFGH"[:len(options)]:
                continue  # unparseable question: skipped (counted in the meta file)
            out.append({"source": "llava_video", "video_id": r["id"], "qid": k // 2, "type": "mc",
                        "question": "\n".join(l for l, m in zip(lines, opts) if not m and not l.startswith("Please")),
                        "options": options, "answer_id": "ABCDEFGH".index(letter),
                        "video": f"vdm/videos/llava_video/{fname}", "in_chunk1": fname in in_chunk1})
    return out


def hold_out(rows, n, rng, eligible=lambda v: True, exclude=frozenset()):
    """Sample whole videos (<= PER_VIDEO random questions each) until ``n`` questions."""
    by = defaultdict(list)
    for r in rows:
        by[r["video_id"]].append(r)
    vids = sorted(v for v in by if eligible(v) and v not in exclude)
    rng.shuffle(vids)
    picked, taken = [], []
    for v in vids:
        if len(picked) >= n:
            break
        qs = sorted(by[v], key=lambda r: str(r["qid"]))
        picked += rng.sample(qs, min(PER_VIDEO, len(qs)))[:n - len(picked)]
        taken.append(v)
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} questions available, need {n}")
    return picked, set(taken)


def write(name, rows, split):
    with storage.drive_guard(), open(storage.path(*OUT, f"{name}_{split}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"set": f"{name}_{split}", **r}) + "\n")


def make(args):
    storage.require_root()
    os.makedirs(storage.path(*OUT), exist_ok=True)
    if os.path.exists(storage.path(*OUT, "split_meta.json")):
        sys.exit("vdm/manifests/split_meta.json exists: the splits are fixed; not overwriting")
    meta = {"seed": SEED, "per_video": PER_VIDEO, "sizes": SIZES, "counts": {}, "inputs": {}}
    sets = {}

    # NExT-QA: dev is the previous dev subset; test from the remaining val videos.
    val, dev = load_manifest("nextqa_val"), load_manifest("nextqa_val1000")
    rng = random.Random(f"{SEED}-nextqa")
    test, _ = hold_out(val, SIZES["nextqa"], rng, exclude={r["video_id"] for r in dev})
    sets["nextqa"] = {"dev": dev, "test": test, "pool": None}  # train pool: NExT-QA train.csv (no overlap: other videos)

    # STAR + Charades-STA: hold out Charades videos jointly.
    st, sta = star(), charades_sta()
    held = set()
    for name, rows in (("star", st), ("charades_sta", sta)):
        rng = random.Random(f"{SEED}-{name}")
        d, dv = hold_out(rows, SIZES[name], rng, exclude=held)
        t, tv = hold_out(rows, SIZES[name], rng, exclude=held | dv)
        held |= dv | tv
        sets[name] = {"dev": d, "test": t}
    for name, rows in (("star", st), ("charades_sta", sta)):
        sets[name]["pool"] = [r for r in rows if r["video_id"] not in held]

    llava = llava_video(args.llava_listing)
    chunk1 = {r["video_id"] for r in llava if r["in_chunk1"]}
    meta["inputs"]["llava_chunk1_videos_with_mc"] = len(chunk1)
    for name, rows, ok in (("clevrer", clevrer(), lambda v: True), ("llava_video", llava, chunk1.__contains__)):
        rng = random.Random(f"{SEED}-{name}")
        d, dv = hold_out(rows, SIZES[name], rng, eligible=ok)
        t, tv = hold_out(rows, SIZES[name], rng, eligible=ok, exclude=dv)
        sets[name] = {"dev": d, "test": t, "pool": [r for r in rows if r["video_id"] not in dv | tv]}

    for name, s in sets.items():
        meta["counts"][name] = {}
        for split, rows in s.items():
            if rows is None:
                continue
            if not (name == "nextqa" and split == "dev"):
                write(name, rows, split)
            meta["counts"][name][split] = {"questions": len(rows), "videos": len({r["video_id"] for r in rows})}
    for p in ("star/STAR_train.json", "charades_sta/Charades_sta_train.txt", "clevrer/train.json",
              "llava_video/30_60_s_youtube_mc_v0_1_qa_processed.json", "nextqa/train.csv"):
        meta["inputs"][p] = sha256(raw(*p.split("/")))
    meta["nextqa_dev"] = "manifests/nextqa_val1000.jsonl"
    meta["perception_test"] = {"dev": "manifests/dev.jsonl", "test": "manifests/test.jsonl",
                               "pool": "manifests/train.jsonl"}
    with storage.drive_guard(), open(storage.path(*OUT, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta["counts"], indent=1))
    check(args)


def _load(name, split):
    p = storage.path(*OUT, f"{name}_{split}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def check(args):
    """No held-out video in any pool or across dev/test; Charades videos checked across STAR and STA."""
    storage.require_root()
    vids = lambda rows: {(r["video"]) for r in rows}
    held, pools, bad = set(), set(), []
    for name in SIZES:
        dev = load_manifest("nextqa_val1000") if name == "nextqa" else _load(name, "dev")
        test, pool = _load(name, "test"), _load(name, "pool")
        if vids(dev) & vids(test):
            bad.append(f"{name}: dev and test share videos")
        held |= vids(dev) | vids(test)
        pools |= vids(pool)
        print(f"{name:13s} dev {len(dev):5d} q / {len(vids(dev)):4d} v   test {len(test):5d} q / "
              f"{len(vids(test)):4d} v   pool {len(pool):6d} q / {len(vids(pool)):5d} v")
    nextqa_train = {r["video"] for r in csv.DictReader(open(raw("nextqa", "train.csv")))}
    if {v.split("/")[-1][:-4] for v in held if v.startswith("nextqa/")} & nextqa_train:
        bad.append("nextqa: held-out video in train.csv")
    if held & pools:
        bad.append(f"{len(held & pools)} held-out videos appear in a pool")
    print("OK: held-out videos are disjoint from every pool" if not bad else "FAIL: " + "; ".join(bad))
    if bad:
        sys.exit(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--llava-listing", required=True, help="file listing chunk 1 (tar tzf ... > file)")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    try:
        {"make": make, "check": check}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
