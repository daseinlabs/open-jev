"""NExT-QA validation: a second, out-of-domain video eval (never used for training).

    python -m video_train.nextqa_data fetch     # annotations + the 570 validation videos -> drive
    python -m video_train.nextqa_data manifest  # manifests/nextqa_val.jsonl (+ a fixed subset)

Annotations: github.com/doc-doc/NExT-QA (MIT), ``dataset/nextqa/val.csv``. Videos: VidOR /
YFCC100M clips; the official Google Drive zip is often over quota, so the validation videos
are read with range requests from the lmms-lab mirror (pinned revision). Their resolution and
frame counts match the official annotations. 5 options per question; question types are
causal (C*), temporal (T*) and descriptive (D*).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict

from . import storage
from .history import log_run

ANN_URL = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/{}"
ANN_SHA256 = {"val.csv": "43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8"}
MIRROR = ("https://huggingface.co/datasets/lmms-lab/NExTQA/resolve/"
          "a0d7729e38399da9c8a70c59aa4ad7f6996d3c00/videos.zip")
TYPES = {"C": "causal", "T": "temporal", "D": "descriptive"}


def ann_path(name="val.csv"):
    return storage.path("nextqa", "annotations", name)


def video_path(vid, rel=False):
    p = os.path.join("nextqa", "videos", "val", f"{vid}.mp4")
    return p if rel else storage.path(p)


def load_val():
    with storage.drive_guard():
        return list(csv.DictReader(open(ann_path())))


def fetch(args):
    from video_eval.remote_zip import open_remote_zip
    storage.require_root()
    awake = storage.keep_awake()
    os.makedirs(os.path.dirname(ann_path()), exist_ok=True)
    for name, sha in ANN_SHA256.items():
        if not os.path.exists(ann_path(name)):
            with urllib.request.urlopen(ANN_URL.format(name), timeout=60) as r, open(ann_path(name), "wb") as f:
                f.write(r.read())
        got = hashlib.sha256(open(ann_path(name), "rb").read()).hexdigest()
        if got != sha:
            raise SystemExit(f"checksum mismatch for {name}: {got}")
    rows = load_val()
    vids = sorted({r["video"] for r in rows})
    todo = [v for v in vids if not os.path.exists(video_path(v))]
    print(f"{len(vids)} validation videos, {len(todo)} to fetch", flush=True)
    z, t0, nbytes = open_remote_zip(MIRROR), time.time(), 0
    meta = {r["video"]: r for r in rows}
    for i, v in enumerate(todo):
        storage.check_alive()
        data = z.read(f"NExTVideo/{v}.mp4")  # CRC-checked
        out = video_path(v)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with storage.drive_guard():
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
        nbytes += len(data)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] {nbytes / 1e9:.2f} GB, {nbytes / (time.time() - t0) / 1e6:.1f} MB/s",
                  flush=True)
    # Same videos as the official release: resolution and frame count must match the annotations.
    import av
    bad = []
    for v in vids:
        with av.open(video_path(v)) as c:
            s = c.streams.video[0]
            want = (int(meta[v]["width"]), int(meta[v]["height"]), int(meta[v]["frame_count"]))
            n = s.frames or sum(1 for _ in c.decode(video=0))  # some headers store no frame count
            if (s.codec_context.width, s.codec_context.height, n) != want:
                bad.append(v)
    res = {"videos": len(vids), "fetched": len(todo), "bytes": nbytes, "metadata_mismatch": len(bad)}
    print(json.dumps(res))
    log_run("nextqa", "data", {"what": "fetch NExT-QA val", "mirror": MIRROR}, {"annotations": ANN_SHA256}, None, res)
    if awake:
        awake.terminate()
    if bad:
        sys.exit(f"{len(bad)} videos do not match the annotation metadata, e.g. {bad[:5]}")


def manifest(args):
    storage.require_root()
    rows = load_val()
    out = []
    for r in rows:
        out.append({"set": "nextqa_val", "source": "nextqa", "video_id": r["video"], "qid": int(r["qid"]),
                    "question": r["question"].strip() + "?", "options": [r[f"a{i}"].strip() for i in range(5)],
                    "answer_id": int(r["answer"]), "type": r["type"], "area": TYPES[r["type"][0]],
                    "duration_s": None, "video": video_path(r["video"], rel=True)})
    out.sort(key=lambda r: (r["video_id"], r["qid"]))
    by_video = defaultdict(list)
    for r in out:
        by_video[r["video_id"]].append(r)
    vids = sorted(by_video)
    random.Random(args.seed).shuffle(vids)
    sub, n = [], 0
    for v in vids:
        if n >= args.subset:
            break
        sub.append(v)
        n += len(by_video[v])
    sub_rows = [r for v in sorted(sub) for r in by_video[v]]
    with storage.drive_guard():
        for name, rs in ((f"nextqa_val", out), (f"nextqa_val{args.subset}", sub_rows)):
            with open(storage.path("manifests", f"{name}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
    for name, rs in (("full", out), (f"subset {args.subset}", sub_rows)):
        c = Counter(r["area"] for r in rs)
        print(f"{name}: {len(rs)} questions, {len({r['video_id'] for r in rs})} videos, "
              + ", ".join(f"{k} {c[k]} ({c[k] / len(rs):.0%})" for k in TYPES.values()))
    log_run("nextqa", "data", {"what": "NExT-QA manifests", "seed": args.seed, "subset": args.subset}, {}, args.seed,
            {"full_questions": len(out), "subset_questions": len(sub_rows), "subset_videos": len(sub)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    m = sub.add_parser("manifest")
    m.add_argument("--subset", type=int, default=1000)
    m.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        {"fetch": fetch, "manifest": manifest}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
