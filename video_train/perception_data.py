"""Step 1: Perception Test multiple-choice data -> train / dev / test manifests and videos.

    python -m video_train.perception_data split            # manifests + counts (seconds)
    python -m video_train.perception_data fetch --set dev   # videos for one set (resumable)
    python -m video_train.perception_data fetch --set all
    python -m video_train.perception_data check            # overlap / content-hash / decode checks
    python -m video_train.perception_data grow-dev --questions 2000   # more dev from unused videos

- train: every multiple-choice question of the Perception Test *train* split.
- validation is split **by video** (seeded): test (~2,000 questions, always including the videos
  of the earlier 300-question subset), dev (~500), and the rest is listed as unused and never read.
- The test split of Perception Test has no public answers and is not used.

Everything is written under ``video_train.storage.DATA_ROOT``:
``perception_test/annotations``, ``perception_test/videos/{train,valid}/<id>.mp4``,
``manifests/{train,dev,test}.jsonl`` (+ ``unused_valid_videos.txt``, ``split_meta.json``),
``results/step1/summary.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.request
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import storage
from .history import log_run

BASE = "https://storage.googleapis.com/dm-perception-test/zip_data"
ANN = {s: f"{BASE}/mc_question_{s}_annotations.zip" for s in ("train", "valid")}
VIDEOS = {s: f"{BASE}/{s}_videos.zip" for s in ("train", "valid")}
SUBSET = "data/perception_test/subset.jsonl"   # the earlier 300-question zero-shot subset
LOCAL_VIDEOS = "data/perception_test/videos"   # its videos, copied instead of re-downloaded
SETS = ("train", "dev", "test")
AREAS = ("memory", "abstraction", "physics", "semantics")
REASONING = ("descriptive", "explanatory", "predictive", "counterfactual")


def ann_path(split):
    return storage.path("perception_test", "annotations", f"mc_question_{split}.json")


def video_path(split, vid, rel=False):
    p = os.path.join("perception_test", "videos", split, f"{vid}.mp4")
    return p if rel else storage.path(p)


def manifest_path(name):
    return storage.path("manifests", f"{name}.jsonl")


def load_manifest(name):
    with storage.drive_guard():
        return [json.loads(l) for l in open(manifest_path(name))]


def sha256_file(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------- split
def annotations():
    for split, url in ANN.items():
        out = ann_path(split)
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extract(os.path.basename(out), os.path.dirname(out))
    return {s: json.load(open(ann_path(s))) for s in ANN}


def rows_for(ann, split, set_name, vids):
    rows = []
    for v in vids:
        md = ann[v]["metadata"]
        for q in ann[v]["mc_question"]:
            rows.append({"set": set_name, "source": "perception_test", "source_split": split, "video_id": v,
                         "qid": q["id"], "question": q["question"], "options": q["options"],
                         "answer_id": q["answer_id"], "area": q["area"], "reasoning": q["reasoning"],
                         "tags": q.get("tag", []), "duration_s": md["num_frames"] / md["frame_rate"],
                         "video": video_path(split, v, rel=True)})
    return rows


def split(args):
    storage.require_root()
    ann = annotations()
    seed, dev_q, test_q = args.seed, args.dev_questions, args.test_questions
    valid = ann["valid"]
    vids = sorted(v for v, x in valid.items() if x.get("mc_question"))
    nq = {v: len(valid[v]["mc_question"]) for v in vids}
    subset_vids = sorted({json.loads(l)["video_id"] for l in open(SUBSET)})
    missing = set(subset_vids) - set(vids)
    if missing:
        raise SystemExit(f"subset videos missing from validation annotations: {sorted(missing)[:5]}")
    rest = [v for v in vids if v not in set(subset_vids)]
    random.Random(seed).shuffle(rest)
    test, dev, it = list(subset_vids), [], iter(rest)
    while sum(nq[v] for v in test) < test_q:
        test.append(next(it))
    while sum(nq[v] for v in dev) < dev_q:
        dev.append(next(it))
    unused = list(it)
    train = sorted(v for v, x in ann["train"].items() if x.get("mc_question"))
    rows = {"train": rows_for(ann["train"], "train", "train", train),
            "dev": rows_for(valid, "valid", "dev", sorted(dev)),
            "test": rows_for(valid, "valid", "test", sorted(test))}
    os.makedirs(storage.path("manifests"), exist_ok=True)
    with storage.drive_guard():
        for name, rs in rows.items():
            with open(manifest_path(name) + ".tmp", "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            os.replace(manifest_path(name) + ".tmp", manifest_path(name))
        with open(storage.path("manifests", "unused_valid_videos.txt"), "w") as f:
            f.write("\n".join(sorted(unused)) + "\n")
        meta = {"seed": seed, "dev_questions_target": dev_q, "test_questions_target": test_q,
                "subset_videos_in_test": len(subset_vids),
                "annotation_sha256": {s: sha256_file(ann_path(s)) for s in ANN},
                "videos": {"train": len(train), "dev": len(dev), "test": len(test), "unused_valid": len(unused)},
                "questions": {k: len(v) for k, v in rows.items()},
                "unused_valid_questions": sum(nq[v] for v in unused)}
        json.dump(meta, open(storage.path("manifests", "split_meta.json"), "w"), indent=1)
    checks = id_checks(rows, unused, set(subset_vids))
    report = counts_report(rows, meta, checks)
    print(report)
    write_summary(report)
    log_run("1", "data", {"what": "split + manifests", "seed": seed, "dev_q": dev_q, "test_q": test_q},
            {"annotations": meta["annotation_sha256"]}, seed, {**meta, "checks": checks})


def grow_dev(args):
    """Grow dev with whole videos from the unused validation pool; test is untouched.

    The current dev videos stay in dev (so earlier dev results remain a subset), new videos
    are drawn in a seeded order until dev reaches ``--questions``. The previous manifest is
    kept as ``dev_v<n>.jsonl``.
    """
    storage.require_root()
    ann = annotations()["valid"]
    rows = {k: load_manifest(k) for k in SETS}
    with storage.drive_guard():
        unused = [v for v in open(storage.path("manifests", "unused_valid_videos.txt")).read().split() if v]
    dev_vids = sorted({r["video_id"] for r in rows["dev"]})
    n = len(rows["dev"])
    if n >= args.questions:
        raise SystemExit(f"dev already has {n} questions")
    pool = sorted(unused)
    random.Random(args.seed).shuffle(pool)
    added = []
    for v in pool:
        if n >= args.questions:
            break
        added.append(v)
        n += len(ann[v]["mc_question"])
    new_dev = sorted(dev_vids + added)
    left = sorted(set(unused) - set(added))
    backup = manifest_path(f"dev_v{len(rows['dev'])}")
    with storage.drive_guard():
        if not os.path.exists(backup):
            os.replace(manifest_path("dev"), backup)
        rows["dev"] = rows_for(ann, "valid", "dev", new_dev)
        with open(manifest_path("dev") + ".tmp", "w") as f:
            for r in rows["dev"]:
                f.write(json.dumps(r) + "\n")
        os.replace(manifest_path("dev") + ".tmp", manifest_path("dev"))
        with open(storage.path("manifests", "unused_valid_videos.txt"), "w") as f:
            f.write("\n".join(left) + "\n")
        meta = json.load(open(storage.path("manifests", "split_meta.json")))
        meta.setdefault("changes", []).append({"what": "grow dev", "seed": args.seed, "added_videos": len(added),
                                               "from_questions": sum(1 for _ in open(backup)),
                                               "to_questions": len(rows["dev"]), "backup": os.path.basename(backup)})
        meta["videos"].update({"dev": len(new_dev), "unused_valid": len(left)})
        meta["questions"]["dev"] = len(rows["dev"])
        meta["unused_valid_questions"] = sum(len(ann[v]["mc_question"]) for v in left)
        json.dump(meta, open(storage.path("manifests", "split_meta.json"), "w"), indent=1)
    subset_vids = {json.loads(l)["video_id"] for l in open(SUBSET)}
    checks = id_checks(rows, left, subset_vids)
    checks["previous_dev_videos_kept"] = set(dev_vids) <= set(new_dev)
    report = counts_report(rows, meta, checks)
    print(report)
    write_summary(report)
    log_run("1", "data", {"what": "grow dev", "seed": args.seed, "target_questions": args.questions},
            {"backup": os.path.basename(backup)}, args.seed, {**meta, "checks": checks})


def id_checks(rows, unused, subset_vids):
    by_set = {k: {r["video_id"] for r in rs} for k, rs in rows.items()}
    out = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        out[f"video_overlap_{a}_{b}"] = len(by_set[a] & by_set[b])
    out["video_overlap_unused_with_sets"] = len(set(unused) & (by_set["dev"] | by_set["test"]))
    out["subset_videos_all_in_test"] = subset_vids <= by_set["test"]
    qkeys = [(r["video_id"], r["qid"]) for rs in rows.values() for r in rs]
    out["duplicate_question_keys"] = len(qkeys) - len(set(qkeys))
    out["videos_split_across_sets"] = sum(1 for v in set().union(*by_set.values())
                                         if sum(v in s for s in by_set.values()) > 1)
    return out


def counts_report(rows, meta, checks):
    lines = ["# Step 1: Perception Test datasets", "",
             f"Seed {meta['seed']}. Validation split by video; test includes all "
             f"{meta['subset_videos_in_test']} videos of the earlier 300-question subset.", "",
             "| set | videos | questions |", "|---|---:|---:|"]
    for k in SETS:
        lines.append(f"| {k} | {meta['videos'][k]} | {meta['questions'][k]} |")
    lines.append(f"| unused validation | {meta['videos']['unused_valid']} | {meta['unused_valid_questions']} |")
    for field, keys in (("area", AREAS), ("reasoning", REASONING)):
        lines += ["", f"Questions by {field} (share of set):", "",
                  "| " + field + " | " + " | ".join(SETS) + " |", "|---|" + "---:|" * len(SETS)]
        cnt = {k: Counter(r[field] for r in rows[k]) for k in SETS}
        for key in keys:
            lines.append(f"| {key} | " + " | ".join(f"{cnt[k][key]} ({cnt[k][key] / len(rows[k]):.1%})"
                                                     for k in SETS) + " |")
    ans = {k: Counter(r["answer_id"] for r in rows[k]) for k in SETS}
    lines += ["", "Correct-answer position A/B/C: " + "; ".join(
        f"{k} " + "/".join(f"{ans[k][i] / len(rows[k]):.0%}" for i in range(3)) for k in SETS), "",
        "ID checks: " + ", ".join(f"{k}={v}" for k, v in checks.items())]
    return "\n".join(lines) + "\n"


def write_summary(text, name="summary.md"):
    d = storage.path("results", "step1")
    os.makedirs(d, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(d, name), "w") as f:
        f.write(text)


# -------------------------------------------------------------------- fetch
def fetch(args):
    from video_eval.remote_zip import open_remote_zip
    storage.require_root()
    awake = storage.keep_awake()
    sets = SETS if args.set == "all" else (args.set,)
    want = []  # (source split, video id)
    for s in sets:
        want += sorted({(r["source_split"], r["video_id"]) for r in load_manifest(s)})
    want = list(dict.fromkeys(want))
    if args.limit:
        want = want[:args.limit]
    t0, got, nbytes = time.time(), 0, 0
    todo = [(sp, v) for sp, v in want if not os.path.exists(video_path(sp, v))]
    print(f"{len(want)} videos wanted, {len(want) - len(todo)} already present, {len(todo)} to fetch, "
          f"{args.workers} workers", flush=True)
    local_zips = threading.local()

    def fetch_one(sp, v):
        storage.check_alive()
        out = video_path(sp, v)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        local = os.path.join(LOCAL_VIDEOS, f"{v}.mp4")
        with storage.drive_guard():
            if sp == "valid" and os.path.exists(local):
                with open(local, "rb") as f:
                    data = f.read()
            else:
                zips = local_zips.__dict__.setdefault("zips", {})
                for attempt in range(5):
                    try:
                        if sp not in zips:
                            zips[sp] = open_remote_zip(VIDEOS[sp])
                        data = zips[sp].read(f"videos/{v}.mp4")  # CRC-checked by zipfile
                        break
                    except (OSError, zipfile.BadZipFile) as exc:
                        if attempt == 4:
                            raise
                        print(f"  retry {v}: {exc}", flush=True)
                        time.sleep(5 * (attempt + 1))
                        zips.pop(sp, None)
            with open(out + ".part", "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(out + ".part", out)
        return len(data)

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_one, sp, v) for sp, v in todo]
        try:
            for fut in as_completed(futures):
                nbytes += fut.result()
                got += 1
                el = time.time() - t0
                if got % 20 == 0 or got == len(todo):
                    eta = el / got * (len(todo) - got)
                    print(f"[{got}/{len(todo)}] {nbytes / 1e9:.2f} GB, {nbytes / el / 1e6:.1f} MB/s, "
                          f"eta {eta / 60:.0f} min", flush=True)
        except BaseException:
            for fut in futures:
                fut.cancel()
            raise
    el = time.time() - t0
    rec = {"set": args.set, "fetched": got, "bytes": nbytes, "seconds": round(el, 1), "workers": args.workers,
           "MBps": round(nbytes / el / 1e6, 1) if el else None, "limit": args.limit}
    log_run("1", "data", {"what": "fetch videos", **rec}, {}, None, rec)
    if awake:
        awake.terminate()
    print(json.dumps(rec))


# -------------------------------------------------------------------- check
def check(args):
    """Every manifest video exists and decodes; no file content appears in two sets."""
    import av
    storage.require_root()
    rows = {s: load_manifest(s) for s in SETS}
    seen, missing, bad, dup = {}, [], [], []
    hashes = defaultdict(set)
    # sha256 per file, reused while size and mtime are unchanged (hashing 33 GB on the HDD is slow)
    cache_path = storage.path("results", "step1", "video_hashes.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for s, rs in rows.items():
        for sp, v in sorted({(r["source_split"], r["video_id"]) for r in rs}):
            storage.check_alive()
            p = video_path(sp, v)
            if not os.path.exists(p):
                missing.append(v)
                continue
            with storage.drive_guard():
                st_ = os.stat(p)
                key = f"{st_.st_size}:{int(st_.st_mtime)}"
                if cache.get(p, {}).get("key") == key:
                    h = cache[p]["sha256"]
                else:
                    h = sha256_file(p)
                    cache[p] = {"key": key, "sha256": h}
                try:
                    with av.open(p) as c:
                        st = c.streams.video[0]
                        next(c.decode(st))
                except Exception as exc:  # noqa: BLE001
                    bad.append((v, str(exc)[:80]))
            hashes[h].add(s)
            if h in seen and seen[h] != v:
                dup.append((seen[h], v))
            seen[h] = v
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with storage.drive_guard(), open(cache_path, "w") as f:
        json.dump(cache, f)
    cross = sum(1 for sets in hashes.values() if len(sets) > 1)
    res = {"videos_checked": len(seen), "missing": len(missing), "undecodable": len(bad),
           "identical_files_with_different_ids": len(dup), "identical_files_across_sets": cross}
    text = ("# Step 1: video checks\n\n" + "\n".join(f"- {k}: {v}" for k, v in res.items()) + "\n"
            + (f"\nMissing (first 10): {missing[:10]}\n" if missing else "")
            + (f"\nUndecodable: {bad[:10]}\n" if bad else "") + (f"\nDuplicates: {dup[:10]}\n" if dup else ""))
    print(text)
    write_summary(text, "video_checks.md")
    log_run("1", "data", {"what": "video checks"}, {}, None, res)
    if missing or bad or cross:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dev-questions", type=int, default=500)
    s.add_argument("--test-questions", type=int, default=2000)
    f = sub.add_parser("fetch")
    f.add_argument("--set", choices=(*SETS, "all"), default="all")
    f.add_argument("--limit", type=int, default=0, help="only the first N videos (timing sample)")
    f.add_argument("--workers", type=int, default=4)
    sub.add_parser("check")
    g = sub.add_parser("grow-dev")
    g.add_argument("--questions", type=int, default=2000)
    g.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        {"split": split, "fetch": fetch, "check": check, "grow-dev": grow_dev}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
