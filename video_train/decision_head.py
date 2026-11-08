"""Step 11: calibration and a decision head on s8_A_ep2 (design in docs/video-training-plan.md).

    python -m video_train.decision_head run                    # every stage below, in order (resumable)
    python -m video_train.decision_head run --limit-videos 3   # smoke test (own adapters, cache, outputs)
    python -m video_train.decision_head calibrate|folds|cache-nextqa|features|train|report

Part A, ``calibrate``: one temperature T on s8_A_ep2's letter log-probabilities, fit by dev NLL.
Dev numbers are 2-fold cross-fitted by video (fit on one half, scored on the other); the shipped T
is fit on all of dev and applied to NExT-QA as an out-of-domain check. Needs only existing rows.

Part B, the decision head:
- ``folds``: two half adapters with s8_A_ep2's recipe (target A, lr 1e-4, 2 passes, rank 8),
  each on half the train videos (``distill --fold k/2``), no dev curves, no dev scoring.
- ``cache-nextqa``: Gemma's vision features for NExT-QA's 115 videos (step 6's cache format).
- ``features``: per question, the final hidden states of every question token, each option's
  token span and the letter logits. Train questions are read by the half adapter that never saw
  them; dev and NExT-QA by s8_A_ep2. Dev letter scores must match s8_A_ep2's dev rows.
- ``train``: the head (``build_head``), 3 seeds, early stop on 10% of train videos.
- ``report``: s8_A_ep2, s8_A_ep2 + T, head, head + T on dev and NExT-QA; the head replaces letter
  scoring only if it beats s8_A_ep2 on dev (paired, more gained than lost, McNemar p < 0.05).

Features: ``<LOCAL_CACHE>/step11/<set>/<video_id>.npz`` (~3 GB). Adapters:
``checkpoints/step11/<tag>/``. Outputs: ``results/step11/``; report ``results/step11/summary.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import storage
from .history import log_run
from .perception_data import load_manifest
from .probe import _by_video, _save_npz

STEP = "step11"
FULL = "s8_A_ep2"                                   # the model being calibrated / extended (checkpoints/step8)
HALVES = {"s11_half0": "0/2", "s11_half1": "1/2"}   # tag -> fold it trains on
RECIPE = ["--variant", "A", "--lr", "1e-4", "--epochs", "2", "--rank", "8", "--lora-alpha", "16"]
RANK, LORA_ALPHA = 8, 16.0
SETS = {"train": "train", "dev": "dev", "nextqa": "nextqa_val1000"}  # set -> manifest
FEATURE_TAG = "video140"                            # step 6's vision cache
MIN_AGREE = 0.99
P_WIN = 0.05
SEEDS = (0, 1, 2)
HP = {"width": 256, "heads": 4, "dropout": 0.2, "lr": 3e-4, "weight_decay": 0.05, "batch": 64,
      "epochs": 40, "patience": 5, "holdout": 0.1}


def smoke_tag(tag, smoke):
    return f"{tag}_smoke" if smoke else tag


def adapter_dir(tag, smoke=False):
    from .distill import ckpt_dir
    return ckpt_dir(FULL, "step8") if tag == FULL else ckpt_dir(smoke_tag(tag, smoke), STEP)


def feature_file(set_name, video_id, smoke=False):
    return os.path.join(storage.LOCAL_CACHE, smoke_tag(STEP, smoke), set_name, f"{video_id}.npz")


def out_dir(smoke=False):
    return storage.path("results", smoke_tag(STEP, smoke))


def calibration_file():
    return os.path.join(out_dir(), "calibration.json")


def head_file(seed, smoke=False):
    return os.path.join(out_dir(smoke), "head", f"seed{seed}.json")


def _questions(set_name):
    return load_manifest(SETS[set_name])


# ------------------------------------------------------------------ part A: temperature
def log_probs(rows, keys):
    """(n, K) log-probabilities from evaluate-style rows (probabilities floored at 1e-12)."""
    return np.log(np.maximum(np.array([rows[k]["probs"] for k in keys], dtype=np.float64), 1e-12))


def tempered(logp, T):
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, y):
    return float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))))


def fit_temperature(logp, y):
    """T > 0 minimising NLL of softmax(logp / T); searched over log T in [log 0.05, log 20]."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: nll(tempered(logp, math.exp(lt)), y), bounds=(math.log(0.05), math.log(20)),
                        method="bounded", options={"xatol": 1e-5})
    return float(math.exp(r.x))


def with_probs(rows, keys, probs):
    """Copies of ``rows`` with new probabilities (and the answers they imply)."""
    out = {}
    for k, p in zip(keys, probs):
        r = dict(rows[k], probs=[float(x) for x in p], pred=int(np.argmax(p)))
        r["correct"] = r["pred"] == r["answer_id"]
        out[k] = r
    return out


def crossfit(rows):
    """(rows with 2-fold cross-fitted tempered probabilities, [T of fold 0, T of fold 1], T on all).

    Folds are by video (``distill.fold_of``), so no video's questions are scored with a T fit on
    that same video."""
    from .distill import fold_of
    keys = sorted(rows)
    logp = log_probs(rows, keys)
    y = np.array([rows[k]["answer_id"] for k in keys])
    folds = fold_of([k[0] for k in keys], 2)
    f = np.array([folds[k[0]] for k in keys])
    probs, Ts = np.empty_like(logp), []
    for i in (0, 1):
        T = fit_temperature(logp[f != i], y[f != i]) if (f != i).any() else 1.0  # tiny smoke sets only
        probs[f == i] = tempered(logp[f == i], T)
        Ts.append(T)
    return with_probs(rows, keys, probs), Ts, fit_temperature(logp, y)


def apply_temperature(rows, T):
    keys = sorted(rows)
    return with_probs(rows, keys, tempered(log_probs(rows, keys), T))


def calibrate(args):
    """Fit T on s8_A_ep2's dev rows; write results/step11/calibration.json."""
    from .evaluate import load_results, metrics
    storage.require_root()
    dev, nq = load_results("dev", FULL), load_results("nextqa", FULL)
    cv, Ts, T = crossfit(dev)
    keep = ("accuracy", "nll", "brier", "ece", "mean_confidence", "coverage_conf_ge_0.8", "accuracy_conf_ge_0.8")

    def m(rows):
        mt = metrics(list(rows.values()))
        return {k: mt[k] for k in keep}

    out = {"T": T, "fold_T": Ts, "dev_raw": m(dev), "dev_crossfit": m(cv), "nextqa_raw": m(nq),
           "nextqa_T": m(apply_temperature(nq, T))}
    out["kept"] = out["dev_crossfit"]["ece"] < out["dev_raw"]["ece"] and out["dev_crossfit"]["nll"] < out["dev_raw"]["nll"]
    os.makedirs(out_dir(), exist_ok=True)
    with storage.drive_guard(), open(calibration_file(), "w") as f:
        json.dump(out, f, indent=1)
    print(f"T = {T:.3f} (folds {Ts[0]:.3f}, {Ts[1]:.3f}); dev ECE {out['dev_raw']['ece']:.3f} -> "
          f"{out['dev_crossfit']['ece']:.3f}, NLL {out['dev_raw']['nll']:.3f} -> {out['dev_crossfit']['nll']:.3f}; "
          f"NExT-QA ECE {out['nextqa_raw']['ece']:.3f} -> {out['nextqa_T']['ece']:.3f}", flush=True)
    log_run(STEP, "eval", {"what": "temperature", "tag": FULL}, {"manifest": "dev"}, None, out)
    return out


# ------------------------------------------------------------------ part B: half adapters
def folds(args):
    from . import distill
    storage.require_root()
    for tag, fold in HALVES.items():
        argv = ["train", *RECIPE, "--tag", smoke_tag(tag, args.limit_videos), "--step", STEP, "--fold", fold,
                "--no-eval", "--dev-every", "0"]
        if args.limit_videos:
            argv += ["--limit-videos", str(args.limit_videos)]
        distill.main(argv)


def cache_nextqa(args):
    from . import cache_features
    argv = ["run", "--sets", "nextqa", "--tag", FEATURE_TAG, "--step", STEP]
    if args.limit_videos:
        argv += ["--limit-videos", str(args.limit_videos)]
    cache_features.main(argv)


# ------------------------------------------------------------------ part B: features
def reader_of(set_name, video_id, folds_map):
    """The adapter that reads a question: s8_A_ep2 for dev / NExT-QA; for train, the half
    adapter that did not train on the video."""
    if set_name != "train":
        return FULL
    return "s11_half1" if folds_map[video_id] == 0 else "s11_half0"


def _todo(limit_videos):
    """{adapter tag: [(set, video_id, questions)]} still without a feature file."""
    from .distill import fold_of
    smoke = bool(limit_videos)
    out = {}
    for s in SETS:
        by = _by_video(_questions(s))
        fm = fold_of(list(by), 2) if s == "train" else {}
        vids = list(by)
        if limit_videos:  # train: the first n videos of each fold; others: the first n
            vids = ([v for v in by if fm[v] == 0][:limit_videos] + [v for v in by if fm[v] == 1][:limit_videos]
                    if s == "train" else vids[:limit_videos])
        for v in vids:
            if not os.path.exists(feature_file(s, v, smoke)):
                out.setdefault(reader_of(s, v, fm), []).append((s, v, by[v]))
    return out


def question_tokens(enc, prompt, n_options):
    """(token ids of the question text after the video prefix, option spans as token indices)."""
    from .cache_features import suffix_ids
    from .probe import option_spans
    t = enc.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = suffix_ids(enc, prompt)
    if list(t["input_ids"]) != list(ids):
        raise SystemExit(f"question tokens differ from the scoring suffix for {prompt[:60]!r}")
    return ids, option_spans(prompt, n_options, t["offset_mapping"])


def features(args):
    import mlx.core as mx
    from . import mlx_model as mm
    from .cache_features import FORMAT, feature_path, load
    from .evaluate import prompt_of
    from .inputs import Gemma4Inputs
    from .models import CONFIG, local_model
    storage.require_root()
    smoke = bool(args.limit_videos)
    todo = _todo(args.limit_videos)
    missing = [v for jobs in todo.values() for _, v, _ in jobs if not os.path.exists(feature_path(FEATURE_TAG, v))]
    if missing:
        raise SystemExit(f"{len(missing)} videos have no vision cache (e.g. {missing[0]}): run `cache-nextqa` / step 6")
    for tag in todo:
        if not os.path.exists(os.path.join(adapter_dir(tag, smoke), "adapter.safetensors")):
            raise SystemExit(f"adapter {tag} is not trained: run `folds` first")
    print("features: " + ", ".join(f"{t} {len(j)} videos" for t, j in todo.items()) or "features: nothing to do",
          flush=True)
    if not todo:
        return
    awake = storage.keep_awake()
    enc = Gemma4Inputs(CONFIG)
    base = mm.load_model(local_model(FORMAT["model"]))
    model = mm.attach_lora(base, rank=RANK, alpha=LORA_ALPHA, checkpoint=False)
    for tag, jobs in todo.items():
        model.load_weights(list(mx.load(os.path.join(adapter_dir(tag, smoke), "adapter.safetensors")).items()),
                           strict=False)
        model.eval()
        t0, n = time.time(), 0
        for i, (s, vid, qs) in enumerate(jobs):
            storage.check_alive()
            feats, prefix, _ = load(FEATURE_TAG, vid)
            toks = [question_tokens(enc, prompt_of(q), len(q["options"])) for q in qs]
            K = {len(q["options"]) for q in qs}
            if len(K) != 1:
                raise SystemExit(f"{vid}: questions with different option counts")
            K = K.pop()
            suffixes = [ids for ids, _ in toks]
            h = mm.grouped_hidden(model, prefix, feats, FORMAT["frames_as"], suffixes)
            letter = mm.last_letter_logits(model, h, suffixes, enc.letter_ids(K))
            mx.eval(h, letter)
            L = h.shape[1]
            spans = np.zeros((len(qs), K, L), dtype=bool)
            for j, (_, sp) in enumerate(toks):
                for k, idx in enumerate(sp):
                    spans[j, k, idx] = True
            _save_npz(feature_file(s, vid, smoke), qid=np.array([q["qid"] for q in qs]),
                  hidden=np.array(h.astype(mx.float16)), length=np.array([len(x) for x in suffixes]),
                  spans=spans, letter=np.array(letter).astype(np.float32),
                  answer_id=np.array([q["answer_id"] for q in qs]), reader=np.array(tag))
            n += len(qs)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"  {tag} [{i + 1}/{len(jobs)} videos] {el / n:.2f}s/question, "
                      f"eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min", flush=True)
    if awake:
        awake.terminate()
    check_features(args)


def check_features(args):
    """Dev letter scores from the features must give s8_A_ep2's dev answers (>= MIN_AGREE)."""
    from .evaluate import load_results
    smoke = bool(args.limit_videos)
    rows = load_results("dev", FULL)
    same = n = 0
    dmax = 0.0
    for vid, qs in _by_video(_questions("dev")).items():
        f = feature_file("dev", vid, smoke)
        if not os.path.exists(f):
            continue
        a = np.load(f)
        p = tempered(a["letter"], 1.0)
        for qid, pi in zip(a["qid"].tolist(), p):
            r = rows[(vid, qid)]
            same += int(np.argmax(pi)) == r["pred"]
            dmax = max(dmax, float(np.abs(pi - np.array(r["probs"])).max()))
            n += 1
    if not n:
        raise SystemExit("no dev features to check")
    print(f"dev features vs {FULL} rows: {same}/{n} same answer ({same / n:.2%}), max prob diff {dmax:.4f}", flush=True)
    if same / n < MIN_AGREE:
        raise SystemExit(f"dev features disagree with {FULL}'s rows ({same / n:.2%} < {MIN_AGREE:.0%})")


def load_features(set_name, smoke=False):
    """[question dict: video_id, qid, hidden (L, H) float16, spans (K, L), letter (K,), answer_id]."""
    out = []
    for vid in _by_video(_questions(set_name)):
        f = feature_file(set_name, vid, smoke)
        if not os.path.exists(f):
            continue
        a = dict(np.load(f))  # read each member once, not once per question
        for j, qid in enumerate(a["qid"].tolist()):
            L = int(a["length"][j])
            out.append({"video_id": vid, "qid": qid, "hidden": a["hidden"][j, :L], "spans": a["spans"][j, :, :L],
                        "letter": a["letter"][j], "answer_id": int(a["answer_id"][j])})
    return out


# ------------------------------------------------------------------ part B: head
def build_head(hidden, hp=HP):
    """Letter scoring plus a learned correction from the question tokens' hidden states.

    utility_k = s * letter_k + MLP([o_k, a_k, last, o_k * last]), where o_k is option k's span mean
    and a_k what (o_k + last) attends to over the question tokens. s starts at 1 and the MLP's last
    layer at zero, so the untrained head is exactly letter scoring.
    """
    import torch
    from torch import nn
    w = hp["width"]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(hp["dropout"]), nn.Linear(hidden, w))
            self.attn = nn.MultiheadAttention(w, hp["heads"], dropout=hp["dropout"], batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Dropout(hp["dropout"]), nn.Linear(w, 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            self.log_s = nn.Parameter(torch.zeros(()))

        def forward(self, h, mask, spans, letter):
            """h (B, L, H), mask (B, L) real tokens, spans (B, K, L), letter (B, K) -> utilities (B, K)."""
            x = self.proj(h)
            sp = spans.float()
            o = (sp @ x) / sp.sum(-1, keepdim=True).clamp(min=1)
            last = x[torch.arange(x.shape[0], device=x.device), mask.sum(1) - 1]
            q = o + last[:, None]
            a, _ = self.attn(q, x, x, key_padding_mask=~mask, need_weights=False)
            lastk = last[:, None].expand_as(o)
            delta = self.mlp(torch.cat([o, a, lastk, o * lastk], -1)).squeeze(-1)
            return self.log_s.exp() * letter + delta

    return Head()


def _batch(rows, idx, device):
    import torch
    rs = [rows[i] for i in idx]
    L, K, H = max(r["hidden"].shape[0] for r in rs), rs[0]["letter"].shape[0], rs[0]["hidden"].shape[1]
    h = np.zeros((len(rs), L, H), dtype=np.float32)
    mask = np.zeros((len(rs), L), dtype=bool)
    spans = np.zeros((len(rs), K, L), dtype=bool)
    for j, r in enumerate(rs):
        n = r["hidden"].shape[0]
        h[j, :n], mask[j, :n], spans[j, :, :n] = r["hidden"], True, r["spans"]
    t = lambda a: torch.from_numpy(a).to(device)
    return (t(h), t(mask), t(spans), t(np.stack([r["letter"] for r in rs]).astype(np.float32)),
            t(np.array([r["answer_id"] for r in rs])))


def holdout_videos(video_ids, frac):
    """A fixed, seed-independent ~frac of the train videos, for early stopping."""
    from .distill import fold_of
    n = max(2, round(1 / frac))
    return {v for v, f in fold_of(video_ids, n).items() if f == 0}


def train_head(seed, train_rows, evals, hp=HP, log=print):
    """Train one head; return ({set: probs (n, K)}, curve, head seconds per question on the first eval set)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_head(train_rows[0]["hidden"].shape[1], hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    held = holdout_videos({r["video_id"] for r in train_rows}, hp["holdout"])
    fit = [i for i, r in enumerate(train_rows) if r["video_id"] not in held]
    val = [i for i, r in enumerate(train_rows) if r["video_id"] in held] or fit  # empty only in a smoke test
    if len(fit) == 0:
        fit = val

    def run(rows, idx):
        model.eval()
        tot, probs = 0.0, []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                h, m, sp, lt, y = _batch(rows, idx[s:s + 128], device)
                u = model(h, m, sp, lt)
                tot += float(torch.nn.functional.cross_entropy(u, y, reduction="sum"))
                probs.append(torch.softmax(u, -1).float().cpu().numpy())
        if device == "mps":
            torch.mps.synchronize()
        return tot / max(len(idx), 1), np.concatenate(probs)

    v0, _ = run(train_rows, val)
    best, best_state, bad, curve = v0, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0, [v0]
    for ep in range(hp["epochs"]):
        model.train()
        order = rng.permutation(fit)
        for s in range(0, len(order), hp["batch"]):
            h, m, sp, lt, y = _batch(train_rows, order[s:s + hp["batch"]], device)
            loss = torch.nn.functional.cross_entropy(model(h, m, sp, lt), y)
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
    out, per_q = {}, None
    for name, rows in evals.items():
        t0 = time.perf_counter()
        out[name] = run(rows, list(range(len(rows))))[1]
        if per_q is None:
            per_q = (time.perf_counter() - t0) / max(len(rows), 1)
    log(f"head seed {seed}: {len(curve) - 1} epochs, held-out NLL {curve[0]:.3f} (letter scoring) -> "
        f"{best:.3f} at epoch {int(np.argmin(curve))}, s = {float(model.log_s.exp()):.3f}", flush=True)
    return out, curve, per_q


def train(args):
    smoke = bool(args.limit_videos)
    todo = [s for s in SEEDS if not os.path.exists(head_file(s, smoke))]
    print(f"head: {len(todo)} seeds to train", flush=True)
    if not todo:
        return
    tr = load_features("train", smoke)
    evals = {s: load_features(s, smoke) for s in ("dev", "nextqa")}
    evals = {s: r for s, r in evals.items() if r}
    if not tr or "dev" not in evals:
        raise SystemExit("features missing: run `features` first")
    print(f"head: {len(tr)} train / " + " / ".join(f"{len(r)} {s}" for s, r in evals.items()) + " questions",
          flush=True)
    for seed in todo:
        probs, curve, per_q = train_head(seed, tr, evals)
        out = {"seed": seed, "hp": HP, "curve": curve, "head_s_per_question": per_q,
               "rows": {s: [{"video_id": r["video_id"], "qid": r["qid"], "probs": p.tolist()}
                            for r, p in zip(evals[s], probs[s])] for s in evals}}
        os.makedirs(os.path.dirname(head_file(seed, smoke)), exist_ok=True)
        with storage.drive_guard(), open(head_file(seed, smoke), "w") as f:
            json.dump(out, f)
    log_run(STEP, "smoke" if smoke else "train", {"what": "decision head", "hp": HP, "seeds": list(SEEDS)},
            {"train_questions": len(tr), "reader": "other-half adapters"}, None, None)


# ------------------------------------------------------------------ report
def head_rows(set_name, base, smoke=False):
    """Seed-mean head probabilities as evaluate-style rows (fields from ``base``), or None."""
    files = [head_file(s, smoke) for s in SEEDS]
    if not all(os.path.exists(f) for f in files):
        return None, None
    acc, seeds, per_q = {}, [], []
    for f in files:
        d = json.load(open(f))
        per_q.append(d["head_s_per_question"])
        rows = {(r["video_id"], r["qid"]): np.array(r["probs"]) for r in d["rows"].get(set_name, [])}
        seeds.append(rows)
        for k, p in rows.items():
            acc[k] = acc.get(k, 0) + p / len(files)
    keys = sorted(k for k in acc if k in base)
    return with_probs(base, keys, [acc[k] for k in keys]), float(np.mean(per_q))


def verdict(full_rows, head):
    """(state, (gained, lost, p)): 'incomplete' | 'pass' | 'fail' for the head against s8_A_ep2 on dev."""
    from .teacher import paired
    if head is None:
        return "incomplete", None
    g, lost, p = paired(full_rows, head)
    return ("pass" if g > lost and p < P_WIN else "fail"), (g, lost, p)


def report(args):
    from .evaluate import load_results, metrics
    from .teacher import paired
    storage.require_root()
    smoke = bool(args.limit_videos)
    cal_file = os.path.join(out_dir(), "calibration.json")
    if not os.path.exists(cal_file):
        calibrate(args)
    cal = json.load(open(cal_file))
    full = {s: load_results(s, FULL) for s in ("dev", "nextqa")}
    head, per_q = {}, None
    for s in full:
        head[s], pq = head_rows(s, full[s], smoke)
        per_q = per_q if per_q is not None else pq
    if head["dev"] is not None:  # compare on the questions the head scored (all of them outside a smoke test)
        full = {s: {k: full[s][k] for k in head[s]} if head[s] else full[s] for s in full}
    arms = {"s8_A_ep2": full,
            "s8_A_ep2 + T": {"dev": crossfit(full["dev"])[0], "nextqa": apply_temperature(full["nextqa"], cal["T"])}}
    head_T = None
    if head["dev"] is not None:
        cv, _, head_T = crossfit(head["dev"])
        arms["head"] = head
        arms["head + T"] = {"dev": cv, "nextqa": apply_temperature(head["nextqa"], head_T) if head["nextqa"] else None}
    state, test = verdict(full["dev"], head["dev"])
    title = f"# {STEP}: calibration and a decision head on {FULL}" + (" (smoke test)" if smoke else "")
    lines = [title, "",
             f"Dev T rows are 2-fold cross-fitted by video; NExT-QA uses the T fit on all of dev "
             f"({FULL}: T = {cal['T']:.3f}" + (f", head: T = {head_T:.3f}" if head_T else "") + "). "
             "Temperature never changes an answer. The head trains on train questions read by the half adapter "
             "that did not see them; it is scored on features from s8_A_ep2.", ""]
    for s, name in (("dev", "Perception Test dev"), ("nextqa", "NExT-QA val (out of domain)")):
        lines += [f"## {name}", "", "| arm | n | accuracy | NLL | Brier | ECE | conf ≥ 0.8: cov / acc | vs s8_A_ep2 (gained / lost, p) |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm, by in arms.items():
            rows = by.get(s)
            if not rows:
                continue
            m = metrics(list(rows.values()))
            g, lost, p = paired(full[s], rows)
            hi = "-" if m["accuracy_conf_ge_0.8"] is None else f"{m['accuracy_conf_ge_0.8']:.1%}"
            lines.append(f"| {arm} | {m['n']} | {m['accuracy']:.1%} ± {m['ci95']:.1%} | {m['nll']:.3f} | "
                         f"{m['brier']:.3f} | {m['ece']:.3f} | {m['coverage_conf_ge_0.8']:.0%} / {hi} | "
                         f"{'-' if arm.startswith('s8') else f'{g} / {lost} ({p:.2g})'} |")
        lines.append("")
    if head["dev"] is not None:
        areas = sorted({r["area"] for r in full["dev"].values()})
        lines += ["## Dev accuracy by area", "", "| area | n | s8_A_ep2 | head |", "|---|---|---|---|"]
        for a in areas:
            ks = [k for k, r in full["dev"].items() if r["area"] == a]
            lines.append(f"| {a} | {len(ks)} | {np.mean([full['dev'][k]['correct'] for k in ks]):.1%} | "
                         f"{np.mean([head['dev'][k]['correct'] for k in ks]):.1%} |")
        lines += ["", f"Head compute: {per_q * 1000:.2f} ms per question on top of s8_A_ep2 (MPS, batch 128).", ""]
    lines += ["## Verdict", ""]
    lines.append(f"- **Calibration: {'kept' if cal['kept'] else 'not kept'}.** Dev ECE {cal['dev_raw']['ece']:.3f} -> "
                 f"{cal['dev_crossfit']['ece']:.3f}, NLL {cal['dev_raw']['nll']:.3f} -> {cal['dev_crossfit']['nll']:.3f} "
                 f"(cross-fitted); NExT-QA ECE {cal['nextqa_raw']['ece']:.3f} -> {cal['nextqa_T']['ece']:.3f}.")
    if state == "incomplete":
        lines.append("- Head: not trained yet.")
    else:
        g, lost, p = test
        lines.append(f"- **Head: {'replaces letter scoring' if state == 'pass' else 'not kept'}.** Against s8_A_ep2 on dev: "
                     f"gains {g}, loses {lost} (net {g - lost:+d}, p = {p:.2g}; bar: more gained than lost, p < {P_WIN}).")
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir(smoke), exist_ok=True)
    with storage.drive_guard(), open(os.path.join(out_dir(smoke), "summary.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {os.path.join(out_dir(smoke), 'summary.md')}")
    if not smoke:
        log_run(STEP, "eval", {"what": "report"}, {"manifest": "dev"}, None,
                {"calibration": cal, "head": state, "test": test})


def run(args):
    storage.require_root()
    calibrate(args)
    folds(args)
    cache_nextqa(args)
    features(args)
    train(args)
    report(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": run, "calibrate": calibrate, "folds": folds, "cache-nextqa": cache_nextqa, "features": features,
            "train": train, "report": report}
    for c in cmds:
        p = sub.add_parser(c)
        p.add_argument("--limit-videos", type=int, default=0, help="smoke test: first N videos per set (and fold)")
    args = ap.parse_args(argv)
    try:
        cmds[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
