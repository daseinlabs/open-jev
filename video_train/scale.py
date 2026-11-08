"""Step 8: scale the step 7 winner (target A, lr 1e-4). Two runs, each one change against s7_A:

    python -m video_train.scale check                  # ~10 min: NExT-QA scoring path vs training's (20 dev videos)
    python -m video_train.scale run                    # check, NExT-QA for s7_A, then each run: train, dev, NExT-QA
    python -m video_train.scale run --only s8_A_r32    # one run
    python -m video_train.scale report                 # dev + NExT-QA against zero-shot and s7_A, curves, verdict

Runs (everything else as in step 7: all 7,392 train questions, step 6 cache, seed 0, 8 videos per
optimizer step, AdamW, 5% warmup then cosine decay over the whole run, lr 1e-4, answer-only loss):
- ``s8_A_r32``: 1 pass, LoRA rank 32 / alpha 64 instead of 8 / 16. alpha / rank stays 2, so only
  capacity changes, not the update scale. Runs first: it is the shorter one (~7-10 h).
- ``s8_A_ep2``: 2 passes (490 steps), rank 8 / alpha 16. A fresh run with the schedule stretched over
  both passes, not a continuation of s7_A (whose rate has already decayed to zero). ~12-19 h.

Decision (step 1's small-effects rule): a run replaces s7_A only if it beats s7_A on dev, paired,
with more gained than lost and McNemar p < 0.05; otherwise s7_A stays, being the cheaper setting.
If both pass, the more accurate one wins. NExT-QA (out of domain) is reported next to dev, as at
every step; it informs, it does not gate.

``check`` scores 20 dev videos through ``evaluate.py --adapter`` (fresh video decode, adapter
installed per request as in serving) and requires >= 98% answer agreement with s7_A's dev rows
(cached features, adapter attached as in training), so NExT-QA numbers come from the same model.

Checkpoints: ``checkpoints/step8/<tag>/``. Rows: ``results/evals/{dev,nextqa}/<tag>.jsonl``.
Report: ``results/step8/summary.md``. All resumable: rerun the same command after a stop.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import storage
from .history import log_run

STEP = "step8"
S7 = "s7_A_lr1e-04"  # the step 7 winner, in checkpoints/step7/
LR = "1e-4"
RUNS = {  # tag -> distill train flags; one change each against s7_A
    "s8_A_r32": ["--rank", "32", "--lora-alpha", "64"],
    "s8_A_ep2": ["--epochs", "2"],
}
CHECK_TAG = "s7_A_adapter_check"
CHECK_VIDEOS = 20
P_WIN = 0.05


def adapter_dir(tag):
    from .distill import ckpt_dir
    return ckpt_dir(tag, "step7" if tag == S7 else STEP)


def _evaluate(argv):
    from . import evaluate
    evaluate.main(argv)


def check(args):
    from .evaluate import fresh_check
    fresh_check(S7, CHECK_TAG, adapter_dir(S7), STEP, CHECK_VIDEOS)


def nextqa(tag):
    _evaluate(["run", "--set", "nextqa", "--tag", tag, "--soft-tokens", "140", "--adapter", adapter_dir(tag),
               "--step", STEP])


def run(args):
    from . import distill
    storage.require_root()
    check(args)
    nextqa(S7)
    for tag in [args.only] if args.only else RUNS:
        distill.main(["train", "--variant", "A", "--lr", LR, "--tag", tag, "--step", STEP, *RUNS[tag]])
        nextqa(tag)


# ------------------------------------------------------------------ report
def verdict(s7_rows, runs):
    """(winning tag or None, {tag: (gained, lost, p)}) for complete runs against s7_A on dev."""
    from .evaluate import metrics
    from .teacher import paired
    tests = {tag: paired(s7_rows, rows) for tag, rows in runs.items()}
    passed = [t for t, (g, lost, p) in tests.items() if g > lost and p < P_WIN]
    win = max(passed, key=lambda t: metrics(list(runs[t].values()))["accuracy"]) if passed else None
    return win, tests


def report(args):
    from .distill import BASE, TEACHER, _curve_lines, _pct
    from .evaluate import load_results, metrics, result_path
    from .teacher import paired
    storage.require_root()

    def complete(set_name, tag, n):
        if not os.path.exists(result_path(set_name, tag)):
            return None
        rows = load_results(set_name, tag)
        return rows if len(rows) == n else None

    zero = {"dev": load_results("dev", BASE), "nextqa": load_results("nextqa", BASE)}
    tags = [S7, *RUNS]
    res = {s: {t: complete(s, t, len(zero[s])) for t in tags} for s in zero}
    if not any(res["dev"][t] for t in RUNS):
        raise SystemExit("no complete step 8 dev results yet")
    s7 = res["dev"][S7]
    lines = [f"# {STEP}: scaling the step 7 winner ({S7})", "",
             "One change per run against s7_A; a run replaces it only if it beats s7_A on dev "
             f"(paired, more gained than lost, p < {P_WIN}).", ""]
    lines += _curve_lines({t: adapter_dir(t) for t in tags if os.path.exists(adapter_dir(t))},
                          load_results("dev", TEACHER), zero["dev"])
    summary = {}
    for s, title in (("dev", "Perception Test dev"), ("nextqa", "NExT-QA val (1,003 questions, out of domain)")):
        base, ref = zero[s], res[s][S7]
        mb = metrics(list(base.values()))
        lines += ["", f"## {title}", "",
                  "| | accuracy | net vs zero-shot (p) | net vs s7_A (p) | NLL | ECE | conf ≥ 0.8: cov / acc |",
                  "|---|---|---|---|---|---|---|",
                  f"| zero-shot ({BASE}) | {mb['accuracy']:.1%} ± {mb['ci95']:.1%} | - | - | {mb['nll']:.3f} | "
                  f"{mb['ece']:.3f} | {mb['coverage_conf_ge_0.8']:.0%} / {_pct(mb['accuracy_conf_ge_0.8'])} |"]
        for t in tags:
            rows = res[s][t]
            if rows is None:
                lines.append(f"| {t} | not run | | | | | |")
                continue
            m = metrics(list(rows.values()))
            g0, l0, p0 = paired(base, rows)
            vs = "-" if t == S7 or ref is None else "{:+d} ({:.2g})".format(*_net(paired(ref, rows)))
            lines.append(f"| {t} | {m['accuracy']:.1%} ± {m['ci95']:.1%} | {g0 - l0:+d} ({p0:.2g}) | {vs} | "
                         f"{m['nll']:.3f} | {m['ece']:.3f} | "
                         f"{m['coverage_conf_ge_0.8']:.0%} / {_pct(m['accuracy_conf_ge_0.8'])} |")
            summary.setdefault(t, {})[s] = {k: v for k, v in m.items() if not isinstance(v, dict)}
        if s == "dev":
            done = [t for t in tags if res[s][t]]
            lines += ["", "| area | n | zero-shot | " + " | ".join(done) + " |", "|---|---|---|" + "---|" * len(done)]
            by = {t: metrics(list(res[s][t].values()))["by_area"] for t in done}
            for area, v in mb["by_area"].items():
                lines.append(f"| {area} | {v['n']} | {v['accuracy']:.1%} | "
                             + " | ".join(f"{by[t][area]['accuracy']:.1%}" for t in done) + " |")
    runs = {t: res["dev"][t] for t in RUNS if res["dev"][t]}
    win, tests = verdict(s7, runs)
    lines += ["", "## Verdict", ""]
    for t, (g, lost, p) in tests.items():
        lines.append(f"- {t} vs s7_A on dev: gains {g}, loses {lost} (net {g - lost:+d}, p={p:.2g})")
    missing = [t for t in RUNS if t not in runs]
    lines.append("")
    lines.append(f"**{win} replaces s7_A.**" if win else
                 "**s7_A stays** (no run beats it on dev)" + (f"; still to run: {', '.join(missing)}" if missing else "")
                 + ".")
    text = "\n".join(lines) + "\n"
    out = storage.path("results", STEP)
    os.makedirs(out, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    log_run(STEP, "eval", {"what": "report"}, {"manifest": "dev,nextqa"}, None,
            {"winner": win or S7, "vs_s7": {t: dict(zip(("gained", "lost", "p"), v)) for t, v in tests.items()},
             **summary})


def _net(test):
    g, lost, p = test
    return g - lost, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--only", choices=RUNS, default=None, help="one run instead of both")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    try:
        {"check": check, "run": run, "report": report}[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
