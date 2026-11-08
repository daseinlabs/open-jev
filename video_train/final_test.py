"""Step 12: the one test-set evaluation (design in docs/video-training-plan.md).

    python -m video_train.final_test run            # check, then score test (zero-shot, s8_A_ep2), report
    python -m video_train.final_test check|score|report
    python -m video_train.final_test run --smoke    # the same path on 2 dev videos; never touches test

Model: s8_A_ep2 with step 11's temperature (T from ``results/step11/calibration.json``), and
zero-shot ``video140`` for the gain. Both on fresh video decodes through ``evaluate.py run``
(MLX 8-bit, the serving path with the adapter installed). ``check`` first requires s8_A_ep2 on 20 dev
videos through that path to agree with its dev rows on >= 98% of answers.

Rows: ``results/evals/test/<tag>.jsonl`` (resumable). Report: ``results/step12/summary.md``.
Nothing is gated or tuned on the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import storage
from .history import log_run

STEP = "step12"
FULL = "s8_A_ep2"
ZERO = "video140"
CHECK_TAG = "s8_A_ep2_fresh_check"
CHECK_VIDEOS = 20
SMOKE_VIDEOS = 2
D3_MINI = 0.703  # self-reported, its own protocol
DEV_RESOLUTION = 0.02


def adapter_dir():
    from . import decision_head
    return decision_head.adapter_dir(decision_head.FULL)


def arms(smoke=False):
    """{tag: evaluate.py run flags}. A smoke test scores dev under its own tags."""
    sfx = "_step12smoke" if smoke else ""
    return {ZERO + sfx: ["--soft-tokens", "140"],
            FULL + sfx: ["--soft-tokens", "140", "--adapter", adapter_dir()]}


def target_set(smoke=False):
    return "dev" if smoke else "test"


def temperature():
    from .decision_head import calibration_file
    with storage.drive_guard():
        return json.load(open(calibration_file()))["T"]


def _evaluate(argv):
    from . import evaluate
    evaluate.main(argv)


def check(args):
    from .evaluate import fresh_check
    fresh_check(FULL, CHECK_TAG, adapter_dir(), STEP, CHECK_VIDEOS)


def score(args):
    storage.require_root()
    s = target_set(args.smoke)
    for tag, flags in arms(args.smoke).items():
        argv = ["run", "--set", s, "--tag", tag, *flags, "--step", STEP]
        if args.smoke:
            argv += ["--limit-videos", str(SMOKE_VIDEOS)]
        else:
            argv += ["--final"]
        _evaluate(argv)


def summarize(zero, full, T):
    """Metrics for zero-shot, s8_A_ep2 and s8_A_ep2 + T on shared questions, and the paired test."""
    from .decision_head import apply_temperature
    from .evaluate import metrics
    from .teacher import paired
    keys = sorted(set(zero) & set(full))
    zero, full = {k: zero[k] for k in keys}, {k: full[k] for k in keys}
    cal = apply_temperature(full, T)
    return ({"zero-shot": metrics(list(zero.values())), FULL: metrics(list(full.values())),
             f"{FULL} + T": metrics(list(cal.values()))}, paired(zero, full))


def report(args):
    from .decision_head import calibration_file
    from .evaluate import load_results, result_path
    storage.require_root()
    s, tags = target_set(args.smoke), list(arms(args.smoke))
    for t in tags:
        if not os.path.exists(result_path(s, t)):
            raise SystemExit(f"no {s} rows for {t}: run `score` first")
    n_expected = 2000 if not args.smoke else None
    rows = {t: load_results(s, t) for t in tags}
    if n_expected and any(len(r) != n_expected for r in rows.values()):
        raise SystemExit(f"test incomplete: {[len(r) for r in rows.values()]} of {n_expected}; rerun `score`")
    T = temperature()
    m, (g, lost, p) = summarize(rows[tags[0]], rows[tags[1]], T)
    dev, _ = summarize(load_results("dev", ZERO), load_results("dev", FULL), T)
    name = "Perception Test dev, 2 videos (smoke test)" if args.smoke else "Perception Test test (2,000 questions, scored once)"
    lines = [f"# {STEP}: the final evaluation" + (" (smoke test)" if args.smoke else ""), "",
             f"{FULL} with step 11's temperature (T = {T:.3f}) and zero-shot, MLX 8-bit, fresh video decodes, "
             "16 frames in Gemma 4's video layout at 140 tokens per frame. Temperature never changes an answer.", "",
             f"## {name}", "",
             "| | n | accuracy | NLL | Brier | ECE | conf ≥ 0.8: cov / acc | latency (model) | decode |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, x in m.items():
        hi = "-" if x["accuracy_conf_ge_0.8"] is None else f"{x['accuracy_conf_ge_0.8']:.1%}"
        lines.append(f"| {arm} | {x['n']} | {x['accuracy']:.1%} ± {x['ci95']:.1%} | {x['nll']:.3f} | {x['brier']:.3f} | "
                     f"{x['ece']:.3f} | {x['coverage_conf_ge_0.8']:.0%} / {hi} | {x['latency_s']:.2f} s | {x['decode_s']:.2f} s |")
    lines += ["", f"{FULL} vs zero-shot: gains {g}, loses {lost} (net {g - lost:+d}, McNemar p = {p:.2g}).", "",
              "## By area", "", "| area | n | zero-shot | " + FULL + " |", "|---|---|---|---|"]
    for a, v in m[FULL]["by_area"].items():
        lines.append(f"| {a} | {v['n']} | {m['zero-shot']['by_area'][a]['accuracy']:.1%} | {v['accuracy']:.1%} |")
    gap = m[FULL]["accuracy"] - dev[FULL]["accuracy"]
    lines += ["", "## Against dev and the reference", "",
              "| | dev | " + ("this run" if args.smoke else "test") + " |", "|---|---|---|"]
    for arm in m:
        lines.append(f"| {arm} accuracy | {dev[arm]['accuracy']:.1%} | {m[arm]['accuracy']:.1%} |")
    with storage.drive_guard():  # dev's ECE with T as step 11 reported it: cross-fitted, not fit and scored on dev
        dev_ece = json.load(open(calibration_file()))["dev_crossfit"]["ece"]
    lines.append(f"| {FULL} + T ECE | {dev_ece:.3f} (cross-fitted) | {m[f'{FULL} + T']['ece']:.3f} |")
    lines += ["", f"- Test minus dev: {gap:+.1%} "
              + ("(beyond dev's ~2-point resolution: worth noting, not a reason to change anything)."
                 if abs(gap) > DEV_RESOLUTION else "(within dev's ~2-point resolution)."),
              f"- d3-mini: {D3_MINI:.1%} self-reported on its own protocol (about 71% on our dev, step 4); "
              f"{FULL} here: {m[FULL]['accuracy']:.1%}.",
              "- Measured on MLX 8-bit, the same path as every dev number; no torch bf16 cross-check (see the plan)."]
    text = "\n".join(lines) + "\n"
    d = storage.path("results", STEP + ("_smoke" if args.smoke else ""))
    os.makedirs(d, exist_ok=True)
    with storage.drive_guard(), open(os.path.join(d, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {os.path.join(d, 'summary.md')}")
    if not args.smoke:
        log_run(STEP, "eval", {"what": "report", "T": T}, {"manifest": "test"}, None,
                {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in m.items()})


def run(args):
    storage.require_root()
    check(args)
    score(args)
    report(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": run, "check": check, "score": score, "report": report}
    for c in cmds:
        sub.add_parser(c).add_argument("--smoke", action="store_true", help="2 dev videos; never touches test")
    args = ap.parse_args(argv)
    try:
        cmds[args.cmd](args)
    except storage.DriveMissing as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
