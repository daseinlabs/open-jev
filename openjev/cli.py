"""Command line entry points: score, eval, bench, check, serve, features, train, eval-head."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time

from .scorer import DEFAULT_MODEL, NORMS, OptionScorer, iter_jsonl


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps (torch backend)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="local dir or HF repo id")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir to load on top of --model (e.g. adapters/chess-lora)")
    p.add_argument("--batch-size", type=int, default=8, help="options per forward pass")
    p.add_argument("--norm", choices=NORMS, default="mean",
                   help="mean: per-token log-prob; sum: total; pmi: sum minus unconditional")
    p.add_argument("--chat", action="store_true",
                   help="wrap context in Gemma's chat template; options score as the reply")
    p.add_argument("--sep", default="", help="string inserted between context and option")


def _scorer(args: argparse.Namespace) -> OptionScorer:
    t = time.perf_counter()
    s = OptionScorer(args.model, batch_size=args.batch_size, chat=args.chat, sep=args.sep, adapter_path=args.adapter, backend=args.backend, device=args.device)
    print(f"loaded {args.model} in {time.perf_counter() - t:.1f}s", file=sys.stderr)
    return s


def _read_options(path: str) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def cmd_score(args: argparse.Namespace) -> None:
    if args.options_file:
        args.option = _read_options(args.options_file) + args.option
    if len(args.option) < 2:
        sys.exit("need at least two options (--option ... or --options-file)")
    scorer = _scorer(args)
    result = scorer.score(args.context, args.option, norm=args.norm)
    scorer.score(args.context, args.option, norm=args.norm)  # warm second run for timing
    payload = {
        "context": args.context,
        "norm": args.norm,
        "timing": scorer.last_timing,
        "options": [r.to_dict() for r in result],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    ranked = sorted(result, key=lambda r: -r.score)
    for r in ranked:
        print(f"{r.probability:7.3%}  score={r.score:8.3f}  sum={r.logprob_sum:8.3f}  "
              f"n={r.n_tokens:3d}  {r.option!r}")
    t = scorer.last_timing
    print(f"[{t['context_tokens']} ctx tok, {t['option_tokens']} opt tok] "
          f"prefill {t['prefill_s']*1000:.0f} ms, options {t['options_s']*1000:.0f} ms, "
          f"total {t['total_s']*1000:.0f} ms", file=sys.stderr)


def cmd_eval(args: argparse.Namespace) -> None:
    scorer = _scorer(args)
    rows = list(iter_jsonl(args.data))
    if args.limit:
        rows = rows[: args.limit]
    fixed = _read_options(args.fixed_options) if args.fixed_options else None
    top1 = top3 = labelled = 0
    lat: list[float] = []
    for i, row in enumerate(rows):
        options = row.get("options") or fixed
        if not options:
            sys.exit(f"row {i} has no 'options' and no --fixed-options given")
        res = scorer.score(row["context"], options, norm=args.norm)
        lat.append(scorer.last_timing["total_s"])
        order = sorted(range(len(res)), key=lambda j: -res[j].score)
        label = row.get("label")
        if label is not None:
            labelled += 1
            top1 += order[0] == label
            top3 += label in order[:3]
        if args.verbose or label is None:
            print(json.dumps({"i": i, "label": label, "pred": order[0], "pred_option": options[order[0]],
                              "probs": [round(r.probability, 4) for r in res]}))
    n = len(rows)
    print(json.dumps({
        "examples": n,
        "labelled": labelled,
        "top1": top1 / labelled if labelled else None,
        "top3": top3 / labelled if labelled else None,
        "norm": args.norm,
        "median_latency_s": statistics.median(lat),
        "mean_latency_s": statistics.fmean(lat),
    }, indent=2))


def _synthetic(scorer: OptionScorer, ctx_tokens: int, n_opts: int, opt_tokens: int, seed: int):
    rng = random.Random(seed)
    words = ("river stone cloud engine quiet market ledger signal orbit velvet "
             "harbour lantern cipher meadow granite").split()
    def text(n):  # roughly n tokens of plain words
        return " ".join(rng.choice(words) for _ in range(n))
    context = text(ctx_tokens)
    options = [text(opt_tokens) for _ in range(n_opts)]
    return context, options


def cmd_bench(args: argparse.Namespace) -> None:
    scorer = _scorer(args)
    context, options = _synthetic(scorer, args.context_tokens, args.options, args.option_tokens, 0)
    ctx_n = len(scorer.context_ids(context))
    opt_n = sum(len(scorer.option_ids(o)) for o in options)
    scorer.score(context, options, norm="mean")  # warm-up
    scorer.score_naive(context, options[:1])

    cached = []
    for _ in range(args.repeat):
        scorer.score(context, options, norm="mean")
        cached.append(scorer.last_timing["total_s"])
    naive = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        scorer.score_naive(context, options)
        naive.append(time.perf_counter() - t)
    print(json.dumps({
        "context_tokens": ctx_n,
        "options": len(options),
        "option_tokens_total": opt_n,
        "batch_size": scorer.batch_size,
        "prefix_cached_batched_s": {"median": statistics.median(cached), "min": min(cached)},
        "naive_per_option_s": {"median": statistics.median(naive), "min": min(naive)},
        "speedup": statistics.median(naive) / statistics.median(cached),
    }, indent=2))


def cmd_check(args: argparse.Namespace) -> None:
    """Verify the prefix-shared batched path against naive full re-encoding."""
    scorer = _scorer(args)
    cases = [
        ("The capital of France is", [" Paris", " Berlin", " a city in Europe", " Lyon"]),
        ("Q: What is 2 + 2?\nA:", [" 4", " 5", " four", " twenty-two"]),
    ]
    context, options = _synthetic(scorer, args.context_tokens, args.options, args.option_tokens, 1)
    cases.append((context, options))
    worst = 0.0
    for ctx, opts in cases:
        fast = [r.logprob_sum for r in scorer.score(ctx, opts, norm="sum")]
        slow = scorer.score_naive(ctx, opts)
        diffs = [abs(a - b) for a, b in zip(fast, slow)]
        worst = max(worst, max(diffs))
        rel = max(d / max(1e-6, abs(b)) for d, b in zip(diffs, slow))
        print(f"ctx_tokens={len(scorer.context_ids(ctx)):4d} options={len(opts):2d} "
              f"max_abs_diff={max(diffs):.4f} max_rel_diff={rel:.4%}")
        for o, a, b in zip(opts, fast, slow):
            print(f"    cached={a:9.3f} naive={b:9.3f}  {o[:40]!r}")
    ok = worst < args.tol
    print("OK" if ok else f"MISMATCH (worst abs diff {worst:.4f} > tol {args.tol})")
    sys.exit(0 if ok else 1)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(args.host, args.port, args.model, args.batch_size, backend=args.backend, device=args.device)


def cmd_features(args: argparse.Namespace) -> None:
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64" or getattr(args, "backend", "auto") == "torch":
        raise SystemExit("Feature extraction and head training/evaluation require MLX on Apple silicon. Scoring and serving support Windows/Linux via torch.")
    from .features import extract_dataset

    scorer = _scorer(args)
    meta = extract_dataset(scorer, args.data, args.out, limit=args.limit, chat=args.chat, sep=args.sep,
                           contextual=args.contextual)
    print(json.dumps(meta))


def cmd_train(args: argparse.Namespace) -> None:
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64" or getattr(args, "backend", "auto") == "torch":
        raise SystemExit("Feature extraction and head training/evaluation require MLX on Apple silicon. Scoring and serving support Windows/Linux via torch.")
    from .train import train

    print(json.dumps(train(args.train, args.validation, args.out, rank=args.rank, epochs=args.epochs,
                           batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed)))


def cmd_eval_head(args: argparse.Namespace) -> None:
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64" or getattr(args, "backend", "auto") == "torch":
        raise SystemExit("Feature extraction and head training/evaluation require MLX on Apple silicon. Scoring and serving support Windows/Linux via torch.")
    from .features import FeatureSet
    from .head import AttentionHead
    from .train import evaluate

    head, cfg = AttentionHead.load(args.checkpoint)
    fs = FeatureSet(args.features)
    print(json.dumps({"model": evaluate(head, fs), "shuffled_context": evaluate(head, fs, shuffle_context=True),
                      "checkpoint": args.checkpoint, "rank": cfg["rank"]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="openjev", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score options for one context")
    _add_common(s)
    s.add_argument("--context", required=True)
    s.add_argument("--option", action="append", default=[], help="repeat for each option")
    s.add_argument("--options-file", help="text file with one predefined option per line")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_score)

    e = sub.add_parser("eval", help="top-k accuracy on jevlike-style JSONL {context, options, label}")
    _add_common(e)
    e.add_argument("data")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--fixed-options", help="text file of options used for every row lacking an 'options' field")
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(fn=cmd_eval)

    for name, fn, helptext in (
        ("bench", cmd_bench, "latency of prefix-cached batched scoring vs naive re-encoding"),
        ("check", cmd_check, "verify cached batched scores match naive full-sequence scores"),
    ):
        b = sub.add_parser(name, help=helptext)
        _add_common(b)
        b.add_argument("--context-tokens", type=int, default=200)
        b.add_argument("--options", type=int, default=8)
        b.add_argument("--option-tokens", type=int, default=30)
        b.add_argument("--repeat", type=int, default=5)
        b.add_argument("--tol", type=float, default=0.5, help="check: max abs log-prob diff allowed")
        b.set_defaults(fn=fn)

    f = sub.add_parser("features", help="cache frozen-Gemma features for a JSONL file into an .npz")
    _add_common(f)
    f.add_argument("data")
    f.add_argument("--out", required=True)
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--contextual", action="store_true",
                   help="encode options as continuations of the context (leaks the match into option features)")
    f.set_defaults(fn=cmd_features)

    tr = sub.add_parser("train", help="train the attention head on cached features")
    tr.add_argument("train")
    tr.add_argument("--validation", required=True)
    tr.add_argument("--out", default="runs/head.safetensors")
    tr.add_argument("--rank", type=int, default=256)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--learning-rate", type=float, default=5e-4)
    tr.add_argument("--seed", type=int, default=7)
    tr.set_defaults(fn=cmd_train)

    eh = sub.add_parser("eval-head", help="top-k, ECE and shuffled-context control for a trained head")
    eh.add_argument("checkpoint")
    eh.add_argument("features")
    eh.set_defaults(fn=cmd_eval_head)

    v = sub.add_parser("serve", help="HTTP server with the model loaded once (POST /score, /v1/systemone)")
    v.add_argument("--model", default=DEFAULT_MODEL)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    v.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(fn=cmd_serve)

    from .decision_cli import register
    register(sub)
    from .tod.cli import register as register_tod
    register_tod(sub)

    args = p.parse_args(argv)
    if len(getattr(args, "option", []) or []) == 1:
        p.error("--option must be given at least twice")
    args.fn(args)


if __name__ == "__main__":
    main()
