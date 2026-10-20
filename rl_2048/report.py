"""Turn a run directory into charts.svg and report.md (no plotting library needed).

    .venv/bin/python -m rl_2048.report runs/rl2048-<timestamp>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

W, H, PAD = 560, 220, 44


def _rolling(xs: list[float], k: int) -> list[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= k:
            acc -= xs[i - k]
        out.append(acc / min(i + 1, k))
    return out


def _line_panel(title: str, ys: list[float], y_label: str, raw: list[float] | None = None) -> str:
    if not ys:
        return f'<text x="{PAD}" y="30" class="t">{title}: no data yet</text>'
    lo, hi = min(raw or ys), max(raw or ys)
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    n = max(1, len(ys) - 1)
    px = lambda i: PAD + (W - 2 * PAD) * i / n
    py = lambda v: H - PAD + 10 - (H - 2 * PAD) * (v - lo) / (hi - lo)
    parts = [f'<text x="{PAD}" y="22" class="t">{title}</text>',
             f'<line x1="{PAD}" y1="{H - PAD + 10}" x2="{W - PAD}" y2="{H - PAD + 10}" class="ax"/>',
             f'<text x="{PAD - 6}" y="{py(hi) + 4:.1f}" class="l" text-anchor="end">{hi:.3g}</text>',
             f'<text x="{PAD - 6}" y="{py(lo) + 4:.1f}" class="l" text-anchor="end">{lo:.3g}</text>',
             f'<text x="{W - PAD}" y="{H - 8}" class="l" text-anchor="end">{y_label}</text>']
    if raw:
        parts += [f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="2" class="dot"/>' for i, v in enumerate(raw)]
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    parts.append(f'<polyline points="{pts}" class="ln"/>')
    return "\n".join(parts)


def _bar_panel(evals: list[dict]) -> str:
    if not evals:
        return '<text x="44" y="30" class="t">Evaluation: not run yet</text>'
    top = max(e["mean_score"] for e in evals) or 1
    bw = (W - 2 * PAD) / len(evals)
    parts = ['<text x="44" y="22" class="t">Evaluation: mean score per game (same starting seeds)</text>']
    for i, e in enumerate(evals):
        h = (H - 2 * PAD - 10) * e["mean_score"] / top
        x = PAD + i * bw + bw * 0.2
        parts += [f'<rect x="{x:.1f}" y="{H - PAD + 10 - h:.1f}" width="{bw * 0.6:.1f}" height="{h:.1f}" class="bar"/>',
                  f'<text x="{x + bw * 0.3:.1f}" y="{H - PAD + 10 - h - 6:.1f}" class="l" text-anchor="middle">{e["mean_score"]:.0f}</text>',
                  f'<text x="{x + bw * 0.3:.1f}" y="{H - 14}" class="l" text-anchor="middle">{e["policy"]}</text>']
    return "\n".join(parts)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def build(run: Path) -> None:
    steps = _read_jsonl(run / "train_log.jsonl")
    episodes = _read_jsonl(run / "episodes.jsonl")
    evals = [json.loads((run / f).read_text()) for f in
             ("eval_random.json", "eval_base.json", "eval_trained.json") if (run / f).exists()]
    k = max(1, len(steps) // 20)
    panels = [
        _line_panel(f"Probability on the best move (rolling mean of {k} updates)",
                    _rolling([s["best_move_prob"] for s in steps], k), "update"),
        _line_panel(f"Policy advantage, z-score of chosen mix vs average move (rolling {k})",
                    _rolling([s["advantage_z"] for s in steps], k), "update"),
        _line_panel("Training games: final score (dots) and rolling mean of 5",
                    _rolling([e["score"] for e in episodes], 5), "game", [e["score"] for e in episodes]),
        _bar_panel(evals),
    ]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H * len(panels)}" '
           f'viewBox="0 0 {W} {H * len(panels)}" font-family="sans-serif">',
           '<style>.t{font-size:13px;font-weight:600;fill:#222}.l{font-size:11px;fill:#555}'
           '.ax{stroke:#999}.ln{fill:none;stroke:#2563eb;stroke-width:2}.dot{fill:#93c5fd}'
           '.bar{fill:#2563eb}</style><rect width="100%" height="100%" fill="#fff"/>']
    for i, p in enumerate(panels):
        svg.append(f'<g transform="translate(0,{i * H})">{p}</g>')
    svg.append("</svg>")
    (run / "charts.svg").write_text("\n".join(svg))

    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    lines = [f"# 2048 RL run: {run.name}", "", "![charts](charts.svg)", "",
             f"Updates: {len(steps)}, training states: {sum(s['states'] for s in steps)}, "
             f"training games finished: {len(episodes)}", ""]
    if evals:
        lines += ["| Policy | Games | Mean score | Median | Best | Mean moves | Reached 256 | 512 | 1024 | Top choice invalid |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for e in evals:
            r = {int(t): v for t, v in e["reached"].items()}
            lines.append(f"| {e['policy']} | {e['games']} | {e['mean_score']:.0f} | {e['median_score']:.0f} | "
                         f"{e['best_score']} | {e['mean_moves']:.0f} | {r[256]:.0%} | {r[512]:.0%} | "
                         f"{r[1024]:.0%} | {e['invalid_top_rate']:.1%} |")
        lines.append("")
    if steps:
        first, last = steps[: max(1, len(steps) // 10)], steps[-max(1, len(steps) // 10):]
        avg = lambda xs, key: sum(x[key] for x in xs) / len(xs)
        lines += ["| Training metric (first 10% vs last 10% of updates) | Start | End |", "|---|---|---|"]
        for key, name in [("best_move_prob", "Probability on best move"),
                          ("advantage_z", "Policy advantage (z)"),
                          ("invalid_prob", "Probability on invalid moves"),
                          ("entropy", "Entropy (nats, max 1.386)"),
                          ("kl", "KL from base model")]:
            lines.append(f"| {name} | {avg(first, key):.3f} | {avg(last, key):.3f} |")
        lines.append("")
    if cfg:
        lines += ["## Config", "", "```json", json.dumps(cfg, indent=2), "```"]
    (run / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print(f"wrote {sys.argv[1]}/report.md and charts.svg")
