"""Watch a model play 2048 in the terminal, through openjev's public OptionScorer API.

    .venv/bin/python -m rl_2048.play --adapter runs/rl2048-<timestamp>/adapter
    .venv/bin/python -m rl_2048.play            # base model, for comparison

The adapter written by training is a normal mlx_lm LoRA adapter, so the same
directory also works with ``openjev serve --chat --adapter ...``.
"""
from __future__ import annotations

import argparse
import time

from openjev.scorer import DEFAULT_MODEL, OptionScorer

from .evaluate import EVAL_SEED
from .game import ACTIONS, Game, render
from .policy import prompt_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--delay", type=float, default=0.15, help="seconds between moves")
    ap.add_argument("--max-moves", type=int, default=2000)
    args = ap.parse_args()

    scorer = OptionScorer(args.model, chat=True, adapter_path=args.adapter)
    game = Game(seed=args.seed)
    while not game.over() and game.moves < args.max_moves:
        valid = game.valid_actions()
        ranked = scorer.score(prompt_for(game.board, valid), list(ACTIONS), norm="sum")
        probs = {r.option: r.probability for r in ranked}
        action = max(valid, key=probs.get)
        bars = "  ".join(f"{a:>5} {probs[a]:6.1%}{'*' if a == action else ' '}" for a in ACTIONS)
        print("\033[2J\033[H" + render(game.board))
        print(f"\nscore {game.score}   move {game.moves}   max tile {game.max_tile}")
        print(f"{bars}   ({scorer.last_timing['total_s'] * 1000:.0f} ms)")
        game.step(action)
        time.sleep(args.delay)
    print("\033[2J\033[H" + render(game.board))
    print(f"\n{'game over' if game.over() else 'stopped'}: score {game.score}, max tile {game.max_tile}, {game.moves} moves")


if __name__ == "__main__":
    main()
