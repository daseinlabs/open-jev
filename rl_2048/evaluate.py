"""Play full games with a fixed policy and summarise the results.

Game ``i`` always uses spawn seed ``EVAL_SEED + i``, so every policy starts from
the same boards. Model policies play greedily (the most likely valid move).
"""
from __future__ import annotations

import random
import statistics
import time
from typing import Callable

from .game import ACTIONS, Game

EVAL_SEED = 10_000

# A chooser gets the game and returns (action, top_choice_was_invalid).
Chooser = Callable[[Game], tuple[str, bool]]


def random_chooser(seed: int = 0) -> Chooser:
    rng = random.Random(seed)
    return lambda game: (rng.choice(game.valid_actions()), False)


def greedy_chooser(policy) -> Chooser:
    def choose(game: Game) -> tuple[str, bool]:
        valid = game.valid_actions()
        p = policy.probs(policy.encode(game.board, valid)).tolist()
        top = ACTIONS[max(range(4), key=p.__getitem__)]
        best = max(valid, key=lambda a: p[ACTIONS.index(a)])
        return best, top not in valid
    return choose


def play(choose: Chooser, n_games: int, max_moves: int, label: str, log=print) -> dict:
    games = []
    t0 = time.time()
    for i in range(n_games):
        game, invalid_top = Game(seed=EVAL_SEED + i), 0
        while not game.over() and game.moves < max_moves:
            action, bad = choose(game)
            invalid_top += bad
            game.step(action)
        games.append({"score": game.score, "max_tile": game.max_tile, "moves": game.moves,
                      "invalid_top_rate": invalid_top / max(1, game.moves)})
        log(f"  [{label}] game {i + 1}/{n_games}: score {game.score}, max tile {game.max_tile}, "
            f"{game.moves} moves ({time.time() - t0:.0f}s)")
    return summarise(label, games)


def summarise(label: str, games: list[dict]) -> dict:
    scores = [g["score"] for g in games]
    tiles = [g["max_tile"] for g in games]
    return {
        "policy": label,
        "games": len(games),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "best_score": max(scores),
        "mean_moves": statistics.mean(g["moves"] for g in games),
        "reached": {t: sum(x >= t for x in tiles) / len(tiles) for t in (128, 256, 512, 1024, 2048)},
        "invalid_top_rate": statistics.mean(g["invalid_top_rate"] for g in games),
        "per_game": games,
    }
