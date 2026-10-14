"""2048 on a 4x4 board, in plain Python so rollouts are cheap.

The board is a list of 16 ints in row-major order (0 = empty). A move slides
every tile toward one side; two equal tiles that collide merge once into their
sum, and the merged values are the points earned. After every move that changes
the board, a 2 (90%) or 4 (10%) appears in a random empty cell.
"""
from __future__ import annotations

import math
import random

ACTIONS = ("up", "down", "left", "right")
SIZE = 4


def _lines(direction: str) -> list[list[int]]:
    """Board indices for each line, ordered so tiles slide toward index 0."""
    rows = [[r * SIZE + c for c in range(SIZE)] for r in range(SIZE)]
    cols = [[r * SIZE + c for r in range(SIZE)] for c in range(SIZE)]
    return {
        "left": rows,
        "right": [row[::-1] for row in rows],
        "up": cols,
        "down": [col[::-1] for col in cols],
    }[direction]


LINES = {a: _lines(a) for a in ACTIONS}


def slide(line: list[int]) -> tuple[list[int], int]:
    """Slide one line toward index 0. Returns the new line and points earned."""
    tiles = [v for v in line if v]
    out, gain, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gain += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (len(line) - len(out)), gain


def apply_move(board: list[int], action: str) -> tuple[list[int], int, bool]:
    """Apply a move without spawning. Returns (new board, points, changed)."""
    new = board[:]
    gain = 0
    for idx in LINES[action]:
        line, g = slide([board[i] for i in idx])
        gain += g
        for i, v in zip(idx, line):
            new[i] = v
    return new, gain, new != board


def spawn(board: list[int], rng: random.Random) -> None:
    empty = [i for i, v in enumerate(board) if not v]
    if empty:
        board[rng.choice(empty)] = 4 if rng.random() < 0.1 else 2


def valid_actions(board: list[int]) -> list[str]:
    return [a for a in ACTIONS if apply_move(board, a)[2]]


def render(board: list[int]) -> str:
    """Fixed-width text grid, '.' for empty cells."""
    return "\n".join(
        " ".join(f"{(board[r * SIZE + c] or '.'):>4}" for c in range(SIZE)) for r in range(SIZE)
    )


class Game:
    """One game of 2048. ``rng`` drives tile spawns only."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.board = [0] * (SIZE * SIZE)
        self.score = 0
        self.moves = 0
        spawn(self.board, self.rng)
        spawn(self.board, self.rng)

    @property
    def max_tile(self) -> int:
        return max(self.board)

    def valid_actions(self) -> list[str]:
        return valid_actions(self.board)

    def over(self) -> bool:
        return not self.valid_actions()

    def step(self, action: str) -> int:
        """Play a move. Invalid moves change nothing and earn 0."""
        new, gain, changed = apply_move(self.board, action)
        if changed:
            self.board = new
            self.score += gain
            self.moves += 1
            spawn(self.board, self.rng)
        return gain


def rollout_values(
    board: list[int], rng: random.Random, n_rollouts: int, depth: int, death_penalty: float
) -> dict[str, float | None]:
    """Monte Carlo estimate of how good each move is right now.

    For each valid move: play it, then play ``depth`` random valid moves, and add
    up the points. Average over ``n_rollouts`` tries; subtract ``death_penalty``
    when the game ends inside the horizon. Invalid moves get None.
    """
    values: dict[str, float | None] = {}
    for a in ACTIONS:
        after, gain, changed = apply_move(board, a)
        if not changed:
            values[a] = None
            continue
        total = 0.0
        for _ in range(n_rollouts):
            b = after[:]
            spawn(b, rng)
            points = 0
            for _ in range(depth):
                moves = valid_actions(b)
                if not moves:
                    points -= death_penalty
                    break
                b, g, _ = apply_move(b, rng.choice(moves))
                points += g
                spawn(b, rng)
            total += points
        values[a] = gain + total / n_rollouts
    return values


def normalised_values(values: dict[str, float | None]) -> list[float]:
    """Rollout values -> z-scores over the 4 moves. Invalid moves rank below every valid one."""
    valid = [v for v in values.values() if v is not None]
    spread = max(1.0, (max(valid) - min(valid)))
    raw = [values[a] if values[a] is not None else min(valid) - spread for a in ACTIONS]
    mean = sum(raw) / 4
    std = math.sqrt(sum((r - mean) ** 2 for r in raw) / 4)
    return [0.0] * 4 if std < 1e-6 else [(r - mean) / std for r in raw]
