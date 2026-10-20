"""Watch Gemma play 2048 in the browser, with the move probabilities beside the board.

    .venv/bin/python -m rl_2048.web                      # newest runs/rl2048-*/adapter
    .venv/bin/python -m rl_2048.web --adapter runs/rl2048-<timestamp>/adapter

Then open http://127.0.0.1:8001. One model is loaded; "Base Gemma" is the same
model with the LoRA switched off, so both players fit in memory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from openjev.scorer import DEFAULT_MODEL

from .evaluate import EVAL_SEED
from .game import ACTIONS, Game, apply_move
from .policy import GemmaPolicy

PAGE = Path(__file__).with_name("web.html")


class NewGame(BaseModel):
    seed: int | None = None
    player: str = "trained"


def latest_adapter() -> str | None:
    runs = sorted(Path("runs").glob("rl2048-*/adapter/adapters.safetensors"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1].parent) if runs else None


def create_app(policy: GemmaPolicy, adapter: str | None) -> FastAPI:
    app = FastAPI(title="openjev 2048 viewer")
    # Handlers are async on purpose: MLX streams belong to the thread that loaded the
    # model, so model calls must stay on the event-loop thread, not FastAPI's threadpool.
    state: dict = {"game": Game(seed=EVAL_SEED), "player": "trained" if adapter else "base", "seed": EVAL_SEED}

    def snapshot() -> dict:
        g = state["game"]
        return {"board": g.board, "score": g.score, "moves": g.moves, "max_tile": g.max_tile,
                "over": g.over(), "player": state["player"], "seed": state["seed"],
                "valid": g.valid_actions()}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/info")
    def info() -> dict:
        return {"adapter": adapter, "actions": ACTIONS, **snapshot()}

    @app.post("/api/new")
    async def new(req: NewGame) -> dict:
        seed = req.seed if req.seed is not None else int(time.time()) % 100_000
        state.update(game=Game(seed=seed), seed=seed,
                     player=req.player if (req.player == "base" or adapter) else "base")
        return snapshot()

    @app.post("/api/step")
    async def step() -> dict:
        g = state["game"]
        if g.over():
            return {"done": True, **snapshot()}
        valid = g.valid_actions()
        t0 = time.perf_counter()
        ids = policy.encode(g.board, valid)
        p = (policy.probs(ids) if state["player"] == "trained" else policy.reference_probs(ids)).tolist()
        ms = (time.perf_counter() - t0) * 1000
        top = ACTIONS[max(range(4), key=p.__getitem__)]
        chosen = max(valid, key=lambda a: p[ACTIONS.index(a)])
        before = g.board[:]
        slid, _, _ = apply_move(before, chosen)
        gain = g.step(chosen)
        spawned = next((i for i, (a, b) in enumerate(zip(slid, g.board)) if a != b), None)
        return {"done": False, "before": before, "probs": dict(zip(ACTIONS, p)), "chosen": chosen,
                "top_choice": top, "gain": gain, "spawned": spawned, "ms": ms, **snapshot(),
                "valid_before": valid}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="trained adapter dir (default: newest runs/rl2048-*/adapter)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    adapter = args.adapter or latest_adapter()
    print(f"loading {args.model}" + (f" with adapter {adapter}" if adapter else " (no adapter found: base only)"))
    # temperature 1: show the model's real probabilities; play is greedy either way.
    policy = GemmaPolicy(args.model, adapter_path=adapter, temperature=1.0)
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(create_app(policy, adapter), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
