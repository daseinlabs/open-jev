"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \\
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import lane_metrics
from .scorer import DEFAULT_MODEL, OptionScorer
from .systemone import SystemOneRequest, SystemOneResponse, system_one

# Bound in-flight inference. Every route here is a sync `def`, so Starlette
# dispatches it to anyio's 40-token threadpool. An unbounded burst of scoring
# requests fills all 40 workers with multi-minute Gemma3 forwards, /health
# can no longer get a slot, and the process must be SIGKILLed. Reject excess
# work immediately instead of queueing it into another pileup.
_MAX_INFLIGHT = max(1, int(os.environ.get("OPENJEV_MAX_INFLIGHT", "2")))
_INFER_GATE = threading.Semaphore(_MAX_INFLIGHT)


def _acquire_infer_slot() -> None:
    if not _INFER_GATE.acquire(blocking=False):
        raise HTTPException(
            429,
            f"too many concurrent inference requests (max {_MAX_INFLIGHT}); retry later",
            headers={"Retry-After": "60"},
        )


class ScoreRequest(BaseModel):
    context: str
    options: list[str] = Field(min_length=2)
    norm: Literal["mean", "sum", "pmi"] = "mean"
    chat: bool = False
    sep: str = ""


class OptionOut(BaseModel):
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float


class ScoreResponse(BaseModel):
    best: str
    best_index: int
    options: list[OptionOut]
    timing: dict[str, float]


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str = "auto", device: str = "auto") -> FastAPI:
    model_path = model_path or os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL)
    adapter_path = os.environ.get("OPENJEV_ADAPTER")  # optional LoRA adapter dir
    app = FastAPI(title="openjev", version="0.1.0")
    lane_metrics.attach(app)
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer <key>"
    model_name = os.path.basename(model_path.rstrip("/"))

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    @app.on_event("startup")
    def _load() -> None:
        t = time.perf_counter()
        scorer = OptionScorer(model_path, batch_size=batch_size, adapter_path=adapter_path, backend=backend, device=device)
        scorer.score("warm up", ["a", "b"])  # compile kernels before the first request
        state["scorer"] = scorer
        state["load_s"] = time.perf_counter() - t

    @app.get("/health")
    def health() -> dict:
        return {"ok": "scorer" in state, "model": model_path, "load_s": state.get("load_s"), "backend": getattr(state.get("scorer"), "backend", None), "device": getattr(state.get("scorer"), "device", None)}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        _acquire_infer_slot()
        try:
            res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            _INFER_GATE.release()
        best = max(range(len(res)), key=lambda i: res[i].score)
        return ScoreResponse(
            best=res[best].option,
            best_index=best,
            options=[OptionOut(**r.to_dict()) for r in res],
            timing=scorer.last_timing,
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    def systemone(req: SystemOneRequest) -> SystemOneResponse:
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai)."""
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        _acquire_infer_slot()
        t0 = time.perf_counter()
        try:
            resp = system_one(scorer, req, model_name=req.model or model_name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            _INFER_GATE.release()
        # every question type gets a real scored answer here — no
        # stub/heavier-tier fallback path, so escalations stay 0.
        lane_metrics.METRICS.observe(time.perf_counter() - t0,
                                     questions=len(resp.answers),
                                     escalations=0)
        return resp

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str = "auto", device: str = "auto") -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size, backend=backend, device=device), host=host, port=port, workers=1)
