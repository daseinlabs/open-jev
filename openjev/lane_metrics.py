"""Cheap monotonic /metrics for /v1/systemone lane servers.

Stdlib-only on purpose: the prod gemma lanes run the `openjev` package
(~/CascadeProjects/open-jev — a separate repo outside chaba). Vendor this
file there to give those lanes the same endpoint; nest-overview.py falls
back to "no metrics" for lanes that don't expose it yet.

GET /metrics -> JSON, all counters monotonic since process start:

    uptime_s           wall seconds since process start
    cpu_seconds        process CPU time (user+system), monotonic
    requests_total     /v1/systemone calls served
    questions_total    individual questions answered
    escalations_total  questions answered with a fallback/stub — the
                       caller must route these to a heavier tier
    latency_seconds    {count, sum, buckets{le_*: cumulative}}

Cheap by design: a few int/float adds per request, no locks beyond
FastAPI's per-request thread, no persistence.
"""
import time
from collections import OrderedDict

# Latency histogram edges (seconds) — covers a 10ms distilbert gate up to
# a 60s gemma-4b CPU decision.
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0)


class LaneMetrics:
    def __init__(self):
        self._t0 = time.monotonic()
        self.requests = 0
        self.questions = 0
        self.escalations = 0
        self.latency_sum = 0.0
        self.latency_buckets = OrderedDict((b, 0) for b in BUCKETS)
        self.latency_count = 0

    def observe(self, elapsed_s: float, questions: int = 0,
                escalations: int = 0):
        """Record one served request. `escalations` = questions answered
        with a stub/fallback the caller must route to a heavier tier."""
        self.requests += 1
        self.questions += questions
        self.escalations += escalations
        self.latency_sum += elapsed_s
        self.latency_count += 1
        for edge in self.latency_buckets:
            if elapsed_s <= edge:
                self.latency_buckets[edge] += 1

    def snapshot(self) -> dict:
        return {
            "uptime_s": round(time.monotonic() - self._t0, 3),
            "cpu_seconds": round(time.process_time(), 3),
            "requests_total": self.requests,
            "questions_total": self.questions,
            "escalations_total": self.escalations,
            "latency_seconds": {
                "count": self.latency_count,
                "sum": round(self.latency_sum, 3),
                "buckets": {"le_inf": self.latency_count,
                            **{f"le_{b:g}": n
                               for b, n in self.latency_buckets.items()}},
            },
        }


METRICS = LaneMetrics()


def attach(app):
    """Register GET /metrics on a FastAPI app."""
    @app.get("/metrics")
    def metrics():
        return METRICS.snapshot()
