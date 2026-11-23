"""The per-request adapter choice survives the server's real request path.

Requests go through the real FastAPI app (ASGI, Starlette's worker threadpool for sync
endpoints), many at once. The stub scorer records the adapter active in the thread that does
the scoring. Covered:
- text requests on the existing /score endpoint run on the base weights, even while video
  requests are in flight;
- a video request's choice reaches the scorer through the threadpool handoff, and through a
  further handoff to a separate worker thread (e.g. one MLX thread) via ``run_scoped``;
- control: a plain ``executor.submit`` drops the choice, which is why ``run_scoped`` exists.
"""
import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    import httpx
    from fastapi import HTTPException  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

from openjev.adapter_scope import VIDEO, active_adapter, request_scope, run_scoped


class _Result:
    def __init__(self, option, score):
        self.option, self.score = option, score

    def to_dict(self):
        return {"option": self.option, "n_tokens": 1, "logprob_sum": self.score, "logprob_mean": self.score,
                "logprob_uncond": None, "score": self.score, "probability": 0.5}


class StubScorer:
    """Records which adapter was active, and in which thread, when scoring ran."""

    backend, device = "stub", "cpu"

    def __init__(self):
        self.last_timing = {}
        self.seen = []
        self.lock = threading.Lock()

    def score(self, context, options, **kwargs):
        time.sleep(0.02)  # let concurrent requests overlap
        with self.lock:
            self.seen.append((context, active_adapter(), threading.get_ident()))
        return [_Result(o, -float(i)) for i, o in enumerate(options)]

    def probe(self, tag):
        time.sleep(0.02)
        return {"tag": tag, "adapter": active_adapter(), "thread": threading.get_ident()}


@unittest.skipUnless(HAVE_DEPS, "needs httpx and fastapi")
class ServerAdapterScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openjev.server import create_app
        cls.scorer = StubScorer()
        cls.app = create_app("models/stub", scorer=cls.scorer)
        cls.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-worker")  # one model thread
        worker, scorer = cls.worker, cls.scorer

        # Same request path as a future video endpoint: sync handler (Starlette threadpool),
        # adapter chosen from the request, optional handoff to the model worker thread.
        @cls.app.post("/_test/probe")
        def probe(has_video: bool, handoff: str = "none", tag: str = ""):
            if handoff == "run_scoped":
                return run_scoped(has_video, scorer.probe, tag, executor=worker)
            if handoff == "plain_submit":  # the bug run_scoped prevents
                with request_scope(has_video):
                    return worker.submit(scorer.probe, tag).result()
            return run_scoped(has_video, scorer.probe, tag)

    @classmethod
    def tearDownClass(cls):
        cls.worker.shutdown()

    def fire(self, requests):
        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(*(client.post(url, json=body, params=params)
                                              for url, body, params in requests))
        return asyncio.run(go())

    def test_mixed_concurrent_requests_get_the_right_weights(self):
        reqs = []
        for i in range(6):
            reqs.append(("/score", {"context": f"text-{i}", "options": ["a", "b"]}, None))
            reqs.append(("/_test/probe", None, {"has_video": True, "tag": f"v-{i}"}))
            reqs.append(("/_test/probe", None, {"has_video": True, "handoff": "run_scoped", "tag": f"vw-{i}"}))
            reqs.append(("/_test/probe", None, {"has_video": False, "handoff": "run_scoped", "tag": f"t-{i}"}))
        out = self.fire(reqs)
        self.assertTrue(all(r.status_code == 200 for r in out), [r.text for r in out if r.status_code != 200])
        main = threading.get_ident()
        for r, (url, body, params) in zip(out, reqs):
            if url == "/score":
                continue
            got = r.json()
            want = VIDEO if params["has_video"] else None
            self.assertEqual(got["adapter"], want, got)
            self.assertNotEqual(got["thread"], main)
        text = [s for s in self.scorer.seen if s[0].startswith("text-")]
        self.assertEqual(len(text), 6)
        self.assertTrue(all(adapter is None for _, adapter, _ in text), text)
        self.assertIsNone(active_adapter())  # nothing leaked into the caller

    def test_worker_handoff_runs_on_the_worker_thread(self):
        a, b = self.fire([("/_test/probe", None, {"has_video": True, "handoff": "run_scoped", "tag": "x"}),
                          ("/_test/probe", None, {"has_video": True, "handoff": "run_scoped", "tag": "y"})])
        self.assertEqual(a.json()["thread"], b.json()["thread"])  # the single model-worker thread
        self.assertEqual(a.json()["adapter"], VIDEO)

    def test_plain_submit_loses_the_choice(self):
        (r,) = self.fire([("/_test/probe", None, {"has_video": True, "handoff": "plain_submit", "tag": "z"})])
        self.assertIsNone(r.json()["adapter"])


if __name__ == "__main__":
    unittest.main()
