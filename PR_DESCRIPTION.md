# Zero-shot video input on the torch backend + Perception Test eval (step 1)

OpenJev can now score options about a video. Gemma 3 4B has no native video, so a video becomes
N uniformly sampled frames, passed in as images ahead of the prompt text. It is still a decision
model: no text generation, only probabilities over the given options.

This step answers the accuracy question on the **PyTorch/HF backend**, which handles multi-image
Gemma 3 with the correct attention mask natively. There is no MLX vision path, no prefix cache and
no `/v1/systemone` change yet. Those are step 2, pending review of the numbers below.

## What's in it

- **`openjev/video.py`** (backend-independent, PyAV):
  - Loads a video from a file path, http(s) URL, `data:` URI / raw base64, or bytes. Audio is ignored.
  - Samples N frames (default 16) at the midpoints of N equal time bins, `(i + 0.5) · duration / N`.
    Each target time seeks to the nearest keyframe and decodes forward, so long videos are never
    decoded end to end. The frame chosen is the one on screen at the target time.
  - `VideoLimits` caps bytes (512 MB), duration (30 min) and frames (64). Errors raise `VideoError`
    (a `ValueError`) with messages that are safe to return to clients.
  - Returns frames, timestamps, actual frame times, duration, and the sha256 of the bytes. The hash
    is the future cache key.
- **Torch backend image input**: `OptionScorer.score(..., images=frames)`.
  - The prompt is laid out as frames, then the existing System One text (state, question, options,
    `Answer:`), then the options scored one row per option as before.
  - Frames go through `Gemma3Processor`: 896×896, 256 soft tokens plus markers, about 261 tokens
    per frame.
  - SigLIP features are computed a few frames at a time and scattered into the embeddings, exactly
    as `Gemma3Model.forward` does.
  - The prefill is chunked and never cuts through an image block, which keeps memory flat at
    N=64 (16.6k tokens).
  - `image_features=` lets several questions about one video reuse the vision-tower output.
- **Dtypes**:
  - New `dtype` and `vision_dtype` options. Run the language model in bf16 and the vision tower
    in fp16 (details under Limitations).
  - Plumbed through `OptionScorer(dtype=..., vision_dtype=...)` and `TorchBackend`.
- **`video_eval/perception_test.py`**: `fetch | run | report` for the Perception Test MC-VQA
  validation split.
  - `fetch` samples whole videos with a fixed seed.
  - It pulls only those videos out of the 75 GB validation zip, using HTTP range requests
    (`video_eval/remote_zip.py`).
- **Packaging and docs**:
  - New `video` extra (`av`, `pillow`) and an updated `uv.lock`.
  - `docs/video.md`, plus links from the API reference, the mkdocs nav and the README layout section.

### Dataset license

The Perception Test annotations and videos are licensed under CC BY 4.0; the code is under Apache 2.0
([google-deepmind/perception_test](https://github.com/google-deepmind/perception_test)). The
videos are downloaded to `data/perception_test/` (git-ignored) and nothing is redistributed.

## Results

**Setup:**
- Model: Gemma 3 4B-it on an M3 Pro (18 GB), torch 2.14 MPS, bf16 LM / fp16 vision.
- Subset: seed 0, **300 questions over 92 validation videos**, 3 options each. Mean clip
  length is 22.5 s; chance is 33.3%.
- Prompt: the System One `choice` rendering, with options as `A/B/C` described by the option text,
  `norm="sum"`, and the plain (non-chat) layout the API uses. "(chat)" rows wrap the same text
  in Gemma's chat template.
- **N=64 ran on the first 100 questions only (38 videos)**, at about 144 s per question uncached.
  At that rate 300 questions would have taken about 7 hours.

| Frames | Questions | Accuracy (95% CI) | Tokens / request | Uncached latency, mean (p50) | of which vision | Top prob, correct | Top prob, wrong |
|---|---|---|---|---|---|---|---|
| 4 | 300 | 48.7% ± 5.7 | 1101 | 8.8s (8.7s) | 6.0s | 0.926 | 0.890 |
| 4 (chat) | 300 | 49.0% ± 5.7 | 1108 | 8.4s (8.4s) | 5.8s | 0.956 | 0.950 |
| 16 | 300 | 50.3% ± 5.7 | 4209 | 34.0s (33.9s) | 23.4s | 0.915 | 0.892 |
| 16 (chat) | 100 | 51.0% ± 9.8 | 4218 | 33.0s (33.1s) | 23.0s | 0.966 | 0.936 |
| 64 | 100 | 53.0% ± 9.8 | 16643 | 144.0s (137.3s) | 92.7s | 0.943 | 0.916 |

Same 100 questions in every run, which is the fair comparison across N:

| Frames | Accuracy |
|---|---|
| 4 | 54.0% ± 9.8 |
| 4 (chat) | 48.0% ± 9.8 |
| 16 | 57.0% ± 9.7 |
| 16 (chat) | 51.0% ± 9.8 |
| 64 | 53.0% ± 9.8 |

Accuracy by area and by reasoning type. The N=64 and N=16 (chat) columns cover only the first
100 questions, so compare their rows with each other rather than with the 300-question columns.

| Area | N=4 | N=4 chat | N=16 | N=16 chat | N=64 |
|---|---|---|---|---|---|
| abstraction | 42.2% (83) | 39.8% (83) | 49.4% (83) | 52.4% (21) | 33.3% (21) |
| memory | 46.7% (15) | 53.3% (15) | 33.3% (15) | 100.0% (1) | 100.0% (1) |
| physics | 42.7% (103) | 40.8% (103) | 40.8% (103) | 38.3% (47) | 51.1% (47) |
| semantics | 60.6% (99) | 64.6% (99) | 63.6% (99) | 67.7% (31) | 67.7% (31) |

| Reasoning | N=4 | N=4 chat | N=16 | N=16 chat | N=64 |
|---|---|---|---|---|---|
| counterfactual | 66.7% (3) | 66.7% (3) | 66.7% (3) | – | – |
| descriptive | 52.3% (239) | 52.7% (239) | 53.1% (239) | 57.3% (75) | 56.0% (75) |
| explanatory | 29.8% (47) | 29.8% (47) | 29.8% (47) | 31.6% (19) | 42.1% (19) |
| predictive | 45.5% (11) | 45.5% (11) | 72.7% (11) | 33.3% (6) | 50.0% (6) |

**How the latency was measured:**
- *Uncached latency* is decode + preprocessing + vision tower + LM prefill + option scoring, for
  one question as a request with no cache would pay it.
- To keep the eval affordable, the vision features were computed once per video and shared by
  that video's questions, but the full vision time is still added to every question.
- Image preprocessing is counted twice (once in vision, once in prefill), so the latencies are
  slightly conservative.
- Breakdown at N=16: vision 23.4 s, LM prefill 9.4 s, decode 1.0 s, option scoring 0.17 s.

### What the numbers say

1. **Accuracy matches the published figure.** N=16 gives 50.3%. The Gemma 3 technical report gives
   50.6% for 4B on Perception Test MC-VQA, also with 16 frames but answering by generation
   ([Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)). Scoring options instead of
   generating loses nothing, and the image path is behaving.
2. **More frames barely help on this benchmark.** 4 → 16 frames is +1.6 points over 300 questions.
   On the shared 100 questions, 4 / 16 / 64 frames score 54 / 57 / 53%. Every difference is well
   inside the ±5.7 to ±9.8 CIs, while tokens and latency grow about 4× per step. The one hint of a
   frame effect is explanatory questions at N=64 (42% vs 30%, n=19), which is too few to rely on.
3. **The vision tower dominates cost**: 1.45 s per frame at 896 px on MPS in any dtype, about 65%
   of uncached latency. A prefix cache helps repeat questions about the same video, but the
   first question on a video still pays for the tower.
4. **Confidence is poorly calibrated.** Mean top probability is 0.92 when right and 0.89 when
   wrong at N=16. About half of all answers come with p > 0.99, and only 55–59% of those are
   correct. With the chat template, 71–74% of answers are p > 0.99 at the same accuracy.
   Treat the probabilities as rankings, not confidences.
5. **The chat template doesn't help multiple choice**: no gain on 300 questions at N=4, and
   −6 points on the shared 100. It is more overconfident. It does fix a yes-bias on noul
   (see Limitations).
6. **Strengths and weaknesses by area.** Semantics, i.e. what objects and actions are present,
   is strongest at 61–68%. Physics and explanatory ("why") questions sit near 40% and 30%.

## Tests

- `tests/test_video.py` (offline, synthetic clips with each frame's index encoded in pixels):
  - Correct count, timestamps and selected frame for short videos (16 samples from 10 frames),
    for long videos (120 s seeking across GOPs), and for N ∈ {1, 4, 16, 64}.
  - Downscaling, the path, base64, data-URI and bytes sources (same hash for all), and every
    error case.
- `tests/test_torch_video.py` (offline, tiny random Gemma 3 multimodal model, fp32):
  - Our prefill matches HF's own forward for every chunk size and frame batch size.
  - The bidirectional image mask is really applied.
  - Precomputed image features give identical results.
  - Option scores after images match full-sequence re-encoding.
  - Chunks never split an image block.
- `tests/test_video_smoke.py` (real model, opt-in via `OPENJEV_VIDEO_SMOKE=1`):
  - noul "Is a red ball visible?" on clips with and without a ball: passes.
  - A 4-option "what does this video show": passes.
  - The score crowdedness check is marked as an expected failure (see Limitations).
- The existing `test_torch_backend`, `test_package_api` and `test_portability` still pass.

```sh
uv sync --extra torch --extra video --group test
uv run python -m unittest tests.test_video tests.test_torch_video tests.test_torch_backend
OPENJEV_VIDEO_SMOKE=1 uv run python -m unittest tests.test_video_smoke
```

## Design notes

- **The bidirectional image mask relies on HF.** HF builds Gemma 3's within-image bidirectional
  mask from `token_type_ids` and indexes it by **absolute** position.
  - A chunked prefill must therefore pass token types from position 0 to the chunk end, not just
    the chunk's own slice.
  - Passing only the chunk silently degraded the mask to plain causal attention (logit error 1.1,
    against a mask effect of 1.4). The fp32 test pins this down.
- **Prompt order** is frames → state → question → options. This makes frames plus state the shared
  prefix for the step 2 cache. Images are placed by prefixing one `<start_of_image>` per frame to
  the text and letting the processor expand them, so the layout is exactly HF's.
- **The bf16/fp16 split is required on MPS.** In fp16, the Gemma 3 language model produces NaN
  for every option once images are present. The vision tower is fine in fp16, where its maximum
  feature error against fp32 is 0.19, compared with 2.2 in bf16. Speed is the same in every dtype.
- **Memory.** The KV cache from Hugging Face's `DynamicCache` keeps only the last 1024 tokens on
  sliding layers. N=64 (16.6k tokens) ran with the existing per-option cache expansion and no
  out-of-memory errors on 18 GB.

## Limitations found

- **The plain System One prompt has a strong "yes" prior with images.** On an empty white clip,
  "Is a red ball visible?" gets yes = 0.65, against 0.99 with a ball and 0.88 with no images at
  all. The chat template fixes this (0.00 vs 1.00).
  - This conflicts with the multiple-choice result, where the chat template doesn't help.
  - So the prompt layout per question type needs deciding before the API change.
- **Score questions fail zero-shot.** Asked how crowded a scene is (0 = empty / 1 = one or two
  balls / 2 = many), the model picks level 2 with p ≈ 1.0 for empty and crowded clips alike,
  with or without the chat template. This needs prompt work (or the decision head) before
  `score` with video is usable.
- **Calibration** is poor (see above). Probabilities are not confidence estimates.
- **Speed.** This is the torch-on-MPS path; MLX should be faster but won't change the
  vision-tower FLOPs. N=64 is not interactive (about 2.5 minutes per uncached question).
- **Small subsets.** The CIs are ±6 points at 300 questions and ±10 at 100. Memory (15
  questions) and the predictive and counterfactual types (14 combined) are too small to read.
- **Torch only.** `images=` raises a clear error on the MLX backend.

## Suggested next step

Step 2 as planned: port SigLIP to MLX, write our own forward with the HF mask checked against these
torch logits, and add the prefix and LRU caches.

Given the numbers, I'd default to N=8–16. More frames buy almost no accuracy here, and the
vision tower is the cost to optimise. I'd also settle the prompt-layout question (plain vs chat,
per question type) and the score-question failure before exposing `videos` on `/v1/systemone`.
