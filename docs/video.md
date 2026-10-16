# Video input (experimental)

OpenJev can score options about a short video with the zero-shot scorer. Gemma 3 has
no native video, so a video becomes N still frames passed to the model as images,
followed by the prompt text. No text is generated; options are scored as before.

This first step runs on the **PyTorch backend only**. The MLX stack (`mlx-lm`) drops
Gemma 3's vision tower when loading, so MLX video input is planned separately. There
is no `/v1/systemone` video field and no prefix cache yet.

```sh
pip install -e ".[torch,video]"
```

```python
from openjev import OptionScorer
from openjev.systemone import NoulQuestion, render_noul
from openjev.video import load_video

scorer = OptionScorer("models/gemma-3-4b-it", backend="torch",
                      dtype="bfloat16", vision_dtype="float16")
video = load_video("clip.mp4", n_frames=16)   # path, http(s) URL, data: URI or bytes
prompt, labels = render_noul("A short clip.", NoulQuestion(
    type="noul", instructions="Is a red car visible in this video?"))
res = scorer.score(prompt, labels, norm="sum", images=video.frames)
print({r.option: r.probability for r in res})
```

## Frames

- Frame *i* of *N* is the frame on screen at the midpoint of the *i*-th of *N* equal
  time bins: `(i + 0.5) * duration / N`. Audio is ignored. Short videos may repeat
  frames when N exceeds the frame count.
- Each frame is resized to 896×896 by the Gemma 3 processor and costs about 261
  prompt tokens (256 image tokens plus the start/end markers). `max_side` only
  downscales at decode time; the vision tower always sees 896×896.
- `VideoLimits` caps file size, duration and frame count (64 by default). Bad input
  raises `VideoError` (a `ValueError`) with a message safe to return to clients.

## Gemma 4 (E4B)

Gemma 4 checkpoints (e.g. `models/gemma-4-E4B-it`) go through the same OpenJev scorer.
`scorer._engine.frame_mode` picks how the frames enter the model:

- `"images"` (default, like Gemma 3): each frame is a separate image (`<|image>` + soft
  tokens + `<image|>`), resized with its aspect ratio kept to 280 tokens. 16 frames come to
  about 4.3k prompt tokens, against 4.2k for Gemma 3.
- `"video"`: Gemma 4's native video input. Each frame gets its `mm:ss` label and 70 tokens.
  Pass the frame times:

```python
scorer = OptionScorer("models/gemma-4-E4B-it", backend="torch",
                      dtype="bfloat16", vision_dtype="float16")
res = scorer.score(prompt, labels, norm="sum", images=video.frames)       # frames as images
scorer._engine.frame_mode = "video"
res = scorer.score(prompt, labels, norm="sum", images=video.frames, timestamps=video.timestamps)
```

`scorer._engine.frame_soft_tokens` overrides the per-frame budget (70, 140, 280, 560 or 1120).

- The frames are preprocessed with the PIL image processor, which runs the same steps
  as HF's video processor, so torchvision is not needed.
- The audio tower is dropped at load time. The per-layer embedding table (2.8B
  parameters, only ever indexed) stays on the CPU, which keeps E4B at about 10 GB on MPS.
- On MPS the prefill frees the allocator cache after each chunk. Without that, the growing
  KV cache left a freed block of a new size every chunk, and 64 frames (17k tokens) ran out
  of memory on 18 GB.
- E-series models attend causally to vision tokens, with no bidirectional block mask.
  Chunked prefill is therefore exact. `tests/test_torch_video_gemma4.py` checks it, and
  option scoring, against HF's own forward on a tiny random model with KV sharing.
- The chat template's user turn (`<|turn>user`) is detected like Gemma 3's
  `<start_of_turn>user`, so `chat=True` works for both.

## Precision on Apple silicon

Gemma 3's language model overflows to NaN in float16 on MPS when images are present,
so use `dtype="bfloat16"`. The SigLIP vision tower is fine in float16 and tracks
float32 much more closely than bfloat16 does, hence `vision_dtype="float16"`.

## Attention mask

Gemma 3 attends bidirectionally within each image and causally elsewhere. The torch
path builds that mask through Hugging Face from `token_type_ids` and prefills in chunks
that never cut an image in half; tests check it against Hugging Face's own forward.

## Evaluation

`video_eval/perception_test.py` runs a zero-shot multiple-choice eval on a seeded
subset of the Perception Test validation split (CC BY 4.0), fetching only the needed
videos from the 75 GB archive with HTTP range requests:

```sh
python -m video_eval.perception_test fetch --questions 300
python -m video_eval.perception_test run --frames 16
python -m video_eval.perception_test run --frames 16 --model models/gemma-4-E4B-it
python -m video_eval.perception_test run --frames 16 --model models/gemma-4-E4B-it --frames-as video
python -m video_eval.perception_test report runs/video_eval/perception_test*.jsonl
```

Each `run` logs itself under `runs/video_eval/`: per-question rows (`<stem>.jsonl`, which a
rerun resumes), the console (`<stem>.log`), the settings, versions and git commit
(`<stem>.meta.json`), one summary line per invocation (`history.jsonl`), and a regenerated
`report.md` comparing every run in the folder across models. These are written even when a
run is interrupted.
