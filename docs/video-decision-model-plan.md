# Video decision model plan

Running record for the `video-decision-model` branch. This page holds the overall plan; each
stage is discussed and decided when we reach it, and its decisions and results are recorded
under it (same rules as `docs/video-training-plan.md`: design fixed before results, newest
decisions win).

## Goal

One fine-tuned video LLM that:

1. does well on video in general, not one benchmark: measured on benchmarks it never trains on;
2. makes calibrated decisions through the `openjev decision` contract (state = video, question,
   candidates → probabilities);
3. is shown to the team on its benchmark clips (question, answer probabilities, truth).

## Decided (2026-11-09)

- **Backbone: Qwen3-VL-4B** (Apache 2.0) with its own vision encoder (SigLIP 2-based, trained
  jointly with the LLM). Chosen for general video scores at its size (Video-MME 69.3, MLVU 75.3,
  LVBench 56.2, self-reported), time handling (interleaved MRoPE, text timestamps) and
  token-efficient video (two frames per patch, 2×2 merge). 8B is too tight for LoRA on 18 GB.
- **Encoder frozen; LoRA on the LLM.** No added encoders (step 9 v2 of the previous plan: they
  added nothing).
- **Decisions come from the LLM's own scoring** (letters or candidate likelihoods) plus a fitted
  temperature. A separate head only if scoring hits a ceiling, and then trained jointly with
  the LoRA, not on frozen features (step 11 of the previous plan).
- **Training on the Mac only** (MLX).
- **Research-only datasets are allowed** (the model is not sold). Record which were used in case
  weights are ever published.
- **Trained models only (2026-11-16):** no runs on the downloaded model without an adapter, now or
  later; `--adapter` is required in `vdm_eval`, and training no longer scores the dev
  sets at step 0. Code removed then (results below stay as history): `vdm_stage0` and the Gemma
  reference in `vdm_eval`.

## Storage (same rule as before)

- Everything bulky lives on the external drive: `/Volumes/One Touch/open-jev-video-model-data`
  (`video_train.storage.DATA_ROOT`). New top-level folders for this plan: `vdm/` (datasets,
  manifests, results, checkpoints of this plan) next to the existing ones.
- **Every model version and checkpoint is saved to the drive.** Models are copied to the Mac
  only while a run needs them and removed afterwards (`video_train.models`: `stage`, `release`).
- Random-access feature caches stay on the internal disk (the drive is an HDD, ~18 ms per
  random read); report their size before building them.
- Big downloads: give the terminal commands; the user runs them.
- Free space on 2026-11-09: drive 297 GB, Mac 131 GB.

## Stages

Each stage lists what it must answer; the details are decided when we get there.

### Stage 0: feasibility
Does Qwen3-VL-4B run and train here? mlx-vlm support at the pinned 0.7.8 (or a bump), 8-bit vs
bf16 agreement check, LoRA memory with video on 18 GB, seconds per training step and per question.
Video input format (frames, resolution, fps) verified against the official processor.

**Design (2026-11-09).** `video_train/qwen3vl.py` (inputs, scoring, LoRA, loss) and
`video_train/vdm_stage0.py` (`format`, `precision`, `speed`, `report`); results in
`vdm/results/stage0/`.

- **Weights:** `Qwen/Qwen3-VL-4B-Instruct@ebb281e` (bf16, 8.9 GB) on the drive; the 8-bit MLX
  model is converted locally from it with `mlx_vlm convert` (no third-party quant).
- **Inputs:** frames decoded by `openjev.video` (bin midpoints), then the official transformers
  Qwen3-VL processor, chat template, prompt ending "Answer with the option's letter from the
  given choices directly." Default 16 frames at 192 vision tokens per 2 frames (16:9 video:
  1,440 tokens). The processor classes are named explicitly: importing mlx-vlm's qwen3_vl
  re-registers `AutoProcessor` to mlx-vlm's own class, which ignores the pixel budget (12,064
  vision tokens instead of 1,440 on the same video).
- **Readout:** float32 logits of the option letters at the last position (tied head rows).
- **Training format: all questions of a video packed into one multi-turn conversation**
  (video, Q1, A1, Q2, A2, ...), cross-entropy on the assistant tokens (answer + `<|im_end|>`).
  One forward and backward per video; replaces the previous plan's KV-cache grouping, which
  does not carry over to MRoPE. Later questions see earlier answers: accepted, since
  LLaVA-Video's own data is multi-turn.
- **Checks:** `format`: grid and one `<t seconds>` per temporal patch match the processor's own
  PyAV decode, and the bf16 and 8-bit folders' processors give identical inputs. `precision`:
  torch bf16 (transformers, MPS) vs MLX 8-bit on `manifests/precision100.jsonl`, same gate as
  before (>= 98% agreement on questions torch decides by >= 0.1, |P(correct) shift| <= 0.01).
  `speed`: LoRA r8 step on 9 Perception Test train videos at 16 and 32 frames (peak memory,
  s/video). Verified before the real weights on a 4-layer random Qwen3-VL: forward matches
  mlx-vlm's to 3e-6, checkpointed gradients equal plain ones exactly, gradients reach all LoRA
  matrices.

**Results (2026-11-10): pass.** Precision (103 questions): torch bf16 66.0%, MLX 8-bit 68.9%;
agreement 99.0% of the 99 questions torch decides by >= 0.1, P(correct) shift +0.0034. LoRA
r8 step: 16 frames 19.4 s/video, 7.4 GB peak (1,711 prompt tokens); 32 frames 36.5 s/video,
8.2 GB. **Decision: train at 16 frames, one pass over the full mix (~58 h).** 32 frames is
not scored (Gemma 4 showed no gain from more frames, step 3 of the previous plan).

### Stage 1: evaluation suite and zero-shot baseline
Fix the suite before any training: in-domain dev sets, never-trained-on benchmarks (the
generality check), a grounding set, speed. Score Qwen3-VL-4B zero-shot (and Gemma 4 E4B /
s8_A_ep2 as references). Test splits are held back for one final evaluation.

**In-domain dev and test fixed (2026-11-09)** by `video_train/vdm_splits.py` in
`vdm/manifests/` (`<ds>_dev`, `<ds>_test`, `<ds>_pool`, `split_meta.json` with input hashes).
Held out by video, seed 0, at most 2 questions per held-out video; stage 2 samples training
data from the pools only. `vdm_splits check` passes (no held-out video in any pool).

| dataset | dev | test | from |
|---|---|---|---|
| Perception Test | 2,003 q / 531 v | 2,000 q / 562 v | previous plan's `manifests/dev`, `test` (unchanged) |
| NExT-QA | 1,003 q / 115 v | 900 q / 450 v | val: dev = previous `nextqa_val1000`, test = the other val videos |
| STAR | 500 q / 252 v | 500 q / 252 v | train json; Charades videos held out jointly with Charades-STA |
| Charades-STA | 500 q / 300 v | 500 q / 302 v | train txt (grounding; metric decided with the conversion) |
| CLEVRER | 500 q / 250 v | 500 q / 250 v | train json, raw (descriptive answers open; others multi-answer) |
| LLaVA-Video mc | 300 q / 150 v | 300 q / 150 v | only from videos in chunk 1 (760 with mc questions) |

- Rows keep the raw annotation; the prompt format is stage 2's conversion. `video` points
  where stage 2 extracts the file (`vdm/videos/...`).
- Perception Test test was already scored once (step 12 of the previous plan) for the old
  model; it is reused as this plan's final test (nothing here is tuned on it).
- **Download problems found 2026-11-09:** `lmms-eval/NExTQA` holds only val + test videos
  (1,570; none of the 3,870 train videos). Fixed 2026-11-10: the official NExTVideo zip (Google
  Drive, linked from github doc-doc/NExT-QA) is at `vdm/raw/nextqa/NExTVideo_official.zip`
  and holds all 3,870 train videos (checked against `train.csv`).
  LLaVA-Video chunk 2 is truncated (3.75 of 5.30 GB) and chunk 3 is missing. Fixed
  2026-11-10: both re-downloaded at `6d8c562`, sha256 equal to the HF etags.
- **Benchmarks chosen 2026-11-10** (`video_train/vdm_data.py bench`, `fetch-bench`):
  Video-MME short (300 videos < 2 min, 900 questions, no subtitles; 3.1 GB read out of the
  remote zips) and TempCompass multiple choice (410 videos, 1,580 questions; 0.4 GB). MLVU and
  LVBench left out: 3 min - 2 h videos at 16 frames say little. Used whole (never tuned on).
- **Zero-shot run** (`video_train/vdm_eval.py`): prompts from `vdm_data.turns` (the training
  format); the prompt up to the last video token is cached once per video, then each question
  runs only its own tokens (same answer as the full pass, max prob diff 0.013 on 5 questions;
  ~0.3-0.5 s per question instead of 3-4 s). CLEVRER multi-answer: one yes/no per option.
  CLEVRER descriptive and Charades-STA: greedy answer; Charades-STA metric mIoU and R@0.3/0.5/0.7.
  STAR: the clip is given as text ("From 3.2 to 9.4 seconds: ...") over the whole video, so not
  comparable to STAR's published (clip-only) numbers.
- **Zero-shot results (2026-11-10)**, Qwen3-VL-4B MLX 8-bit, tag `zeroshot_f16` (16 frames,
  192 tokens per 2 frames). s / question = model compute only.

  | set | questions | accuracy | other | s / q |
  |---|---:|---:|---|---:|
  | perception_dev | 2,003 | 66.6% ± 2.1% | | 2.35 |
  | nextqa_dev | 1,003 | 79.0% ± 2.5% | | 0.96 |
  | star_dev | 500 | 72.4% ± 3.9% | | 2.58 |
  | charades_sta_dev | 500 | 37.4% ± 4.2% (R@0.5) | mIoU 0.390, R@0.3 61.8%, R@0.7 15.8% | 3.03 |
  | clevrer_dev | 500 | 61.6% ± 4.3% | descriptive 76.9%, per option 60.6% | 2.01 |
  | llava_video_dev | 300 | 82.0% ± 4.3% | | 2.43 |
  | videomme_short | 900 | 70.2% ± 3.0% | benchmark | 1.78 |
  | tempcompass_mc | 1,580 | 70.8% ± 2.2% | benchmark | 1.47 |

  Weakest: Charades-STA (timestamps, R@0.7 15.8%) and CLEVRER (physics), both in the mix.
  Gemma 4 E4B / s8_A_ep2 references not run (`vdm_eval` is Qwen-only).

### Stage 2: general training data
Pick, download and convert the Stage A datasets (candidates below); manifests on the drive;
overlap checks against every evaluation set.

**Mix decided 2026-11-09: ~37k questions, ~10.5k videos.** Size set by Mac training time: the
previous plan measured ~6-8 s per video per pass (questions of a video share one encoding), so
~10.5k videos ≈ 20 h per pass (to re-measure for Qwen3-VL-4B in stage 0). Re-measured: 19.4 s
per video, so ~58 h per pass; kept at full size (2026-11-10).

**Built 2026-11-10** (`vdm_data mix`, `vdm/manifests/train_mix.jsonl`, one row per video with
its packed turns): 36,892 questions, 10,829 videos, 46,588 turns. Perception Test 7,392 / 1,955;
NExT-QA 7,500 / 2,502; STAR 7,000 / 1,853 (<= 4 per video); CLEVRER 6,000 / 1,500 (one per
type in turn); Charades-STA 3,000 / 1,519; LLaVA-Video 6,000 / 1,500 from chunks 1-3 (mc and
open-ended in turn; only 1,426 mc, as most videos have one mc question). `vdm_data overlap`:
no video of any dev, test or benchmark set in the mix (LLaVA-Video's 300 held-out videos
dropped; no Video-MME YouTube ID found).

| dataset | take | share | why |
|---|---|---|---|
| Perception Test train | all 7.4k questions, 1,955 videos | 20% | our area breakdown; on the drive |
| NExT-QA train | 7.5k, ≤ 3 per video (~2,500 videos) | 20% | cause and effect, timing |
| STAR train | 7k (~2,000 clips) | 19% | interactions, order of events, prediction, feasibility |
| CLEVRER train | 6k (~1,500 videos), balanced over its 4 question types | 16% | physics and counterfactuals (weakest area); rendered scenes |
| Charades-STA train | 3k (~1,000 videos) | 8% | finding when an event happens (timestamps) |
| LLaVA-Video-178K, 30-60 s YouTube, chunks 1-3 | 6k (~1,500 videos) | 16% | broad open-ended and multiple-choice questions |

- No dataset above 20%; few questions per video, so the model sees many videos. Weighted toward
  time and physics.
- Not now: egocentric data (Ego4D needs an agreement) and image grounding. Revisit if stage 1
  shows a gap.
- **Overlap check before training:** by video ID against every evaluation set. LLaVA-Video's
  YouTube clips may overlap Video-MME; its academic subsets (which contain Perception Test and
  NExT-QA videos) are not used.
- **Sources** (raw files under `vdm/raw/` on the drive, ~52 GB; HF files at pinned revisions):
  NExT-QA `train.csv` (github doc-doc/NExT-QA) + videos (`lmms-eval/NExTQA@a0d7729`, 6.5 GB);
  STAR `STAR_train.json` (star-benchmark S3, 0.4 GB); Charades v1 480p (AI2 S3, 16.3 GB, for STAR
  and Charades-STA); Charades-STA `Charades_sta_train.txt` (github Alvin-Zeng/DRN, 12,408 rows);
  CLEVRER `train.json` + `video_train.zip` (data.csail.mit.edu, 12.4 GB);
  LLaVA-Video-178K (`lmms-lab/LLaVA-Video-178K@6d8c562`, mc/oe json + 3 video chunks, ~16 GB).
- Evaluation data does not all fit next to this (Video-MME 101 GB, LVBench 62 GB, MLVU 206 GB):
  stage 1 picks a subset.

### Stage 3: Stage A training (general video)
LoRA on the mix. Gate: in-domain dev improves and never-trained-on benchmarks stay within their
resolution.

**Passes (2026-11-09): one pass first.** ~10.5k videos at 8 per step ≈ 1,300 steps, 2.7x the
video exposures of s8_A_ep2 (2 passes over 1,955 videos). LoRA rank 8 / alpha 16 on all LLM
linear layers, AdamW 1e-4, 5% warmup then cosine, clip 1 (the s7/s8 recipe; rank 32 lost in
step 8). If the dev curve is still rising at the end, a fresh 2-pass run follows (schedule over
both passes, as s8_A_ep2), not a continuation. Hours per pass from stage 0's `speed`.

**Training run ready (2026-11-10):** `python -m video_train.vdm_train train` (tag `stageA_ep1`,
checkpoints and logs in `vdm/checkpoints/stageA_ep1/`, resumable every 25 steps).
- 10,829 videos at 8 per step = 1,354 steps, warmup 67; 16 frames, 192 tokens per 2 frames.
- Loss: mean over the step's assistant turns of each turn's mean token CE, so a 15-token
  LLaVA-Video answer weighs as much as a one-letter one (a plain token mean would let the
  open-ended answers dominate: LLaVA-Video answers average 15 tokens, every other source 1-9).
- Dev curve at step 0 and every 150 steps: 15 videos per dev set (~180 questions, ~8 min),
  accuracy per set and NLL of the right option on multiple choice; `train_log.jsonl` has loss
  per source.
- After training: `vdm_eval run --set all --tag stageA_ep1 --adapter <checkpoint dir>`.
- `vdm_eval` also takes `--model gemma` (Gemma 4 E4B in the previous plan's format, with or
  without s8_A_ep2 via `--adapter`) for the reference scores; not run (not needed now).
- Smoke test (2 steps, 16 videos): loop, save, dev curve and `vdm_eval --adapter` work; peak
  8.0 GB; refactored Qwen eval reproduces `zeroshot_f16` rows exactly. Its 436 s/step (vs ~155
  expected) was measured while another MLX job shared the GPU: check s/step in the first log
  lines of the real run.

### Game stages removed (2026-11-17)
The open-world game (old stages 4-6: `openworld/`, `vdm_game`, `vdm_cams`, the closed-loop
and 4-camera runs) was removed from the code. The model never trained on it; best zero-shot
result 20% success on 30 landmark episodes (`cams_stageA_ep1_v2`) against 93% with the map's
answers, so the questions, not the rule, were the limit. Results stay on the drive
(`vdm/game/`, `vdm/results/stage5/`); `git log` has the code and the old write-up. The demo is
now the general video model on its benchmarks, for the team.

**Stage A full eval (2026-11-18):** perception 76.8%, NExT-QA 81.5%, STAR 78.8%, Charades-STA
R@0.5 61.2% (mIoU 0.529), CLEVRER 80.0%, LLaVA-Video 87.0%, Video-MME short 69.9%, TempCompass
73.0% (`vdm/results/stage1/stageA_ep1/summary.md`).

### Stage B: dashcam incident triage (decided 2026-11-20)
Demo use case: a dashcam event clip comes in live, the model decides what happened, its cause,
whether it was avoidable. Data: SUTD-TrafficQA (10,080 videos, 62,535 multiple-choice questions;
annotations on Zenodo 7431011, raw videos by request; licence not stated, research use).
- Splits (`vdm_splits trafficqa`): dev 500 questions from R2_train videos, test 1,000 from
  R2_test videos (final run only), pool = the other R2_train videos; <= 2 questions per video.
- First `vdm_eval run --set trafficqa_dev` with stageA_ep1; train only if it is weak.
- Training, if needed: continue from stageA_ep1 (not a new LoRA from the base model) on
  TrafficQA pool + ~20% Stage A replay; check that Stage A sets and benchmarks hold.
- **stageA_ep1 on trafficqa_dev (2026-11-21): 66.8% ± 4.1%** (chance 38%). By options: 2 → 73.2%
  (265 q), 3 → 64.7% (51), 4 → 58.2% (184). By type: U 70.3% (290), C 67.5%, A 62.9%, F 62.5%,
  R 60.0%, I 55.3% (30-70 q each, ± ~15 points). Weak: train.
- **Stage B mix built 2026-11-21** (`vdm_data stageb`, `vdm/manifests/stageB_mix.jsonl`): 4,000
  TrafficQA pool videos, <= 4 questions each with types in turn (14,855 questions; U 57%, A 17%,
  C/F/I/R 5-7% each) + 1,000 whole videos replayed from `train_mix.jsonl` (3,418 questions);
  5,000 videos, 19,204 turns. `vdm_data overlap`: no evaluation video in it.
- **Run:** `vdm_train train --tag stageB_ep1 --mix vdm/manifests/stageB_mix.jsonl --lr 5e-5
  --init-adapter <DATA_ROOT>/vdm/checkpoints/stageA_ep1`: one pass (625 steps of 8 videos), half
  the Stage A learning rate, same rank / alpha / warmup / cosine; optimizer starts fresh. Dev curve
  includes trafficqa_dev. Smoke run: ~16 s per video, so ~22 h plus dev checks.
- **Stage B full eval (2026-11-22):** all 625 steps in ~22 h. trafficqa_dev **79.6%** (+12.8,
  paired p=5e-9; U 83.1%, A 84.3%, F 81.2%, C 72.5%, I 65.8%, R 60.0%). Stage A sets hold
  (paired sign test, no set p<0.05): perception 75.9%, NExT-QA 81.9%, STAR 79.6%, Charades-STA
  R@0.5 58.6% (mIoU 0.515), CLEVRER 79.6%, LLaVA-Video 84.3% (p=0.057), Video-MME short 70.4%,
  TempCompass 72.3% (`vdm/results/stage1/stageB_ep1/summary.md`).
- **Main adapter from 2026-11-22: `stageB_ep1`** (`<DATA_ROOT>/vdm/checkpoints/stageB_ep1`),
  used for stages 7 and 8. Watch LLaVA-Video and Charades-STA, which both slipped ~2.5 points.

### Stage 7: calibration and decision API
Temperature per task type; the `openjev decision` contract for video.

### Stage 8: final evaluation and demo
Held-out test sets scored once; latency (model compute only) and decode reported separately;
the team demo.
- **Showcase reel (2026-11-23, branch `showcase`):** `python -m video_train.showcase --adapter
  <stageB_ep1 dir>` plays held-out clips at real speed and feeds each 2-frame group to the model
  as it plays (vision encoder + KV cache extension, ~0.6 s per group); the answer shows 0.6-1.5 s
  after a 5-12 s clip ends. Live probabilities match the stored eval within 0.006. Output:
  `vdm/showcase/<tag>_<set>_s<seed>.mp4` (seed 0, TrafficQA: 4 of 6 right).

## Training data candidates (for stage 2; not decided)

General video (Stage A):

| dataset | what it teaches | notes |
|---|---|---|
| Perception Test train | physics, memory, abstraction, semantics | on the drive (7,392 questions, 1,955 videos) |
| NExT-QA train | cause and effect, timing | train videos in `NExTVideo_official.zip` |
| STAR | situated reasoning: interaction, sequence, prediction, feasibility | Charades videos |
| CLEVRER | physics, counterfactuals | rendered scenes |
| Charades-STA | when an event happens (temporal grounding) | same Charades videos as STAR |
| LLaVA-Video-178K (slice) | broad open-ended and multiple-choice video QA | very large; a slice only |
| Egocentric video (e.g. Ego4D / EPIC-KITCHENS) | first-person view | Ego4D needs a signed agreement |
| Image grounding (e.g. RefCOCO) | pointing at objects | |

Never trained on (generality check): Video-MME, MLVU, LVBench. MVBench is not a clean check: it
is built partly from Perception Test, CLEVRER and STAR.
