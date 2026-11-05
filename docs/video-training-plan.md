# Video training plan: decisions and changes

Running record of changes to the original Gemma 4 E4B video training plan (steps 0-12), with
the reason for each. Newest decisions win. Data, results and `history.jsonl` live on the
external drive (`video_train/storage.py`).

## Step 0: framework and checks (done, 2026-10-16)

- **Training stack: mlx-vlm 0.7.8 as a library**, pinned (`video-train` extra), our own loss
  and loop (`video_train/mlx_model.py`). PyTorch on MPS was ~3x slower per LoRA step and does
  not fit 280-token frames in memory.
- **Base precision: MLX 8-bit by default**, 4-bit only where 8-bit does not fit. 4-bit changed
  22% of zero-shot answers against torch bf16 (102 dev questions); 8-bit changed 1%.
- **Dev evaluations run on MLX 8-bit** (gate: >= 98% answer agreement with torch bf16, passed
  at 99.0% for images@280; re-checked for the video format before step 2). Step 12 (test)
  stays on torch bf16.
- **Gate amended for video@70 (2026-10-16).** The original rule failed twice: 97.1% (n=103) and
  96.4% (n=302, 11 disagreements). 10 of the 11 are torch near-ties (top-two margin <= 0.12), 4
  of them exact ties inside torch bf16 itself, decided by its tie-break; the mean absolute
  probability difference (0.016) equals that of the images@280 run that passed, and MLX does not
  shift P(correct) (mean -0.0004, lower on 52% of questions). Video@70 just has more near-ties, so
  the same noise flips more answers. A float32 reference (~32 GB) does not fit in 18 GB, and torch
  cannot load the video adapter, so a torch dev path would need that port first.
  **New rule:** >= 98% agreement on questions torch decides by a margin >= 0.1, and mean shift in
  P(correct) within +-0.01 (`video_train/precision_check.py`). Result: **PASS**, 98.9% of 282
  decided questions (3 disagreements), shift -0.0004.
  **Caveat:** the rule was chosen after seeing the data; two of the three remaining disagreements
  sit just above the cut-off (0.109, 0.119), and on the 11 disagreements torch was right 7 times,
  MLX once (4 of torch's 7 are tie-breaks; p ~= 0.07). Watch for a systematic gap: if MLX dev and
  torch test disagree by more than dev's resolution (~1.5-2 points) at step 12, revisit.
- **Video layout verified** against the official Gemma 4 processor: identical token ids
  (including `mm:ss` timestamps), positions, pixels within 2/255
  (`tests/test_gemma4_processor_parity.py`).
- **Encode once per video**: a video's frames and prompt prefix are encoded once and all its
  questions scored on that cache, for training (verified: loss and gradients equal in float32,
  `tests/test_mlx_grouping.py`) and for evaluation (agrees with one pass per question on
  98% of answers, both disagreements near-ties; must be re-shown for each new format).

## Step 1: data (done, 2026-10-16)

- **Training data: Perception Test train split only** (7,392 questions, 1,955 videos).
  NExT-QA, STAR and CLEVRER are not used for training.
- Perception Test validation split by video, seed 0: dev 501 questions (141 videos), test
  2,000 (562 videos, includes the earlier 300-question subset); the rest is never used.
- **Dev grown to 2,003 questions (531 videos), 2026-10-16**, with whole videos from the unused
  pool (seed 1); the original 141 dev videos are kept and test is unchanged. Reason: at ~50%
  accuracy, 501 questions only separate two models (paired) when they differ by ~3-3.5 points
  and had 28 memory questions; 2,003 gets that to ~1.5-2 points and 80 memory questions.
  Areas: physics 767, semantics 589, abstraction 567, memory 80. The old manifest is kept as
  `manifests/dev_v501.jsonl`; the 102 precision-check questions are frozen in
  `manifests/precision102.jsonl`.
- **Small effects:** differences below what dev can resolve are treated as ties, and the cheaper
  setting wins.
- **Second, out-of-domain eval: NExT-QA validation**, fixed 1,003-question subset (115 videos,
  seed 0). Reported next to Perception Test dev at every step. Annotations MIT; videos have no
  explicit license (YFCC100M clips), used for private research evaluation only.

## Checks after training (replaces rule 5b)

- **Adapter used only for video requests**, chosen per request (context variable, survives
  the server's threadpool; `openjev/adapter_scope.py`, `tests/test_server_adapter_scope.py`).
  Text and image requests stay on the base model, so text forgetting cannot reach them:
  **no ARC check**.
- **Badge set**: cheap smoke test that nothing broke.
- **A-OKVQA** (500 validation questions, paired before/after): information on whether the
  adapter generalises beyond video; not a deployment gate. Base 8-bit: 77.2%.
- Video is MLX-only for now; the torch backend cannot load the adapter (README).

## Frame format (changed 2026-10-16)

- **Old:** step 2 baseline at 16 frames as images, 280 tokens/frame (4.3k tokens/question).
  280 was not tested to be better: it was carried over from Gemma 3, whose fixed
  896x896 / ~261-token images it matches.
- **New: baseline and default training format = 16 frames in Gemma 4's native video layout,
  70 tokens/frame, with timestamps** (1.2k tokens/question). On the same 300 questions:
  video@70 50.0% vs images@280 51.7% (difference inside the +-5.7 point interval), at 3.4x
  fewer tokens and ~4x faster LoRA steps.
- **Step 3 (revised)** starts from video@70 and tests, one change at a time against it, on dev
  (and NExT-QA):
  - more detail per frame: 140, 280, 560 tokens/frame (video layout);
  - more frames: 32 x 70;
  - the old setting: 16 frames as images @280;
  - frame selection at 16 x 70: change-based, and question-aware (SigLIP 2 / PE-Core scoring
    of ~32 candidates).
  The chosen setting becomes the training format; changing the format later means re-caching
  features and retraining the LoRA (frame *selection* alone can change without retraining).
- **Frame selection, as built (2026-10-16)** (`video_train/frame_select.py`,
  `evaluate run --select change|question`): decode 32 uniform candidates, keep 16 in time order
  with their own timestamps.
  - `change`: 16 frames at equal steps of cumulative change (64x64 grayscale differences
    between neighbouring candidates, plus a floor of 25% of the mean change so static parts
    keep frames). Uniform change gives uniform frames. One selection per video.
  - `question`: SigLIP 2 (`siglip2-base-patch16-224`) scores candidates against the question
    and question + each option (best cosine similarity); the best frame of each of 8 equal time
    groups is kept for coverage, plus the 8 best remaining. One selection per question, so
    encode-once only shares an encoding between questions that pick the same frames.
    PE-Core is not implemented; add it only if SigLIP 2 selection helps.
  - Selection time counts as model latency; decoding 32 candidates counts as decode time.
    Rows record `frame_idx`, `frame_times`, `select_s`.

## Step 3: results (done, 2026-10-18)

**Decision: training format = 16 frames, video layout, 140 tokens/frame** (2.2k tokens/question,
up from 1.2k). Uniform frame sampling stays; no frame selection.

Perception Test dev (2,003 questions, MLX 8-bit, encode-once); "net" = questions gained minus
lost against video@70, paired (McNemar) p:

| Setting | Accuracy | Net vs video@70 | p | NLL | ECE | Latency/question |
|---|---|---|---|---|---|---|
| video@70 (baseline) | 53.0% | - | - | 1.590 | 0.307 | 1.2 s |
| change-based selection @70 | 52.3% | -14 | 0.27 | 1.611 | 0.311 | 1.1 s |
| question-aware selection @70 | 52.9% | -2 | 0.87 | 1.610 | 0.307 | 6.6 s |
| 32 frames @70 | 52.8% | -4 | 0.76 | 1.622 | 0.312 | 1.9 s |
| **video@140** | **54.7%** | **+34** | **0.02** | 1.582 | 0.302 | 2.0 s |
| video@280 | 54.9% | +39 | 0.02 | 1.602 | 0.307 | 5.5 s |
| video@560 | 55.5% | +50 | 0.004 | 1.604 | 0.304 | 9.3 s |

- **Why 140, not 280 or 560:** against 140, 280 is +5 net (p=0.70) and 560 +16 (p=0.29), ties
  under the small-effects rule, at 2.7x and 4.6x the latency and no better NLL/ECE. Training
  cost scales with tokens too.
- **Why 140 over 70:** +1.7 points sits at the edge of dev's resolution (~1.5-2), so it was
  confirmed out of domain before choosing. NExT-QA (1,003 questions): video@70 64.8%, video@140
  **68.9%**, +41 net (62 gained, 21 lost, p < 0.001), NLL 1.428 -> 1.360, ECE 0.226 -> 0.197.
- **Frame selection and more frames do not help** (all ties or worse), so PE-Core selection is
  not built; `--select` stays available for later.
- **Encode-once re-shown at 140** on the 102 precision questions: 99.0% agreement with one pass
  per question (1 disagreement, max probability difference 0.076), same as video@70 (99.0%).
- **Not run:** images@280 on the full dev set (video@70 replaced it as the baseline, and
  video@560 covers more tokens per frame); the 8-bit-vs-torch precision gate at 140
  (`precision_check.py` has no `--soft-tokens`; 8-bit passed at 70 and images@280, the
  70-vs-140 comparison is on the same 8-bit model, and step 12's torch test is the check).

## Step 4: teacher gate (done, 2026-10-19)

**Decision: d3-mini is the teacher** (`video_train/teacher.py`, torch on MPS, its own video
reading and readout). Run on a 200-video dev sample (712 questions, seed 0), not the full 2,003:
the gap is far outside dev's resolution, so the rest was not worth ~10 h.

| | Accuracy | NLL | Brier | ECE | conf >= 0.8: coverage / accuracy | Latency/question |
|---|---|---|---|---|---|---|
| video@140 (student) | 53.4% +- 3.7 | 1.590 | 0.731 | 0.321 | 67% / 61.3% | 2.1 s |
| d3-mini | 70.9% +- 3.3 | 0.667 | 0.391 | 0.063 | 50% / 89.2% | 53.2 s |

- Paired: d3-mini gains 184, loses 59 (net +125, p = 1.1e-15). Better in every area:
  abstraction 42.3 -> 64.2%, memory 46.2 -> 69.2%, physics 44.8 -> 61.9%, semantics 75.1 -> 88.5%.
- **No sign of training on Perception Test train:** 100-video train sample 70.9% +- 4.6
  (n=371), same as dev; confidence 0.77 vs 0.76.
- Latency depends on other running apps (shared memory and GPU): ~53 s/question with apps
  open, ~28 s with them closed.

## Step 5: teacher labels (done, 2026-10-24)

- **All 7,392 train questions (1,955 videos) labeled**, not a subset: a subset would rank the
  step 7 variants at a smaller data size, and the advantage of teacher labels usually shrinks
  as data grows, so the ranking could flip at full scale.
- Rows: `results/evals/train/d3mini.jsonl` (question ID, `probs` in manifest option order,
  matched by option text through d3-mini's criteria keys; option text is in the manifest).
  Checked: one row per manifest question, option counts match, probabilities sum to 1, no NaN.
- Train labels: accuracy 70.3% +- 1.0, NLL 0.669, ECE 0.060; confidence >= 0.8 on 49.9% of
  questions at 88.6% accuracy. By area: semantics 85.2%, memory 68.3%, abstraction 65.4%,
  physics 62.0%. Mean 28.6 s/question.

## Step 6: feature cache (done, 2026-10-25)

- `video_train/cache_features.py`: one file per video with the projector output (bf16, exactly as
  `mlx_model.vision_features` returns it), the prefix token ids and the frame timestamps, in
  step 3's format (16 frames, video layout, 140 tokens/frame, MLX 8-bit). A question's own ids
  come from its prompt alone (`suffix_ids`), so training never decodes video.
- **On the internal disk** (`~/.cache/openjev-video/features/video140`), against the
  "big files on the drive" default: training reads it at random and the HDD seeks at ~18 ms.
  It is rebuildable; delete it after step 8. ~9.8 MB/video, ~24 GB for train + dev (2,486 videos).
- **Exact:** on 20 dev videos (68 questions), cached features score identically to step 3's
  video140 rows, which encoded the video fresh (100% agreement, max probability difference 0,
  features bit-identical, suffix ids identical).
- ~2.8 s vision + ~1.8 s decode per video: ~3 h for all videos.

## Step 7: target selection (done, 2026-10-28)

`video_train/distill.py` (`prepare`, `train`, `grid`, `report`); loss tested in
`tests/test_distill_loss.py`.

- **Loss** per question over the option-letter probabilities: α·CE(correct answer) +
  (1−α)·KL(teacher_T ‖ student), teacher softened as p^(1/T) renormalized, T = 1.5, student
  untempered (so no T² factor). A α=1, B α=0.5, C α=0 (all questions kept).
- **Shared settings:** all 7,392 train questions, step 6 cache, seed 0 (same LoRA initialization
  and video order for every run), 1 pass, 8 videos per optimizer step (245 steps), AdamW with 5%
  linear warmup then cosine decay, gradient norm clipped at 1, LoRA rank 8 / alpha 16 on the
  language model's linear layers.
- **Changed 2026-10-25: one shared learning rate (1e-4), 3 runs, not a 2-3-rate search per
  variant (6-9 runs).** Reason: a training step costs ~14-30 s per video (language-model forward
  and backward over ~2,100 tokens; the cache only removes the encoder), so 9 runs at 140
  tokens/frame would take ~70-145 h. A shared rate is a fair comparison (identical treatment;
  AdamW normalizes gradient scale, so C's softer gradients are not short-changed). Instead of a
  search, every run logs learning curves to judge whether the rate suits each target: training
  diagnostics every 10 steps (loss / CE / KL, batch accuracy, teacher agreement, confidence,
  gradient norm, update ratio, LoRA B norm) and a fixed 100-video dev sample (videos d3-mini
  scored) every 50 steps (accuracy, NLL, ECE, teacher agreement, drift). If the curves show the
  rate is wrong for a target, or two targets end within dev's resolution, a learning-rate search
  on the contenders follows as its own step.
- **Kept 140 tokens/frame** (not the 70-token reduced setting step 8 allows): with 3 runs it is
  affordable, and the winner needs no retraining at 140.
- Resumable (adapter + optimizer state every 25 steps in `checkpoints/step7/<tag>/` on the drive).
  Dev rows go to `results/evals/dev/s7_<variant>_lr<lr>.jsonl` in the evaluate.py format.
- **Teacher columns cover 712 dev questions**, not 2,003: step 4 scored only a 200-video sample.
  Agreement with the teacher and accuracy where the teacher is wrong are reported on those.
- **Post-training checks:** A-OKVQA with the adapter forced on (`aokvqa run --adapter`). The
  badge set is not wired in: it is a text task, and the adapter is never active for text requests.
- Timing (40 train videos, alone on the GPU, other apps open): 238 s per 8-video step, i.e.
  ~16 h per run; profiling put a video at ~14 s forward + backward when memory is not swapping
  (gradient checkpointing is required: without it a video peaks at ~26 GB on an 18 GB Mac).
  Per run add ~5 dev-sample evaluations and one full dev pass (forward only).

### Step 7: results

**Decision: target A (answer only, α = 1), lr 1e-4, no learning-rate search.** Adapter:
`checkpoints/step7/s7_A_lr1e-04/`. Full report: `results/step7/summary.md` on the drive.

Perception Test dev (2,003 questions, cached features); "net" = gained minus lost against
zero-shot video@140, paired (McNemar) p; teacher columns on the 712 questions d3-mini scored:

| | Accuracy | Net vs zero-shot (p) | NLL | ECE | conf >= 0.8: cov / acc | Teacher agreement | Acc where teacher wrong |
|---|---|---|---|---|---|---|---|
| zero-shot video@140 | 54.7% +- 2.2 | - | 1.582 | 0.302 | 67% / 62.3% | 60.5% | 28.5% |
| **A (α=1)** | **69.1% +- 2.0** | **+289 (6.5e-30)** | **0.626** | **0.018** | 35% / 95.1% | 64.6% | 48.3% |
| B (α=0.5) | 69.3% +- 2.0 | +293 (1.3e-31) | 0.660 | 0.046 | 29% / 96.7% | 70.5% | 36.7% |
| C (α=0) | 64.9% +- 2.1 | +205 (7.4e-18) | 0.764 | 0.034 | 23% / 96.1% | 76.1% | 17.4% |

- **A and B tie** (B vs A: 142 gained, 138 lost, p = 0.81). A wins the tie: better NLL and ECE,
  and it needs no teacher labels (the cheaper setting wins ties, step 1). A and B both beat C
  (p <= 1e-4): teacher-only training copies the teacher's mistakes (17% right where d3-mini is
  wrong, vs 48% for A), and the label-trained student already matches the teacher's accuracy.
- **By area (A):** abstraction 45.1 -> 63.1%, physics 46.8 -> 62.6%, semantics 74.5 -> 85.7%,
  memory 51.2 -> 51.2% (n = 80, too small to read).
- **Curves:** A's dev sample (100 videos) peaked at step 200 (68.1%) and ended at 67.2%, flat from
  step 150 as the learning rate decayed; train batch accuracy ended at 72%, near dev, so one pass
  did not overfit. No learning-rate search (decided 2026-10-28).
- **A-OKVQA (adapter forced on, 500 questions):** 77.2% -> 79.4% (28 gained, 17 lost, p = 0.14),
  NLL 0.998 -> 0.622, ECE 0.031: no forgetting on image QA.
- **Not run: NExT-QA** with the adapter (`evaluate.py` cannot load an adapter yet); step 8 adds it.
- **Training time:** 6.1-9.3 h per run (not the ~16 h estimated), plus ~0.5 h for the dev pass when
  memory is free. Swapping (other apps open) slowed dev scoring ~5x.
- Fixed: `aokvqa run --adapter` re-entered one `adapter_scope` context for every question and
  crashed on the second.

## Step 8: scale the winner (done, 2026-11-02)

`video_train/scale.py` (`check`, `run`, `report`); tests in `tests/test_scale.py`.

- **Two runs, each one change against s7_A** (chosen 2026-10-28; everything else as step 7: all
  7,392 train questions, step 6 cache, seed 0, 8 videos per step, AdamW, 5% warmup then cosine
  decay over the whole run, lr 1e-4, answer-only loss):
  - `s8_A_r32`: 1 pass at LoRA rank 32 / alpha 64 (was 8 / 16). alpha / rank stays 2, so only
    capacity changes. ~7-10 h.
  - `s8_A_ep2`: 2 passes (490 steps) at rank 8. A fresh run with the schedule over both passes,
    not a continuation of s7_A (its rate has decayed to zero). ~12-19 h.
  Testing two changes as separate runs, each against s7_A, keeps one change per comparison;
  combining them (2 passes at rank 32) only follows if both win.
- **Why scale at all:** s7_A's dev sample was flat over the last ~100 steps while the rate decayed,
  and train batch accuracy (72%) sat near dev (69%): no overfitting, so more training or more
  capacity may still help. More data was not chosen: it reverses step 1 and would make NExT-QA a
  less clean out-of-domain check.
- **Decision rule:** a run replaces s7_A only if it beats s7_A on dev, paired, gaining more than
  it loses with McNemar p < 0.05 (step 1's small-effects rule: a tie keeps the cheaper s7_A). If
  both pass, the more accurate wins.
- **NExT-QA with the adapter** (1,003 questions, out of domain; reported, not a gate) for s7_A
  and both runs. `evaluate.py run --adapter <dir>` installs the LoRA per request, as in serving.
  `scale check` first scores 20 dev videos that way (fresh video decode) and requires >= 98%
  answer agreement with s7_A's dev rows (cached features, LoRA attached as in training).
  **Passed 2026-10-28:** 68 questions, 100% agreement, max probability difference 0.
- `distill.py` gained `--step` (checkpoint folder and history step; default step7, so step 7
  paths are unchanged). Checkpoints: `checkpoints/step8/<tag>/`; report `results/step8/summary.md`.
- Order in `scale run`: check, NExT-QA for s7_A, then r32 (train, dev, NExT-QA), then ep2. Each
  stage is resumable; rerun the same command after a stop. Close other apps: swapping slowed
  step 7's dev scoring ~5x.
- Keep step 6's feature cache until step 8 is reported.

### Step 8: results

Full report: `results/step8/summary.md` on the drive.

| | dev acc | vs s7_A on dev (gained / lost, p) | NExT-QA acc | vs s7_A on NExT-QA | dev ECE | conf ≥ 0.8: cov / acc |
|---|---|---|---|---|---|---|
| zero-shot | 54.7% | - | 68.9% | - | 0.302 | 67% / 62.3% |
| s7_A | 69.1% | - | 70.4% | - | 0.018 | 35% / 95.1% |
| s8_A_r32 | 64.8% | 130 / 216 (3.8e-06) | 62.2% | -82 (3.2e-08) | 0.022 | 21% / 95.3% |
| **s8_A_ep2** | **73.5%** | **217 / 128 (1.7e-06)** | 70.1% | -3 (0.79) | 0.057 | 55% / 90.0% |

- **s8_A_ep2 replaces s7_A** under the decision rule. It gains in every area, most in memory
  (51% → 65%, n = 80) and physics (63% → 69%). Its dev curve was still at its peak at the last
  step (68.6% at step 490), so a longer run may still help.
- **Rank 32 lost** on both sets: it learned more slowly at the same rate (dev sample 61.6% at the
  end vs 67.2%), and fell below zero-shot on NExT-QA. The combined run (rank 32, 2 passes) is
  dropped, since it only followed if both won.
- **Out of domain is flat:** NExT-QA stays within noise of s7_A and zero-shot, so the dev gain
  is Perception Test specific.
- **Calibration got worse:** ECE 0.018 → 0.057, and answers at confidence ≥ 0.8 are right 90% of
  the time, down from 95%. Fix this with temperature scaling on dev before anything uses the
  confidence as a threshold.

## Step 9: probe other vision encoders (done, inconclusive, 2026-11-03)

`video_train/probe.py` (`fetch`, `run`, or one stage: `text`, `pool`, `encode`, `train`, `report`);
tests in `tests/test_probe.py`.

- **Question:** does another frozen encoder see something in Perception Test videos that Gemma
  4's own encoder misses? A small probe on frozen features answers this for every candidate in a
  few hours; a step 10 run per encoder would take days.
- **Arms** (same probe and text features in each; only the video features change): `text` (no
  video), `gemma` (step 6 cache, mean of the 140 tokens per frame), `siglip2` (SigLIP 2 base),
  `dinov2` (DINOv2 base, self-supervised), `vjepa2` (V-JEPA 2 ViT-L, a video encoder that sees
  motion), and `gemma+<x>` for each candidate. Candidates kept to sizes that could run next to
  Gemma on the Mac.
- **Inputs:** text = base Gemma (no adapter) reading the question as scored, without the video;
  last-token hidden state plus mean hidden state per option. Video = the step 6 cache's 16
  uniform frames, one vector per frame (V-JEPA 2: per two-frame slot), squashed to each
  encoder's square input with no crop. One vector per frame under-measures fine spatial detail,
  equally for every arm.
- **Probe, fixed before any result:** option text attends over the video vectors (one temporal
  self-attention layer first), MLP score per option, softmax. Width 256, dropout 0.2, AdamW
  1e-3, wd 0.05, batch 64, ≤ 40 epochs, early stop on 10% of train videos held out. Three seeds;
  dev rows are seed-mean probabilities; dev is scored once per arm.
- **Decision:** (1) `gemma` must beat `text` (paired, p < 0.05), otherwise inconclusive. (2) x
  passes if `gemma+x` beats `gemma` with more gained than lost and p < 0.05 / 3 (three candidates,
  so each needs stronger evidence). Best passing x → step 10 builds on it; none → skip step 10
  and go to step 11. `x` alone vs `gemma` and the per-area table are reported, not gated.
- **Not used:** NExT-QA (the probe is a diagnostic, not a model; step 10's model gets that
  check) and the test split.
- **Cost (measured in the smoke test):** text features 0.30 s/question (~47 min), HDD decode
  1.5 s/video, SigLIP 2 0.35 s/video; probe ≤ 5.5 s/epoch for the largest arm. Total ≈ 4-5 h.
  `fetch` downloads DINOv2 base and V-JEPA 2 ViT-L to the drive (~1.6 GB, once).
- Features: `~/.cache/openjev-video/probe/<source>/` (small, rebuildable). Outputs:
  `results/step9/probe/`, report `results/step9/summary.md`. Needs step 6's cache still on disk.

### Step 9: results

| arm | dev acc | net vs text (p) | net vs gemma (p) | semantics | physics | memory |
|---|---|---|---|---|---|---|
| text | 55.1% | - | +9 (0.58) | 59.4% | 55.8% | 36.2% |
| gemma | 54.7% | -9 (0.58) | - | 59.9% | 54.8% | 38.8% |
| siglip2 | 56.5% | +28 (0.11) | +37 (0.069) | 67.1% | 54.6% | 37.5% |
| dinov2 | 57.5% | +47 (0.0091) | +56 (0.0014) | 67.7% | 54.5% | 38.8% |
| vjepa2 | 54.2% | -19 (0.29) | -10 (0.56) | 58.7% | 55.5% | 32.5% |
| gemma+siglip2 | 56.6% | +30 (0.034) | +39 (0.021) | 61.0% | 58.0% | 38.8% |
| gemma+dinov2 | 57.0% | +38 (0.026) | +47 (0.0064) | 66.7% | 56.1% | 35.0% |
| gemma+vjepa2 | 54.4% | -14 (0.39) | -5 (0.77) | 58.7% | 56.2% | 33.8% |

Base Gemma without video: 39.8%. Encode cost per video (16 frames): SigLIP 2 0.25 s, DINOv2
0.29 s, V-JEPA 2 1.65 s.

- **Verdict: inconclusive** under the rule set in advance: `gemma` does not beat `text`
  (-9, p = 0.58), so the probe cannot read Gemma's own features and the comparison against them
  means nothing.
- **Why, most likely: the probe, not Gemma's encoder.** Steps 7-8 took the real model from 54.7%
  to 73.5% using exactly these features, so they hold a lot of usable information. The `gemma`
  arm averages 140 projector tokens per frame; those tokens are made to be read one at a time by
  the language model, not averaged, while SigLIP 2 and DINOv2 have outputs built to summarise an
  image. The arms were therefore not equal, contrary to the design's claim. More broadly, the
  probe gets at most +2.4 points from any video input over text alone (55.1%): it is too weak to
  measure what video adds.
- **The one signal:** DINOv2 and SigLIP 2 summaries help on semantics (what is in the scene,
  59% → 67-68%), the area where s8_A_ep2 is already strongest (87.6%). Nothing helped on physics
  or memory, the weak areas (68.6% and 65.0%), but the probe is too weak to rule that out.
- V-JEPA 2, the only video encoder, added nothing in this probe.

### Step 9 v2: the probe redone with spatial detail (design fixed 2026-11-03, before any v2 result; done, fail, 2026-11-04)

Decided 2026-11-03: rerun step 9 with a fairer probe, record the result, then decide steps 10-12.
`video_train/probe_grid.py`; v1 stays as recorded above.

- **What changes from v1, and why:** v1 squeezed each frame into one vector, which for Gemma
  meant averaging 120-136 tokens that each describe a different region. v2 keeps a **4×4 grid of
  regions per frame for every encoder**, made by the same operation for all: average the
  encoder's own patch tokens over each cell. Gemma: its token grid (8×15 for 16:9 video, 10×13
  for 4:3; tokens are stored row by row, each covering 3×3 patches) from the step 6 cache.
  SigLIP 2 14×14 patches, DINOv2 16×16 patches (CLS dropped), V-JEPA 2 16×16 per two-frame slot.
  So a video is 256 vectors (16 frames × 16 cells; V-JEPA 2: 8 slots × 16 = 128). No pooled or
  CLS summaries: every arm gets patch averages only.
- **Probe v2** (sizes chosen before any v2 run): video vectors get a learned cell-position
  embedding (16 cells) and a time embedding; the question vector joins them and two transformer
  layers (width 256, 4 heads) mix them, so video regions are read in light of the question. Each
  option then attends over the result and an MLP scores it. Same input text features as v1 (base
  Gemma, no adapter). Text-only arm: the same network with the question vector alone. Dropout
  0.2, AdamW lr 3e-4 (transformers are less stable than v1's single attention at 1e-3), weight
  decay 0.05, batch 64, ≤ 40 epochs, early stop (patience 5) on the same 10% of train videos.
  Three seeds; dev rows are seed-mean probabilities.
- **New control (reported, not gated):** `gemma` scored with each dev question given another dev
  video's features. If the probe uses the video, this falls to `text`'s level; if not, any gap
  between `gemma` and `text` is not about the video.
- **Decision rule, unchanged from v1:** (1) `gemma` must beat `text` (paired, more gained than
  lost, p < 0.05), else inconclusive. (2) x passes if `gemma+x` beats `gemma` with p < 0.05 / 3.
  Best passing x → step 10 builds on it; none → no evidence for step 10.
- **What each outcome means for the next decision:**
  - pass: step 10 (second encoder) is worth designing around x.
  - fail (`gemma` readable, nothing adds): Gemma's own features already hold what these encoders
    see, at this level of detail.
  - inconclusive again: two probe designs could not read Gemma's features. No v3: the probe
    route is closed and step 10 is decided without it.
- NExT-QA and the test split are not used, as in v1.
- **Cost (smoke test, 2026-11-03):** one pass over the videos (decode 1.5 s + SigLIP 2 0.3 s +
  DINOv2 0.3 s + V-JEPA 2 1.8 s per video) ≈ 2.7 h; probe training up to 82 s per epoch for the
  largest arm on MPS, so ~5 h if early stopping lands around 15 epochs (worst case ~13 h).
  Reuses v1's text features. Run: `caffeinate -i .venv/bin/python -m video_train.probe_grid run`.

### Step 9 v2: results (done, fail, 2026-11-04)

Full report: `results/step9/v2/summary.md` on the drive.

| arm | dev acc | net vs text (p) | net vs gemma (p) | semantics | physics | memory |
|---|---|---|---|---|---|---|
| text | 54.6% | - | -117 (5.3e-10) | 59.6% | 54.9% | 33.8% |
| gemma | 60.5% | +117 (5.3e-10) | - | 69.4% | 61.1% | 42.5% |
| gemma_shuffled (control) | 53.3% | -26 (0.17) | -143 (4.1e-16) | 55.3% | 54.0% | 36.2% |
| siglip2 | 57.7% | +61 (0.00096) | -56 (0.00093) | 69.1% | 57.5% | 33.8% |
| dinov2 | 59.5% | +98 (2.1e-06) | -19 (0.32) | 74.0% | 56.8% | 40.0% |
| vjepa2 | 60.3% | +113 (8.4e-10) | -4 (0.81) | 65.0% | 65.7% | 37.5% |
| gemma+siglip2 | 60.3% | +114 (2.3e-09) | -3 (0.86) | 72.5% | 59.5% | 35.0% |
| gemma+dinov2 | 59.5% | +97 (2e-06) | -20 (0.24) | 68.4% | 61.3% | 37.5% |
| gemma+vjepa2 | 62.2% | +152 (3.8e-13) | +35 (0.051) | 72.0% | 64.8% | 40.0% |

- **Verdict: fail** under the rule fixed before the run. (1) holds: `gemma` beats `text`
  (+117, p = 5e-10), and the shuffled control falls to `text`'s level, so v2 reads the video.
  (2) fails: no `gemma+x` beats `gemma` at p < 0.017; the closest is `gemma+vjepa2` (+35,
  p = 0.051). Gemma's own features already hold what these encoders see, at this level of
  detail. **Step 10 is skipped; next is step 11.**
- Hints, not gated: V-JEPA 2 alone is best on physics (65.7% vs 61.1%), DINOv2 alone on
  semantics (74.0% vs 69.4%). Abstraction and memory barely move for any arm.
- The probe's best (62.2%) is far below s8_A_ep2's 73.5%: it measures what features make
  available, not what a model built on them would score.
- **V-JEPA 2 input size, fixed 2026-11-03:** its video processor ignores `size` given at load
  and keeps `shortest_edge: 292` with the aspect ratio. v1's V-JEPA 2 features were therefore
  made at ~292×389 / 292×519, not the square 256×256 the v1 design states; the v1 table stands
  as recorded. `FrameEncoder` now passes `size` at call time, so v2's V-JEPA 2 is 256×256 like
  the design says.

## Step 11: decision head and calibration (design fixed 2026-11-04, before any result; done 2026-11-05)

`video_train/decision_head.py` (`run`, or one stage: `calibrate`, `folds`, `cache-nextqa`,
`features`, `train`, `report`); tests in `tests/test_decision_head.py`. Step 10 is skipped
(step 9 v2 failed). Two parts, each judged on its own against s8_A_ep2.

**Part A: calibration.** One temperature T on s8_A_ep2's letter log-probabilities
(probabilities = softmax(log p / T)). It cannot change an answer, only the confidence.
- Fit: minimise dev NLL over T. Reported honestly with 2-fold cross-fitting by video on dev
  (fit on one half, score the other); the shipped T is fit on all of dev.
- Kept if the cross-fitted dev ECE and NLL are both lower than raw. NExT-QA is scored with the
  dev T, as an out-of-domain check (informs, does not gate).

**Part B: decision head.** A small network on s8_A_ep2's frozen hidden states that scores each
option, instead of reading the letter logits alone.
- **Problem it must avoid:** s8_A_ep2 trained 2 passes on every train question, so its train
  features are memorised; a head trained on them learns from answers that are too easy.
- **Fix: cross-fitting.** Two half adapters, `s11_half0` and `s11_half1`, with s8_A_ep2's recipe
  (target A, lr 1e-4, 2 passes, rank 8 / alpha 16, seed 0), each on half the train videos (split
  by video, seeded). Head training features for a question come from the half adapter that never
  saw it; dev and NExT-QA features come from s8_A_ep2, the model the head would ship with. The
  head therefore trains on slightly weaker features than it is scored on; that mismatch is the
  known cost of this setup.
- **Features**, per question: the final hidden states of every question token (the text after
  the video), each option's token span, and the letter logits. Dev letter logits must agree with
  s8_A_ep2's dev rows (≥ 99% same answer) or the stage stops.
- **Head** (sizes fixed now): question tokens projected to width 256; each option (mean of its
  span, plus the last token) attends over them (4 heads); an MLP scores the option.
  utility = s · letter logit + MLP score, s learned (starts at 1) and the MLP's last layer
  starting at zero, so the untrained head *is* letter scoring and training can only move away
  from it if that helps the held-out loss. Dropout 0.2, AdamW lr 3e-4, wd 0.05, batch 64,
  ≤ 40 epochs, early stop (patience 5) on 10% of train videos held out. Three seeds; dev rows are
  seed-mean probabilities. Its own temperature is fit the same way as part A, for calibration.
- **Decision (step 1's rule):** the head replaces letter scoring only if it beats s8_A_ep2 on dev,
  paired, more gained than lost, McNemar p < 0.05. Calibration and NExT-QA are reported next to
  it; they inform, they do not gate.
- Not in this step: the `openjev decision` API wrapper (later, on the winner), the test split.
- **Cost (estimate):** two half adapters ~2.3 h each (no dev curves); NExT-QA vision cache
  (115 videos) ~5 min; features ~1 s per question, ~2.9 h for train + dev + NExT-QA, ~3 GB in
  `<LOCAL_CACHE>/step11/`; head ≤ 1 h. Total ≈ 8 h. Run:
  `caffeinate -i .venv/bin/python -m video_train.decision_head run`.

### Step 11: results (done, 2026-11-05)

Full report: `results/step11/summary.md` on the drive.

| | dev acc | dev NLL | dev ECE | dev conf ≥ 0.8: cov / acc | vs s8_A_ep2 on dev | NExT-QA acc | NExT-QA ECE | vs s8_A_ep2 on NExT-QA |
|---|---|---|---|---|---|---|---|---|
| s8_A_ep2 | 73.5% | 0.590 | 0.057 | 55% / 90.0% | - | 70.1% | 0.073 | - |
| **s8_A_ep2 + T (1.523)** | 73.5% | 0.565 | 0.020 | 44% / 94.8% | - | 70.1% | 0.042 | - |
| head | 72.8% | 0.581 | 0.048 | 52% / 91.5% | 42 / 57 (0.13) | 67.4% | 0.064 | 16 / 43 (0.00044) |
| head + T | 72.8% | 0.566 | 0.021 | 43% / 95.0% | 42 / 57 (0.13) | 67.4% | 0.019 | 16 / 43 (0.00044) |

- **Calibration kept:** T = 1.523 (dev folds 1.60 and 1.45). Cross-fitted dev ECE 0.057 → 0.020,
  NLL 0.590 → 0.565; NExT-QA, with the dev T unchanged, ECE 0.073 → 0.042. This undoes step 8's
  calibration loss.
- **Head not kept:** dev -15 net (p = 0.13), NExT-QA -27 net (p = 0.0004). It did learn on its
  own training distribution (held-out NLL 0.696 → ~0.66 on all 3 seeds), but those features come
  from the half adapters (68.1% on the train questions they never saw), and the corrections did
  not carry over to s8_A_ep2's features; the expected mismatch cost was larger than the gain.
- **Model for step 12: s8_A_ep2 with T = 1.523.**

## Latency

Per-question latency = model compute only (vision encoder + prompt + questions, divided over
the video's questions). Model loading, copying from the drive, and video decoding from the HDD
are excluded; decoding is reported in its own column.
