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
3. drives a character in a Python open-world game from its first-person view on instructions
   like "cross the bridge". This game is the final demo.

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
- **Game in Python**, used for both training data and the demo, so train and demo frames match.
  The model decides about once a second (action or target point); the game's pathfinding walks.
- **Research-only datasets are allowed** (the model is not sold). Record which were used in case
  weights are ever published.
- **Trained models only (2026-11-16):** no runs on the downloaded model without an adapter, now or
  later; `--adapter` is required in `vdm_eval` and `vdm_cams`, and training no longer scores the dev
  sets at step 0. Code removed then (results below stay as history): `vdm_stage0`, the first three
  game tries (`vdm_game drive`/`gen`, `vdm_rules`, `vdm_pano`, `Renderer.panorama`) and the Gemma
  reference in `vdm_eval`. "Zero-shot" below means the game is never trained on.

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
- Not now: egocentric data (Ego4D needs an agreement) and image grounding (the game supplies
  pointing labels in Stage B). Revisit if stage 1 shows a gap.
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

### Stage 4: the game
Python first-person open world (engine to decide, e.g. Ursina / Panda3D): terrain, river,
bridges, landmarks, random maps, expert pathfinding for labels, headless frame capture.
Can run in parallel with stages 2-3.

**Playable first version (2026-11-10)**, `openworld/` (`python -m openworld.play [--seed N]`):
`world.py` builds the map with numpy only (160 m square, hills, a north-running river in a
gorge, 2-3 bridges, 5 named landmarks: red house, stone tower, windmill, big tree, blue tent;
~140 trees) and answers walkability, ground height and nearest landmark, so pathfinding and
labels can use it without the engine. `play.py` draws it in plain Panda3D (extra `game`).
Ursina was tried first and dropped: its GLSL 1.30/1.40 shaders and UI do not render on macOS.
No shaders: sunlight is baked into vertex colours, so headless capture needs no GPU features.

### Stage 5: game data and game baseline
Generate instructions, expert trajectories and labels; held-out maps and instructions for
evaluation. Measure the Stage A model on the game zero-shot.

**Stage A full eval skipped (2026-11-14):** the dev curve (15 videos per set) was flat from
step ~1050; nothing is published yet, so we went straight to the game. Run
`vdm_eval run --set all --tag stageA_ep1` before any result is reported.

**Decided (2026-11-14, later the same day): the game is a zero-shot test, no Stage B.** The
claim is that the general model handles all kinds of video, games like this one included, so
it never trains on the game (like Video-MME and TempCompass). An earlier version of this stage
generated game training data and a Stage B mix; it was dropped and its data deleted.

Built in `openworld/expert.py`, `openworld/render.py`, `video_train/vdm_game.py`:
- **Fair tasks:** go to a landmark (50%), cross the river (25%, incl. "Cross the bridge."),
  get onto the nearest bridge (25%); 4 wordings each. The target can always be found from the
  start by turning on the spot (it may start behind the player; terrain occlusion checked).
- **Each second** the model sees the last 8 frames at 2 fps (4 s, 640×400 offscreen Panda3D,
  192 tokens per 2 frames; the first frame repeated before the start) and is asked:
  1. the task's arrival question ("Are you standing right next to the windmill now?",
     "Have you crossed the river to the other side now?", "Are you standing on a bridge
     now?"): yes (P > 0.5) ends the episode;
  2. otherwise the move, as multiple choice: walk forward 2.5 m, turn left 45°, turn right 45°,
     turn around (argmax of the option-letter probabilities).
- **Score, closed loop** (`vdm_game drive`, 30 dev maps): success = says it arrived inside the
  goal; also reached the goal at some point, stopped early, timed out (2× the expert's steps
  + 20), SPL against the expert's path, move agreement with the expert.
- **Expert** (reference only; it sees the map): distance field to the goal (Dijkstra, 1 m
  grid, steps checked every 0.25 m), per state the best of the 8 headings after 1 s of real
  walking. Reaches the goal on 270/270 test maps with or without 20% random moves.
- **Per-question rows** (`vdm_eval run --set game_dev`): expert rollouts with 20% random
  moves, 2 turns on the spot at the goal; per state the arrival question, the move (when not
  arrived), "Can you see the X right now?" (half the states, ~35% yes) and "Are you standing
  on a bridge right now?" (on a bridge + 10% of the others). Dev 30 episodes (1,311 states,
  3,489 questions), test 30 (1,390 states; final run only); 1.3 GB in `vdm/game/`. The move
  label needs the map, so move accuracy here is a rough guide.
- Railings made solid (walkers keep 0.4 m inside), otherwise the camera ended up inside a post.
- **Smoke (stage A model, 2 dev episodes):** "Head for the stone tower": turned to it and
  walked straight at it (moves 90% as the expert), said arrived 3 steps early. "Get on the
  closest bridge": kept alternating turn around / turn left-right, timed out.

**First zero-shot drive (2026-11-15), `vdm_game drive`, 30 dev episodes:** Stage A ep1 and the
base model both 1/30 (3%), SPL 0.02 / 0.03, moves agree with the expert 32% / 31% (25% by
chance). Stage A stops early more (63% vs 43%). Why, from the traces:
- **The move answer ignores the picture.** Neither model ever chose "turn left" with the target
  on the left (Stage A, 392 such steps: 286 forward, 22 right, 84 around, 0 left). Base gives
  "turn right" 56-60% whatever the right move is. Not the letter B (B is picked 21-32% on the
  other benchmarks) and not the game (a left turn moves the picture right, checked by
  rendering). Dividing out each option's average probability: agreement 25% -> 27% (base),
  21% -> 22% (Stage A), so there is no signal to recover.
- **The arrival answer ignores the picture too:** Stage A P(arrived) 0.29 at the goal vs
  0.14-0.19 elsewhere; it stopped at 6% of the steps at the goal.
- **Stuck walking into things:** 16% (Stage A) / 8% (base) of the moves were "forward" into a
  bank or wall without moving; a few long spells explain many timeouts.
- **Unfair steps:** 9 of the 17 landmark tasks need a river crossing; seeing the target is then
  not enough, the bridge may be out of sight (seeds 100033, 100027: the expert turns back
  to a bridge for ~100 steps with the target in view).

**Decided (2026-11-15): simple questions and a fixed rule** (`video_train/vdm_rules.py`, still
zero-shot: no game training). The model only answers what it can see, one small question at a
time; a fixed, readable rule turns the answers into the move. Decisions:
- **Granular questions, mostly yes/no**, each about the current frame only (shown as a 2-frame
  clip of the same picture): is the target in the picture; where is it (left side / middle /
  right side, three letters); are you right next to it; are you on a bridge; are you at the end
  of a bridge; where is the far end of the bridge; and, after a forward move, "did the camera
  move forward between these two pictures?" (the two frames around the move). Why: these are
  the perception questions general video models answer well, and each one can be checked on
  its own against the map (every answer is stored with the truth).
- **Bridge questions point at the end of the bridge, not the bridge.** Aiming at the middle of
  a bridge walked the agent into the bank beside it or across the deck sideways.
- **"Middle" = the centre third of the picture (±13°)**, what the words mean in a picture. ±25°
  (just over half a turn) does better with the oracle (no zig-zag), but the model will answer
  by the picture's thirds, so the map truth and the oracle use thirds too. The zig-zag is
  handled in the rule: right after turning one way, a target on the other side means "walk
  forward".
- **Fixed rule** (in order): stuck -> turn left, then one step forward; cross task off a bridge
  it was on -> one more step, then stop (checked before "at the end", since the far bank's end
  is an end too); on a bridge or at its end -> steer to the far end, bridge task stops after 2
  forward steps on the deck; landmark seen and near -> stop; seen -> steer (middle forward,
  else turn that way); not seen -> turn left to search, after a full circle walk forward once.
- **Fair episodes only:** landmark tasks across the river are dropped (the bridge may never be
  in sight); cross and bridge tasks already have a bridge in sight by turning from the start
  (`expert.findable`). The first 30 fair dev episodes; the first 21 are the ones the first drive
  played, so the two can be compared on those.
- **`--oracle`**: the same loop with the map's answers instead of the model (seconds, no model):
  the ceiling of the rule, run before any model time. v2 (aim at the bridge's middle) 17/30;
  v3 (bridge ends, the at-the-end question) 19/30 with a ±25° middle, 15/30 with thirds; v4
  (cross stop before at-the-end, thirds) **19/30: landmark 10/10, cross 4/8, bridge 5/12, SPL
  0.43.** Oracle runs: tags `rules_oracle*` in `vdm/results/stage5/` (v4: `rules_oracle_v4`).
- **Game bug found and fixed (2026-11-15): railings were only solid over the water.** They are
  drawn along the whole deck, but where the deck crosses the bank the walker could step through
  them sideways (and onto the deck from the side). With 45° turns a heading can be up to 22.5°
  off the deck, so the agent drifted off. Now `World.walkable` and the expert's `walkable_grid`
  block the railing band along the whole deck (plus 0.4 m outside it, as inside): a bridge is
  entered only at its ends. Side effect fixed with it: the expert's distance lookup took a
  neighbour cell's distance through a railing thinner than a 1 m cell (`expert.remaining` now
  takes `w` and uses only neighbours it can walk straight to); without that the expert failed
  4/30 dev episodes. Expert after both: every dev and test episode reached, and 270/270 random maps with or without 20% random moves (rechecked). The dev and test
  episode lists are unchanged (same seeds, tasks and starts); the expert's step counts changed
  on 18 of the 30 dev episodes. **The stored `game_dev` / `game_test` rows were made under the
  old rules:** regenerate (`vdm_game gen --force`) before using them again; the step 1
  per-question eval of 2026-11-15 scored the old rows.
- **Oracle with solid railings (rule v4): 18/30, landmark 10/10, cross 5/8 (was 4), bridge 3/12
  (was 5).** Bridges are now entered only through the 2.2 m opening at an end; arriving at an
  angle (45° turns), the agent hits the end post and the stuck rule does not get it in. Next for
  the rule: line up before the end (approach from further out along the bridge), checked with
  `--oracle` first. Tags `rules_oracle_v4_rails2` (and `_rails`, before the expert fix).

**Simple-question runs (2026-11-15, rule v4, 30 fair dev episodes):** base 0/30, Stage A 1/30.
Landmark perception was good (in view -> "yes" 80-81%, left/middle/right right 86-88%), but
"did the camera move forward?" said no 70% (base) of the time it had moved (the stuck rule then
turned the walker away), "right next to it?" ended 6 (base) / 4 (Stage A) of the 10 landmark
episodes early, and bridges were hard to see (in view -> "yes" 49-53%; the near end located
39%). Two problems were ours: the map truth's 4° edge margin called bridges at the picture's
edge not visible (the model said yes, rightly), and the rule could spin forever (its
full-circle counter reset whenever the model said it saw something).

**Decided (2026-11-15): landmark tasks from a 360° video** (`video_train/vdm_pano.py`), still
zero-shot. Decisions:
- **Landmark tasks only for now** (on the start's side of the river), to get one task right first.
- **A camera rig, not turning to look:** every half second `Renderer.panorama` renders the full
  360° around the walker from 8 views 45° apart and unrolls it into one 2528x400 strip. Straight
  ahead is the middle; going left from the middle turns to your left, going right to your right;
  the cut is behind you, so both edges are directly behind. Why behind: the direction you walk in
  stays whole and in the middle, and left and right in the picture match left and right as in a
  normal view (the cut would split whatever is straight ahead if it were in front).
- **The input is a video, not one stitched still:** the last 4 s, 8 panorama frames at 2 fps,
  through Qwen's video path (grid t = 4: Qwen fuses each 2 frames 0.5 s apart; that is built
  into Qwen3-VL, `temporal_patch_size` 2, not ours). Each frame is one complete 360° view, so
  the fusing only joins the same directions half a second apart. 768 vision tokens per pair
  (frames resized to 2208x352, about the angular sharpness of the 640x400 frames at 192):
  3,036 vision tokens per step.
- **The prompt says what the frames are**, then asks; one video, encoded once, then one short
  pass per question: "Can you see the X anywhere around you now?" (yes/no); if yes, "Your task:
  go to the X. Where is the X from you now?" with 8 options in order around you (straight
  ahead, ahead on your left, to your left, behind you on your left, directly behind you, behind
  you on your right, to your right, ahead on your right); if straight ahead, "Are you right next
  to the X now?" (yes -> stop).
- **Rule:** straight ahead -> forward; the 3 left directions -> turn left 45°; behind -> turn
  around; the 3 right directions -> turn right 45°; not seen -> forward. The 8 directions are
  45° wide like one turn, so turning puts "ahead on your left" straight ahead: no zig-zag. No
  "did I move" question (it failed) and no stuck rule.
- **Truth:** seen = in sight if you faced it (terrain, 120 m; trees ignored); direction = the
  landmark's angle rounded to the nearest 45°.
- **Oracle (map answers): 28/30, SPL 0.72.** Smoke, base model, 2 episodes: both stopped early
  (a false "right next to it"), "where" 59% (straight ahead 16/18; off to a side it mostly
  answers "to your left" / "to your right", never behind), **11 s per step** (3,036 vision
  tokens): a full 30-episode run would take ~3-4 h; `--tokens 384` halves the tokens.
- **`pano_stageA_ep1` (30 dev episodes):** see 73% (yes 70%), near 97% (yes 60%), where 23% of
  956 (62% counting the next direction as right; 46% of answers "straight ahead", the right side
  almost never). The model does not read directions off the unrolled strip.

**Decided (2026-11-16): 4 head cameras, 4 videos in one prompt** (`video_train/vdm_cams.py`),
still zero-shot. Decisions:
- **Input:** `Renderer.cameras`: front, left, back, right, each 90° wide (no gaps), each the last
  4 s at 2 fps as its own video, labelled in the prompt; 192 tokens per 2 frames of one camera
  (2,933 prompt tokens). Normal camera views, not a layout the model has to learn from text.
- **Front camera last in the prompt:** probe (base model, one scene; to recheck with the adapter)
  found the first video under-seen: the same front view P(yes) 0.2-0.5 as the first video, 1.0 as
  the third or last.
- **Questions:** 4 yes/no "Can you see the X in the {camera} camera now?" (all 4 asked; the
  highest above 0.5, or none). If front: the X's bounding box in the front camera's last frame
  (Qwen's grounding JSON, 0-1000), its centre -> the exact angle -> left / middle (±22.5°) /
  right; the worded 3-option question is always asked too, logged, and decides only when the box
  cannot be read. If front middle: "Are you right next to the X now?".
- **Rule:** none -> forward; left / right camera -> turn 45° that way; back -> turn around; front
  left / right -> turn 45°; front middle -> forward or stop. **Oracle: 28/30, SPL 0.72** (same as
  the strip).
- **Probe, base model, 4 front views of one tower:** camera right 4/4, side by box 4/4 and by
  words 4/4; box angle off by 4-13° (true 0/30/-15/-35, box 4/34/-2/-25). 9 s per step, 11-17 s
  when the box is asked.
- **`cams_stageA_ep1` (cams-v1, stopped after 22 episodes, 2026-11-16): 4 successes.** Three causes:
  - **Early stops (12 of 22), all from "Are you right next to it?":** P(yes) ~0.5 whatever the
    truth (12 false stops at 0.505-0.69; the 6 true ones 0.36-0.58), asked every front-middle step.
  - **Left camera blind:** target seen in 9 of 193 steps where it was in the left camera (front
    78%, back 50%, right 47%). Left is first in the prompt: the probe's first-video effect moved
    from front to left. The walker explores away; seed 100069 had it on the left for 121 of 122 steps.
  - **Deadlock (seed 100043):** box 13.7° (middle), truth 22.8°; forward was blocked, the input
    stayed identical for 55 steps.
- **cams-v2 (2026-11-16):** (1) no "near" question: the box's angular width and the landmark's
  width from the meshes (`VIEW_WIDTH`) give the distance; stop within footprint + 4 m (the goal).
  On v1's 217 logged near steps: 1 false stop, 2 missed (v1: 12 false stops). (2) A blank grey
  warm-up video takes the first prompt slot. (3) A forward step that moved < 0.5 m turns the next
  forward into a 45° turn toward the box's side (left if no box). Oracle `cams_oracle_v2`: 28/30,
  SPL 0.72 (unchanged).
- **Warm-up probe (stageA_ep1, 4 seeds' start poses, target turned into each camera):** P(yes) for
  the true left camera without -> with warm-up: stone tower 0.87 -> 0.98, blue tent 0.23 -> 0.68,
  red house 0.47 -> 0.87, windmill 0.20 -> 0.24. Front, back and right barely change; false "left"
  drops (tower in right camera: 0.34 -> 0.02). The windmill scene (seed 100069) is weak in every
  camera (front 0.46, right camera answered "back" 0.87): a separate problem, not the slot.

### Stage 6: Stage B training (game + replay)
**Dropped (2026-11-14):** the game stays a zero-shot test (stage 5). Revisit only if the
zero-shot game score is poor, and then prefer general first-person navigation data over game
data, so the game stays unseen.

### Stage 7: calibration and decision API
Temperature per task type; the `openjev decision` contract for video; the game control loop.

### Stage 8: final evaluation and demo
Held-out test sets scored once; latency (model compute only) and decode reported separately;
the open-world demo.

## Training data candidates (for stage 2; not decided)

General video (Stage A):

| dataset | what it teaches | notes |
|---|---|---|
| Perception Test train | physics, memory, abstraction, semantics | on the drive (7,392 questions, 1,955 videos) |
| NExT-QA train | cause and effect, timing | train videos in `NExTVideo_official.zip` |
| STAR | situated reasoning: interaction, sequence, prediction, feasibility | Charades videos |
| CLEVRER | physics, counterfactuals | rendered scenes, closest in look to a game |
| Charades-STA | when an event happens (temporal grounding) | same Charades videos as STAR |
| LLaVA-Video-178K (slice) | broad open-ended and multiple-choice video QA | very large; a slice only |
| Egocentric video (e.g. Ego4D / EPIC-KITCHENS) | first-person view, like the game camera | Ego4D needs a signed agreement |
| Image grounding (e.g. RefCOCO) | pointing at objects | keeps the "target point" skill |

Never trained on (generality check): Video-MME, MLVU, LVBench. MVBench is not a clean check: it
is built partly from Perception Test, CLEVRER and STAR.

Game (Stage B): generated by the game: frames + instruction → action or target point,
progress checks ("on the bridge?"), simple scene questions; plus replay of Stage A data.
