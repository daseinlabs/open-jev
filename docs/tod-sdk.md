# Decision-model training SDK (`openjev.tod`)

`openjev.tod` fine-tunes **task-oriented decision (TOD)** models on your data. A decision
model picks one option from a list your code supplies, says how confident it is, and
defers when it isn't sure. It sits in front of a large model (or a person) and takes the
bounded, repetitive decisions: which tool to call, which order an action applies to,
which team gets a ticket.

Two ways to train, with the same data, calibration, serving and RL:

| `--method` | What trains | Hardware |
|---|---|---|
| `tod` | The [TOD picker](https://huggingface.co/parsecai/tod) (Gemma 4 12B + LoRA): its stage-2 LoRA keeps training on your decisions | NVIDIA GPU with 40 GB+ (A100, H100, L40S, A6000), PyTorch. Also runs on a 64 GB Apple silicon Mac (MPS), slower |
| `head` (default) | A ~2M-parameter head per decision kind on a frozen Gemma 3 4B | ~9 GB GPU (PyTorch) or any Apple silicon Mac (MLX) |

See [Fine-tuning TOD](#fine-tuning-tod).

It does not write free text. Anything that needs generation stays with the parent.

## Requirements

- Python 3.12+ and one of:
  - an NVIDIA (CUDA) or AMD (ROCm) GPU on Linux or Windows, through PyTorch;
  - an Apple silicon Mac, through MLX.
- `pip install openjev-sdk` (imports as `openjev`). On Linux, pip's default PyTorch wheel
  includes CUDA. For a specific CUDA or ROCm build, install `torch` first from
  [pytorch.org](https://pytorch.org/get-started/locally/).
- The Gemma 3 4B backbone, which downloads on first use from
  [`parsecai/gemma-3-4b-it`](https://huggingface.co/parsecai/gemma-3-4b-it) into the Hugging
  Face cache. Accept the Gemma terms on that page once, then run `hf auth login`. To use a
  local copy instead, pass `--model DIR` or set `OPENJEV_MODEL`. A trained model records its
  backbone's content fingerprint, so it runs on any machine with identical weights and
  fetches them there if needed.

## Hardware and backends

The backend is chosen automatically: MLX on Apple silicon, otherwise PyTorch on the first
CUDA/ROCm GPU (then MPS, then CPU). Override it with `--backend mlx|torch` and
`--device cuda|cuda:1|mps|cpu` on `openjev tod train` and `openjev tod evaluate`, or with
`OPENJEV_BACKEND` and `OPENJEV_DEVICE` in Python code.

- **GPU memory:** about 9 GB in bfloat16 (8 GB of Gemma weights, plus features and the
  small head). An RTX 3090/4090, A10, L4, A100 or H100 is plenty.
- **GPUs without bfloat16 (T4, V100):** Gemma overflows float16, so it runs in float32,
  which needs about 17 GB.
- **CPU:** works, but expect seconds per row.

Models move between backends. Heads are saved with the same weight names on both, and
PyTorch features match MLX's (cosine similarity 0.9998). A model trained on a Mac serves
on an NVIDIA GPU, and the other way round, without retraining.

Results on PyTorch compared with MLX (PyTorch measured on MPS, which shares the CUDA code path):

| | MLX | PyTorch |
|---|---|---|
| Cookbook 01 test accuracy, mean of 5 seeds: team / priority | 96.4% / 99.8% | 96.2% / 98.6% |
| Cookbook 02 MLX-trained model served on each backend: success with no deferral / deferring below 0.7 | 70.0% / 81.5% | 69.5% / 82.0% |
| Cookbook 02 after `improve` (same SFT model, same seed): closed-window cases solved / success with no deferral | 24/24 / 78.0% | 24/24 / 83.5% |

The PyTorch optimizer reproduces MLX's Adam update (no bias correction), so the learning
rates in this guide and the cookbooks carry over unchanged.

## Fine-tuning TOD

```bash
openjev tod train runs/my-dataset --method tod --out runs/my-tod
```

TOD reads a decision as a lettered multiple-choice prompt and scores the answer letters.
With up to 16 options, that letter read is TOD's whole answer (its stage-1 shortlist keeps
every option), so fine-tuning continues training the stage-2 LoRA (rank 16 on q/k/v/o,
21M parameters) with cross-entropy over the letters, the way TOD was trained. The SDK
renders exactly TOD's prompt (byte-identical to `daseinlabs/tod`, checked in the tests), and
its zero-shot letter logits match TOD's own scorer exactly.

- **One adapter for all kinds.** TOD is a general picker; the question text tells kinds
  apart. The checkpoint with the best validation accuracy is kept, including the starting
  adapter if fine-tuning doesn't beat it.
- **Calibration per kind**, fitted on the calibration split, as for heads.
- **The report includes zero-shot TOD** on the same test split, so you see what fine-tuning bought.
- **Downloads:** `parsecai/tod` (0.1 GB adapter) and `google/gemma-4-12B-it` (24 GB,
  Apache-2.0, not gated) on first use.
- **Limits:** up to 26 candidates per decision; text states (TOD's image path isn't wired in yet).
- **Defaults:** 2 epochs, learning rate 2e-5, gradient accumulation 8, LoRA in float32 on a
  bfloat16 base, gradient checkpointing. `--max-train N` trains on a random subset.

`load_policy(model, spec)` returns the right policy for either method, with `decide`,
`rollout`, `evaluate`, and RL (`improve`, `reinforce`). On TOD, RL updates the shared
adapter, and the rollback guard checks every kind after each update.

## The pipeline

```
your data ──adapter──▶ Decisions ──build_dataset──▶ dataset dir ──openjev tod train──▶ model
                                                                                      │
                          Environment (simulator) ◀────── improve (optional RL) ◀───┘
```

1. **Adapt.** Turn your records or traces into `Decision` rows. Always build them
   through a `DecisionSpec`, which owns candidate enumeration.
2. **Build.** `build_dataset` validates every row against the spec, drops duplicate
   inputs, splits by group (customer or conversation) into train / validation /
   calibration / test, and reports label coverage per decision kind.
3. **Train.** `openjev tod train DATASET --out MODEL` trains **one head per decision kind**
   on a shared frozen backbone. Per kind it caches features, trains, fits a temperature on
   the calibration split, and reports test accuracy. The model records the spec's fingerprint.
4. **Serve.** `load_policy(MODEL, spec).decide(state, kind, threshold)` returns a
   `Prediction`: execute its payload when confident, defer otherwise.
5. **RL (optional).** With an environment that gives a reward, `improve` teaches the model
   what your data never showed.

## Primitives

| Primitive | What it is |
|---|---|
| `Candidate(id, text, payload)` | One option. The model reads `text`; your code executes `payload`. |
| `Kind(name, question, candidates, version)` | A decision type. `candidates(state)` lists the options for a state. Bump `version` whenever that function's behaviour changes. |
| `DecisionSpec(name, kinds)` | All kinds for one product. Builds rows (`decision`), rejects stale rows (`check`), serves (`decide`). Its `fingerprint` binds data and models together. |
| `Decision` | One training row: `kind`, `state`, `question`, `candidates`, `target`, `group_id`. |
| `Choice` / `Preference` / `Outcome` / `NoneCorrect` | Typed supervision. A missing target means unlabelled, never wrong. `NoneCorrect` marks a taken action that wasn't among the candidates (a recall miss). |
| `Episode(id, decisions, outcome)` | One conversation; groups rows for splitting and carries an outcome for filtering. |
| `Dataset` | Output of `build_dataset`; `write(dir)` produces the directory `openjev tod train` reads. |
| `Prediction` | `choice_id`, `confidence`, `probabilities`, `defer`, `reason`, `payload`. |
| `Environment` | `reset(seed)`, `observe() -> (kind, state) or None`, `act(payload)`, `defer()`, `reward()`. Anything that can replay your product's decisions. |
| `load_policy(model, spec)` | A trained model (TOD or heads) acting through a spec: `decide`, `rollout`, `evaluate`. Refuses a model trained with a different spec. Kinds without a trained head always defer. |

## Labelling from what actually happened

When your logs record the action taken (a tool call, a routing), label with
`taken=` and let the spec match it to a candidate:

```python
spec.decision('conv-7:2:action', 'action', state, taken='cancel_order')
```

If the taken action isn't among the candidates the spec enumerated, the row becomes
`NoneCorrect` instead of disappearing. The dataset report's `recall_miss_rate` says how
often that happens per kind. A high rate means your candidate function misses options
the agent really uses, and the model can never choose them.

## Chat traces without writing a spec

For OpenAI-style tool-calling logs, `openjev.tod.chat_traces` derives the spec from the
tool schemas:

```python
from openjev.tod.chat_traces import spec_from_tools, episodes_from_traces

spec = spec_from_tools('my-agent', traces[0]['tools'], id_params={'order_id': ['id']})
episodes, coverage = episodes_from_traces(traces, spec)
```

- `action`: which tool to call next, plus `respond` (answer in text).
- `arg.<param>` for enum parameters (candidates come from the enum).
- `arg.<param>` for ID parameters you list in `id_params` (candidates are the values seen
  earlier in the conversation under that key or its aliases).
- Free-text parameters are not decisions. `coverage` counts them as `free_text_left_to_parent`.

## Confidence and deferral

The calibrated probability of the chosen option is the confidence. Pick a threshold
per kind from the evaluation: higher thresholds mean fewer mistakes and more hand-offs.
Calibration is fitted on its own split, and must be refitted after any further training,
including RL (`openjev tod evaluate MODEL DATASET --refit`).

## Why one head per kind

Choosing a tool and matching an ID are different tasks. In cookbook 03, a single head
trained on both scored 48% on order selection (near chance), while a separate order head
on the same rows scored 89%. Heads are small (~2M parameters) and share the backbone, so
features are still extracted once per row. A kind with no labelled rows in some split
is skipped at training time and always defers.

## Improving with reward

`openjev.tod.rl` offers two methods. Both keep the backbone frozen, update only the
heads, mix in the SFT training rows of each kind as an anchor, and write a new model
(refit calibration afterwards).

- **`improve`** (recommended) needs an environment that is deterministic per seed. For
  each episode that failed, it replays the episode to each decision (latest first), tries
  every other candidate, and lets the policy finish. The latest decision where an
  alternative earns more reward becomes a label. It finds actions the model currently gives
  ~0 probability. After each iteration, a kind whose validation accuracy drops more than
  `tolerance` (2 points) below the starting model is rolled back. Each update averages
  over the new labels and 256 anchor rows together, so a handful of labels can't outweigh
  the anchor and over-generalize. In cookbook 02, across three runs, it fixed the teacher's
  blind spot every time (24/24 closed-window cases, from 2/24 and 9/24) and took never-defer
  success from 70–75% to 78–83%. Order accuracy on test can drop a few points; check the report.
- **`reinforce`** is plain policy gradient. It is cheaper per step, but it only learns about
  actions it happens to sample. In cookbook 02 it never sampled the one action that mattered
  (0.0% probability under the SFT model) and didn't improve.

Why "latest first": branching replays the same seed, so an early alternative can succeed
on information the model couldn't have had yet. In cookbook 02, "transfer immediately"
succeeds whenever the hidden return window is closed. Labelling it would teach the model
to transfer at the first turn of every return request. The latest decision is the most
informed one.

## Cookbooks

- [`cookbooks/01-ticket-routing`](https://github.com/daseinlabs/open-jev/tree/main/cookbooks/01-ticket-routing): labelled CSV to a routing model in about 10 minutes.
- [`cookbooks/02-shopdesk-traces-to-rl`](https://github.com/daseinlabs/open-jev/tree/main/cookbooks/02-shopdesk-traces-to-rl): chat traces to dataset to SFT to RL in a simulated support desk.
- [`cookbooks/03-your-own-chat-traces`](https://github.com/daseinlabs/open-jev/tree/main/cookbooks/03-your-own-chat-traces): the same pipeline on your logs, with the spec derived from your tool schemas.
