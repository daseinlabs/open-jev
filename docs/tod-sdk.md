# Decision-model training SDK (`openjev.tod`)

`openjev.tod` fine-tunes **task-oriented decision (TOD)** models on your data. A decision
model picks one option from a list your code supplies, says how confident it is, and
defers when it isn't sure. It sits in front of a large model (or a person) and takes the
bounded, repetitive decisions: which tool to call, which order an action applies to,
which team gets a ticket.

Three ways to train, with the same data, calibration and serving:

| `--method` | What trains | Hardware |
|---|---|---|
| `tod` | The [TOD picker](https://huggingface.co/parsecai/tod) (Gemma 4 12B + LoRA): its stage-2 LoRA keeps training on your decisions | NVIDIA GPU with 40 GB+ (A100, H100, L40S, A6000), PyTorch. Also runs on a 64 GB Apple silicon Mac (MPS), slower |
| `head` (default) | A ~2M-parameter head per decision kind on a frozen Gemma 3 4B | ~9 GB GPU (PyTorch) or any Apple silicon Mac (MLX) |
| `encoder` | All of any Hugging Face model you name (default ModernBERT-large, 395M), with a per-candidate head. The only method for `Subset` targets (keep any set of candidates, e.g. retrieved passages) | NVIDIA GPU, several under `torchrun`; serves on CPU. Also trains on MPS or CPU, slower |

See [Fine-tuning TOD](#fine-tuning-tod) and [Fine-tuning any model](#fine-tuning-any-model-keep-or-drop-each-candidate).

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

## Fine-tuning any model: keep or drop each candidate

Some decisions aren't "pick one": a RAG filter reads a question and its top 10 retrieved
passages and keeps the ones worth sending to the generator, any number, possibly none. Label
those rows with `Subset(ids)` and train with `--method encoder`:

```python
from openjev.tod import Episode, ListSpec, Subset, build_dataset

spec = ListSpec('ragzilla', [dict(name='keep', question='Which passages help answer the question?',
                                  candidates='passages')])
state = {'question': 'Who wrote Hamlet?', 'passages': ['...', '...', '...']}
row = spec.decision('nq:123', 'keep', state, target=Subset(['p1', 'p3']))   # or Subset([]): none
build_dataset(spec, [Episode('nq:123', (row,))]).write('runs/filter/data')
```

```bash
openjev tod train runs/filter/data --method encoder --out runs/filter/model \
    --loss focal --focal-gamma 2 --focal-alpha 0.5 --eval-every 1000 --checkpoint-every 1000
torchrun --nproc_per_node 2 -m openjev tod train ...      # several GPUs
```

- **Any model.** `--model` takes any Hugging Face id or directory, encoders included (default
  `answerdotai/ModernBERT-large`). Every weight trains. A head (`--head mlp`, the default:
  Linear-GELU-LayerNorm-Linear; or `linear`) gives one logit per candidate. `Choice` kinds train
  a softmax over their candidates; `Subset` kinds an independent sigmoid each.
- **Layout.** A candidate sees the question and state, never the other candidates.
  - `--layout packed` (default): one sequence per decision. The question is encoded once;
    each candidate span attends to it and to itself (a block attention mask, positions
    restarting per candidate) and is pooled over its own span, so its score does not change
    when other candidates are added, removed or reordered. Needs `sdpa` attention and a model
    that takes position ids and a 4D mask; the SDK checks this when the model loads and falls
    back to `pairs` with a message if it fails. BERT-style models and ModernBERT pass.
  - `--layout pairs`: one (question + state, candidate) sequence per candidate. Works with any
    model. With [flash-attn](https://github.com/Dao-AILab/flash-attention) installed on CUDA,
    `--attention auto` uses `flash_attention_2`, which skips padding entirely.

  Packed is faster when the question and state are long relative to the candidates (it
  encodes them once instead of once per candidate). On a small retrieval set with
  ModernBERT-large it needed half the training tokens of pairs and scored 1.65x more
  decisions per second.
- **Loss.** BCE by default. `--loss focal` (gamma 2, alpha 0.5 is a good start) keeps rare
  positives from being drowned out without rescaling probabilities the way class weights do;
  gamma 0 and alpha 0.5 is BCE / 2. `--brier W` adds W times the Brier score. Other methods
  refuse these flags rather than ignore them.
- **Training.** float32 weights with bfloat16 autocast on CUDA; AdamW with linear warmup and
  decay, `--learning-rate` for the trunk (default 3e-5) and `--head-learning-rate` for the new
  head (default 3x). A step holds `--batch-tokens` tokens per GPU (default 32,768), so compute
  and the effective learning rate don't vary with passage length; `--batch-size N` uses N
  decisions instead. Token counts are computed once and cached in `DATASET/cache/`. Within a
  step, decisions are sorted by length before they are cut into forwards, and the next steps
  are tokenized on a background thread while the GPU trains. The step log shows tokens per
  step and useful/padded tokens. Gradient checkpointing is off; `--gradient-checkpointing`
  trades about 30% speed for memory on small GPUs.
- **Monitoring.** `--eval-every N` prints, on the validation split, kept-right /
  wrongly-kept / wrongly-dropped candidate counts and decisions per second.
  `--checkpoint-every N` saves `model/checkpoints/step-N`, keeping the newest `--keep-checkpoints`.
- **Continue from a trained model.** `--model runs/filter/model` starts from its trunk and
  head (and records `fine_tuned_from`): a few thousand of your own rows, `--max-steps 25
  --learning-rate 1e-5`, and the operating points are re-chosen on your calibration split.

### Operating points

A filter is judged per question. A question **survives** when the kept set still holds a
relevant candidate (`--survival any`, the default: answer survival) or all of them
(`--survival all`). Two keep rules:

| rule | keeps candidate i when |
|---|---|
| `tau` | p_i ≥ tau |
| `rho` | p_i ≥ rho × the question's highest p; the top candidate is always kept |

After training, a temperature is fitted per kind on the calibration split (Brier score over
candidates for Subset kinds, NLL for Choice kinds) and applied to every probability the model
reports. Then, for each rule and each target (`--targets 0.95,0.98,0.99`), the largest value
whose survival on the calibrated calibration split reaches the target is stored in
`model/operating_points.json`: the fewest candidates kept for that survival. `decide` uses
the default point (`--default-point`, `rho_99`) unless you name another:

```python
policy = load_policy('runs/filter/model')            # the spec comes from the model's spec.json
p = policy.decide(state, 'keep')                     # p.keep: ('p1', 'p3'), p.point: 'rho_99'
p = policy.decide(state, 'keep', point='rho_98')     # or threshold=0.05 for p >= 0.05
many = policy.decide_many([(state, 'keep') for state in states])   # packed forwards (GPU)
```

`openjev tod calibrate MODEL DATASET` re-chooses the points (and temperatures) on another
calibration split without retraining: do this first on your own data.

### Evaluating and shipping

```bash
openjev tod evaluate runs/filter/model runs/filter/data                    # test split
openjev tod evaluate runs/filter/model runs/trec-dl --out trec.json --probabilities trec.jsonl
openjev tod evaluate runs/filter/model runs/fiqa --match-survival 0.989    # compare at a reference's survival
openjev tod export runs/filter/model runs/filter/shipped
openjev tod export runs/filter/model runs/filter/hf --hf                   # a Hugging Face folder
```

`evaluate` reports the calibration error and Brier score, and for every stored point: answer
and all-relevant survival, the share of candidate tokens kept, candidates kept per question,
and the fixed top-k prefix that would need for the same survival (and its tokens).
`--match-survival X` adds `tau_matched` and `rho_matched`, the values that reach survival X on
the evaluated split itself, for comparing with another method at equal survival.

Per-source rows come from decision ids: give each row an id `SOURCE:ID` (e.g. `nq:123`,
`fever:88`) and the report adds `by_source` (`--source-sep` changes the separator). An
evaluation-only dataset (built with `fractions={'test': 1.0}`) works, and `--out` keeps the
model's own `evaluation.json` untouched.

`export` writes the trunk in bfloat16 and the head in float32 with the spec, calibration and
operating points and no checkpoints, reloads the copy, and fails if its probabilities differ
from the source's by more than 0.02 on a sample of the model's data. The copy serves with
`load_policy(MODEL, device='cpu')` (float32) or `device='cuda'` (bfloat16).

`export --hf` writes one self-contained Hugging Face folder instead: `config.json` (with
`auto_map`, the temperatures and the operating points), `model.safetensors`, the tokenizer, and
`modeling_openjev.py`, the SDK's own layout and head code. It needs transformers, not the SDK:

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('runs/filter/hf', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('runs/filter/hf')
query = 'Which passages help answer the question?\n\n{\n "question": "Who wrote Hamlet?"\n}'
model.score(tokenizer, query, passages, 'keep')   # calibrated probability per passage
model.keep(tokenizer, query, passages, 'keep')    # indices kept at the default point
```

The query is the kind's question, a blank line, then the state without its candidate list
(JSON with indent 1, or the string itself).

Models trained with 0.4.0 (pairs layout, linear head, no Subset temperature) load and give
the same results as before.

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
| `ListSpec(name, kinds)` | A declarative spec whose candidates are a list in the state (`{name, question, candidates: STATE_KEY}`). Written to `spec.json` with the dataset and copied into the model, so `load_policy(MODEL)` needs no code. |
| `Choice` / `Subset` / `Preference` / `Outcome` / `NoneCorrect` | Typed supervision. `Subset(ids)`: these candidates are right and the others wrong (empty allowed); trains with `--method encoder`. A missing target means unlabelled, never wrong. `NoneCorrect` marks a taken action that wasn't among the candidates (a recall miss). |
| `Episode(id, decisions, outcome)` | One conversation; groups rows for splitting and carries an outcome for filtering. |
| `Dataset` | Output of `build_dataset`; `write(dir)` produces the directory `openjev tod train` reads. |
| `Prediction` | `choice_id`, `confidence`, `probabilities`, `defer`, `reason`, `payload`; for Subset kinds also `keep`, `keep_payloads` and the operating `point`. |
| `Environment` | `reset(seed)`, `observe() -> (kind, state) or None`, `act(payload)`, `defer()`, `reward()`. Anything that can replay your product's decisions. |
| `load_policy(model, spec=None)` | A trained model (TOD, heads or encoder) acting through a spec (default: the model's `spec.json`): `decide`, `rollout`, `evaluate`. Refuses a model trained with a different spec. Kinds without a trained head always defer. |

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
- `tables.<param>` for SQL parameters you list in `sql_params` with your database schema
  (`schema_from_sqlite`, or `{table: {columns, references}}`): which tables the next query
  needs, among table sets joinable through foreign keys, labelled by parsing the logged SQL.
  The SQL itself stays with your LLM. See [cookbook 04](https://github.com/daseinlabs/open-jev/tree/main/cookbooks/04-sql-assistant).

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
- [`cookbooks/04-sql-assistant`](https://github.com/daseinlabs/open-jev/tree/main/cookbooks/04-sql-assistant): a chat assistant that writes SQL; the model decides when to query and which tables, from your logs and schema.
