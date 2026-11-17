# 02 · ShopDesk: from chat traces to SFT to RL

**Run it as a notebook:** [`cookbook.ipynb`](cookbook.ipynb) (outputs included) · [Open in Colab](https://colab.research.google.com/github/daseinlabs/open-jev/blob/main/cookbooks/02-shopdesk-traces-to-rl/cookbook.ipynb). The scripts below do the same from a terminal.

**Goal:** see the whole lifecycle on one realistic problem. You take the logs of an
existing support agent, turn them into a decision dataset, fine-tune a small model on
them, measure it with deferral, and then use RL to fix a mistake the logs can't teach.

About 45 minutes on an M-series Mac, mostly Gemma reading text.

```
step1  traces.jsonl          an imperfect agent's chat logs (OpenAI format, with outcomes)
step2  dataset/              adapter.py cuts each conversation into decisions
step3  sft/                  openjev tod train: fine-tune TOD (or heads), calibration, test metrics
step4  (report)              the model runs fresh conversations, with and without deferral
step5  rl/                   improve: learn from reward in the simulator, starting from the SFT model
```

## The problem

ShopDesk (`openjev.tod.envs.shopdesk`) is a simulated retail support desk. A customer
writes in about one of their orders. The agent calls tools until the request is resolved:

| tool | when it's right |
|---|---|
| `get_order(order_id)` | first: the status is only visible after a lookup |
| `cancel_order(order_id)` | cancel or return, order still pending |
| `return_order(order_id)` | cancel or return, delivered, return window **open** |
| `transfer_to_human()` | shipped (can't cancel in transit), or return window **closed** |
| `send_tracking(order_id)` | the customer asked where the order is |

The traces come from a **teacher** agent that is right most of the time but has real
failure modes: it sometimes skips the lookup, looks up the wrong order, or gives up. It
also has one **systematic blind spot**: it ignores the return window. Every conversation
where that matters fails, so no successful trace ever shows what to do with a closed
window. That's the gap RL closes in step 5.

## Step 1 · Traces

```bash
python step1_traces.py
```

1,500 conversations in the format your production logs probably already use: a system
prompt with the customer profile, the user message, assistant `tool_calls`, `tool`
results, and an `outcome` (1 if the request was resolved correctly).

```
1500 traces, teacher success 72.1% -> runs/shopdesk/traces.jsonl
```

## Step 2 · Traces → decisions (`adapter.py`)

This is the part you write for your own data, and the part to read most closely.

**Two kinds of decision:** `action` (which tool to call next) and `arg.order_id` (which
order the call is about). Each kind lists its candidates from the state:

```python
SPEC = DecisionSpec('shopdesk', [
    Kind('action', 'Which tool should the agent call next?', lambda state: ACTIONS, version='1'),
    Kind('arg.order_id', 'Which order should this tool call use?', order_candidates, version='1'),
])
```

**Cutting a conversation:** each assistant tool call produces one `action` row and, if the
tool takes an order, one `arg.order_id` row. The state is the conversation *before* that
call. The label is what the agent actually did (`taken=`):

```python
SPEC.decision(f'{row}:action', 'action', render_state(customer, message, history), taken=tool)
```

**`render_state` is shared with the live environment.** The model is trained on exactly
the view it gets when serving. Most train/serve bugs come from these two drifting apart.

```bash
python step2_dataset.py
```

`min_outcome=1.0` keeps only successful conversations, so the model doesn't learn the
teacher's random mistakes. The report's `recall_miss_rate` is 0 here because every tool
and order the agent used was among the candidates. On real data, check it first.

## Step 3 · Supervised fine-tuning

```bash
openjev tod train runs/shopdesk/dataset --method tod --out runs/shopdesk/sft   # fine-tune TOD (40 GB+ GPU)
openjev tod train runs/shopdesk/dataset --out runs/shopdesk/sft                # or heads on Gemma 3 4B (~9 GB)
```

The numbers below are from the `head` method. Steps 4 and 5 work the same for both:
`load_policy` picks the right model type, and `improve` updates TOD's shared adapter (with
smaller default batches, since every row is a 12B pass) or the per-kind heads.

One head per kind, measured on a held-out test split (calibrated):

| kind | accuracy | mean confidence | n |
|---|---:|---:|---:|
| `action` | 90.5% | 0.91 | 211 |
| `arg.order_id` | 88.6% | 0.89 | 175 |

Mean confidence tracks accuracy closely: that's what calibration buys. Without it, the
deferral threshold in step 4 would be guesswork.

## Step 4 · Play fresh conversations, with deferral

```bash
python step4_play.py runs/shopdesk/sft
```

The model now *is* the agent: `policy.rollout` asks the environment for the next
decision, builds it through `SPEC`, and executes the chosen payload. With a threshold,
low-confidence decisions go to the parent (an oracle here; your big model in production).

On 200 fresh conversations (seeds never used for traces or training):

| agent | success | decisions the model handled | conversations with no hand-off |
|---|---:|---:|---:|
| teacher (the agent that wrote the traces) | 72.5% | n/a | n/a |
| SFT model, never defers | 70.0% | 100% | 100% |
| SFT model, defer below 0.7 | 81.5% | 90% | 63% |
| SFT model, defer below 0.9 | 95.0% | 66% | 12% |

On its own, the SFT model is about as good as the agent it learned from. That's
expected: it imitates. Deferral is what makes it usable. At 0.7 it takes 90% of decisions
itself and still beats the teacher.

Where do the 60 failures of the never-defer model come from?

- **22 of the 24 closed-window conversations failed.** Every closed-window case in the
  training data was a failure and got filtered out, so the model has never seen the right
  move. More data of the same kind won't fix this.
- **38 failures elsewhere**, mostly `send_tracking` where the customer wanted an action.
  These are ordinary model errors, and deferral catches most of them.

## Step 5 · RL: learn what the traces never showed

```bash
python step5_rl.py
openjev tod evaluate runs/shopdesk/rl runs/shopdesk/dataset --refit
python step4_play.py runs/shopdesk/rl
```

`improve` plays 128 conversations per iteration with the current model. For each one that
failed, it goes back through the decisions **from the last one backwards**, resets the desk
to the same seed, replays up to that decision, tries every other candidate there, and lets
the model finish. The latest decision where an alternative succeeds becomes a new
training label. Labels are trained with cross-entropy, mixed with the SFT rows of the
same kind (the *anchor*), so what the traces already taught is kept.

```
{"iteration": 1, "greedy_mean_reward": 0.77, "new_labels": 30, "updated": ["action"], "rolled_back": ["arg.order_id"]}
{"iteration": 2, "greedy_mean_reward": 0.82, "new_labels": 22, "updated": ["action", "arg.order_id"], "rolled_back": []}
{"iteration": 3, "greedy_mean_reward": 0.85, "new_labels": 17, "updated": ["action"], "rolled_back": ["arg.order_id"]}
```

Same 200 evaluation conversations as step 4. RL was run three times: from the SFT model
above, and twice (two RL seeds) from a second SFT model trained the same way with newer
library versions:

| model | never defers | defer below 0.7 (handled) | defer below 0.9 (handled) | closed-window cases solved |
|---|---:|---:|---:|---:|
| SFT | 70.0% | 81.5% (90%) | 95.0% (66%) | 2 / 24 |
| SFT + RL | **78.0%** | **98.0%** (81%) | 100% (62%) | **24 / 24** |
| second SFT | 75.0% | 96.5% (84%) | 100% (62%) | 9 / 24 |
| second SFT + RL, seed A | **82.5%** | **97.0%** (83%) | 100% (64%) | **24 / 24** |
| second SFT + RL, seed B | **83.0%** | **97.5%** (83%) | 100% (62%) | **24 / 24** |

Every run fixes the blind spot. Tool-choice test accuracy holds (90.0–90.5%). Order
accuracy can lose a few points (88.6% → 87.4%; on the second model 90.3% → 84.6% and
88.0%), because the order head's RL labels are hard cases. Read `rolled_back` and the
test report, and set that kind's deferral threshold from the refitted calibration.

Four design choices in `improve` matter, and each comes from a failure we hit building this:

1. **Try every alternative instead of sampling.** Plain policy gradient (`reinforce`, also
   in the SDK) never improved here. The SFT model gives `transfer_to_human` 0.0%
   probability in closed-window states, so sampling never tries it and never sees it rewarded.
2. **Label the latest decision only.** Branching replays the same seed, so an *early*
   alternative can succeed on information the model didn't have yet. "Transfer at the very
   first turn" wins whenever the hidden window is closed. Labelling that taught the model
   to transfer every return request, and success fell to 44%.
3. **Roll back a kind that gets worse on validation.** Some failures are hard cases the
   head can't fit without forgetting others. Forcing them in dropped order accuracy to 72%.
   After each iteration, a kind whose validation accuracy falls more than 2 points is
   rolled back; that's the `rolled_back` field above.
4. **Weigh every row the same.** Each update mixes up to 32 labels with 256 anchor rows and
   averages over all of them. An earlier version averaged the few labels as one term and the
   anchor as another, so each label counted as much as eight anchor rows. The head then
   learned "return request → transfer" instead of "closed window → transfer", lost up to 10
   points of validation accuracy, and the guard rejected the fix in some runs (9/24 solved).
   Weighted per row, the anchor's open-window examples keep the correction specific.

Calibration is refitted after RL (`--refit`) because the head's probabilities changed.

## What carries over to your data

- **Write `adapter.py` for your logs**, or skip it with the generic adapter in
  [cookbook 03](../03-your-own-chat-traces/).
- **SFT copies your agent.** Filtering by outcome removes random mistakes, but not
  systematic ones.
- **RL needs an `Environment`:** a simulator, a sandbox copy of your system, or a replay
  harness that can score outcomes. It needs only `reset`, `observe`, `act`, `defer` and `reward`.
