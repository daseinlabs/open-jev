# 01 · Ticket routing: your first decision model

**Run it as a notebook:** [`cookbook.ipynb`](cookbook.ipynb) (outputs included) · [Open in Colab](https://colab.research.google.com/github/daseinlabs/open-jev/blob/main/cookbooks/01-ticket-routing/cookbook.ipynb). The scripts below do the same from a terminal.

**Goal:** turn a CSV of labelled support tickets into a model that picks a team and a
priority for new tickets, and hands off the ones it isn't sure about.

You will touch every part of the SDK once: a `DecisionSpec`, a dataset, training,
calibration, and confident-or-defer serving. About 10 minutes on an M-series Mac.

```
tickets.csv ──build_dataset.py──▶ runs/tickets ──openjev tod train──▶ runs/tickets-model ──serve.py──▶ decision
```

## 0. The data

`make_tickets.py` writes a sample `tickets.csv` (1,200 tickets). Some mention two topics,
and 5% are deliberately mislabelled, as real routing logs are:

```
ticket_id,customer_id,message,team,priority
t00001,c0373,The website shows an error at checkout. Order #79804. I need this fixed today.,technical,urgent
```

```bash
python make_tickets.py > tickets.csv
```

To use your own tickets, export the same columns and skip this step.

## 1. Describe the decisions (`build_dataset.py`)

A **kind** is one type of decision. Each kind has a question and a function that lists
the candidates for a given state. Here both lists are fixed:

```python
SPEC = DecisionSpec('ticket-routing', [
    Kind('team', 'Which team should handle this ticket?', options(TEAMS), version='1'),
    Kind('priority', 'How urgent is this ticket?', options(PRIORITIES), version='1'),
])
```

Candidate **text** is what the model reads, so describe each option, e.g.
`'Returns: broken items, exchanges, send-backs'` rather than just `returns`. The
**payload** is what your code gets back.

Every row is built through the spec, which is what lets the SDK check later that a model
and the data it serves were built the same way:

```python
SPEC.decision(f"{ticket_id}:team", 'team', {'message': message}, choice=team, group_id=customer_id)
```

`group_id` keeps all of one customer's tickets in the same split, so test accuracy isn't
inflated by near-duplicate tickets the model saw in training.

```bash
python build_dataset.py
```

The report shows rows per split and kind. It also shows `duplicates_dropped`: identical
tickets are kept once, so the same text can't sit in both train and test.

## 2. Train

Pick a method. Everything after this step is the same for both.

**Fine-tune TOD** (NVIDIA GPU with 40 GB+, or a 64 GB Apple silicon Mac):

```bash
openjev tod train runs/tickets --method tod --out runs/tickets-model
```

This continues training [TOD](https://huggingface.co/parsecai/tod)'s adapter on your tickets,
keeps the best checkpoint on validation, fits a temperature per kind, and reports test
accuracy next to zero-shot TOD:

| test accuracy | zero-shot TOD | fine-tuned TOD |
|---|---:|---:|
| team | 84.2% | **97.0%** |
| priority | 66.3% | **100%** |

About 55 minutes on an M5 Pro (MPS, 2 epochs over 1,434 rows); much faster on an A100/H100.

**Or train small heads on a frozen Gemma 3 4B** (any ~9 GB GPU, or any Apple silicon Mac):

```bash
openjev tod train runs/tickets --out runs/tickets-model
```

Each kind (`team`, `priority`) gets its own small head on the shared backbone. Per kind,
this runs four stages:

1. **Features.** Gemma reads every ticket and every candidate once; the hidden states are
   cached in `runs/tickets/features/`. This is most of the runtime. Re-training reuses the cache.
2. **Head.** A small cross-attention head learns to score candidates against the ticket.
   The checkpoint with the best validation accuracy is kept.
3. **Calibration.** A temperature is fitted on the calibration split, so that "0.9
   confident" means right about 90% of the time.
4. **Test.** Accuracy and mean confidence per kind, on tickets never used above.

What you should see (about 6 minutes, mostly feature extraction):

```json
"test": {
  "priority": {"accuracy": 1.0,   "mean_confidence": 1.00, "n": 101},
  "team":     {"accuracy": 0.970, "mean_confidence": 0.96, "n": 101}
}
```

Team routing is harder: 30% of tickets mention two topics, and 5% of labels are wrong
(some of those land in test too). Mean confidence sits close to accuracy, so the
probabilities are honest.

## 3. Serve with a confidence threshold (`serve.py`)

```bash
python serve.py
python serve.py "I was charged twice and also my parcel is late"
```

```
team      -> shipping   confidence 0.98
priority  -> urgent     confidence 1.00
team      -> billing    confidence 0.98
priority  -> normal     confidence 1.00
```

```python
policy = load_policy('runs/tickets-model', SPEC)   # refuses a model trained with another spec
p = policy.decide(state, 'team', threshold=0.8)
if p.defer:
    ...                                            # hand to a person or a bigger model
else:
    route_to(p.payload)
```

The threshold is the product dial: raise it and the model makes fewer mistakes but defers
more. Pick it per kind from the test results.

## What to change for your data

- **Columns:** edit the loop in `build_dataset.py`. The state can be any JSON (subject,
  body, customer tier...). Everything in it is shown to the model.
- **Candidates:** edit `TEAMS` / `PRIORITIES`. If you change a list, bump that kind's
  `version`. The SDK will then refuse to serve old models with the new spec.
- **More kinds:** add a `Kind` and one more `SPEC.decision(...)` per record.

Next: [02 · traces to RL](../02-shopdesk-traces-to-rl/).
