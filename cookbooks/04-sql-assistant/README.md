# 04 · A SQL assistant: decide around the queries

**Run it as a notebook:** [`cookbook.ipynb`](cookbook.ipynb) · [Open in Colab](https://colab.research.google.com/github/daseinlabs/open-jev/blob/main/cookbooks/04-sql-assistant/cookbook.ipynb). The scripts below do the same from a terminal.

**Goal:** your chat assistant answers questions by writing SQL, typically a handful of
queries per conversation, and you log whether each answer was right. Fine-tune a decision
model on those logs so it takes the decisions *around* the SQL:

| decision | candidates | why it matters |
|---|---|---|
| `action` | run another query (`run_sql`) or answer now (`respond`) | answering before the query that answers the question is a common way to be wrong |
| `tables.sql` | which tables the next query needs: every set of tables joinable through foreign keys (up to 3), or `schema` (look at the catalog) | forgetting a join (e.g. querying `orders` alone for a per-region question) is the other one |

Your LLM still writes the SQL, given only the chosen tables' schema: a shorter, more focused
prompt. The model never writes SQL; it routes, with a calibrated confidence, and defers to
your LLM when unsure.

```
traces.jsonl + schema ──step2_dataset.py──▶ runs/sql/{dataset, spec.json} ──openjev tod train──▶ runs/sql/model
                                                                                        │
                                       assist.py (in your loop) ◀─────── step4_play.py (measure) ◀┘
```

## The simulated assistant

`openjev.tod.envs.sqldesk` is a shop database in SQLite with five tables:

```
customers(id, name, region, signup_date)        products(id, name, category, price)
orders(id, customer_id, order_date, total, status)   order_items(order_id, product_id, qty, price)
refunds(id, order_id, amount, reason, refund_date)
```

Users ask seven kinds of question ("How much has Ana Silva been refunded in total?", "Which
product category sold the most units in 2026-03?"). The assistant that writes the logs lists
tables, peeks at rows, runs the queries the question needs (one or two), sometimes runs a
sanity check, and answers: 1 to 6 queries, about 3 on average. It is right 90% of the time.
It forgets a join in 12% of the questions that need one, and sometimes answers one query
too early.

## Step 1 · Traces

```bash
python step1_traces.py
```

The format is OpenAI tool calling, which most agent stacks already log: a `run_sql(sql)`
tool, its results as `tool` messages, the final answer as assistant text, and an `outcome`.

```json
{"id": "chat-00003", "tools": [{"type": "function", "function": {"name": "run_sql", ...}}],
 "messages": [
   {"role": "system", "content": "You are an analytics assistant for a shop. ..."},
   {"role": "user", "content": "What was the most common refund reason in 2026-04?"},
   {"role": "assistant", "tool_calls": [{"function": {"name": "run_sql", "arguments": "{\"sql\": \"SELECT name FROM sqlite_master WHERE type = 'table'\"}"}}]},
   {"role": "tool", "content": "{\"columns\": [\"name\"], \"rows\": [[\"customers\"], ...], \"row_count\": 5}"},
   {"role": "assistant", "tool_calls": [{"function": {"name": "run_sql", "arguments": "{\"sql\": \"SELECT reason FROM refunds WHERE ... LIMIT 1\"}"}}]},
   {"role": "tool", "content": "{\"columns\": [\"reason\"], \"rows\": [[\"late delivery\"]], \"row_count\": 1}"},
   {"role": "assistant", "content": "The answer is late delivery."}],
 "outcome": 1.0}
```

If your logs differ (one tool per query type, a different argument name, results as text),
convert them to this shape first; it's the only contract.

## Step 2 · Traces + schema → decisions

```bash
python step2_dataset.py
python step2_dataset.py my_traces.jsonl --sqlite my.db --sql-param query   # your data
```

Labels come from the logs with no hand-labelling:

- **`action`**: was the assistant's next message a `run_sql` call or the answer?
- **`tables.sql`**: the tables the logged SQL reads (parsed from its `FROM` / `JOIN` clauses),
  matched to a candidate set. Catalog queries (`sqlite_master`, `information_schema`) are `schema`.

The schema is `{table: {columns: [...], references: [tables]}}`. `schema_from_sqlite`
reads it from a SQLite file; for Postgres or MySQL, build the same dict from
`information_schema`. Candidates are every set of up to `--max-tables` tables connected through
foreign keys (14 here), each described by its columns.

Read the report before training:

- `recall_miss_rate` for `tables.sql`: queries whose tables aren't a candidate (a join of more
  than `--max-tables`, or tables with no foreign key between them). Raise `--max-tables`, or
  add the missing `references` to the schema.
- `duplicates_with_different_label`: identical states labelled differently. Logs repeat (the
  same question, the same first step), and an assistant doesn't always do the same thing. The
  first occurrence is kept; a high count means the decision is partly a coin flip.
- `--min-outcome 1` keeps only conversations that ended with a right answer.

## Step 3 · Fine-tune

```bash
openjev tod train runs/sql/dataset --method tod --out runs/sql/model   # fine-tune TOD (40 GB+ GPU)
openjev tod train runs/sql/dataset --out runs/sql/model                # or heads on Gemma 3 4B (~9 GB)
```

## Step 4 · Measure it driving the assistant

```bash
python step4_play.py
```

The model drives the assistant on 200 questions never used for traces: it decides when to
query and which tables; a simulated LLM writes the SQL for those tables (right when they cover
what the question needs, a plausible wrong query when a needed table is missing).

## Step 5 · Put it in your loop

```bash
python assist.py runs/sql conversation.json
```

```python
spec = ChatTraceSpec.load('runs/sql/spec.json')
policy = load_policy('runs/sql/model', spec)

state = spec.state(messages)                      # the conversation so far, as in training
step = policy.decide(state, 'action', threshold=0.8)
if step.defer:
    ...                                           # your LLM decides, as today
elif step.choice_id == 'respond':
    ...                                           # ask your LLM for the final answer
else:
    tables = policy.decide(dict(state, next_call='run_sql'), 'tables.sql', threshold=0.8)
    whole = tables.defer or tables.payload == 'schema'   # 'schema': the LLM wants to look at the catalog
    schema = full_schema if whole else {t: full_schema[t] for t in tables.payload}
    sql = your_llm.write_sql(messages, schema)    # a smaller, focused prompt
```

## What carries over

- **Any tool that takes SQL** works: name its argument with `--sql-param`. Other tools in the
  same logs become `action` candidates automatically.
- **Outcomes are the lever.** Filter on them for SFT. If your assistant is systematically wrong
  somewhere (always forgets one join), imitation keeps the mistake; build an `Environment`
  around a copy of your database and use `improve` ([cookbook 02](../02-shopdesk-traces-to-rl/)).
  `SqlDesk` is a template for that.
