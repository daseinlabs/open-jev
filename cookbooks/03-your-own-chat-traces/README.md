# 03 · Your own chat traces

**Run it as a notebook:** [`cookbook.ipynb`](cookbook.ipynb) (outputs included) · [Open in Colab](https://colab.research.google.com/github/daseinlabs/open-jev/blob/main/cookbooks/03-your-own-chat-traces/cookbook.ipynb). The scripts below do the same from a terminal.

**Goal:** run the cookbook 02 pipeline on *your* agent's logs, without writing a spec or
an adapter by hand. The SDK derives both from your tool schemas.

```
traces.jsonl ──build_dataset.py──▶ runs/mine/{dataset, spec.json} ──openjev tod train──▶ runs/mine/model ──decide.py──▶ next action
```

## Trace format

One JSON object per line, in the OpenAI chat format most agent stacks already log:

```json
{"id": "conv-123",
 "tools": [{"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order that has not shipped",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}],
 "messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "Please cancel W1234"},
   {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
     "function": {"name": "get_order", "arguments": "{\"order_id\": \"W1234\"}"}}]},
   {"role": "tool", "tool_call_id": "c1", "content": "{\"id\": \"W1234\", \"status\": \"pending\"}"},
   {"role": "assistant", "content": "Done, it's cancelled."}],
 "outcome": 1.0}
```

`outcome` is optional. If you have it (task resolved, CSAT, no escalation), use it to keep
only good conversations with `--min-outcome`.

No traces handy? Use the ShopDesk ones from cookbook 02:
`../02-shopdesk-traces-to-rl/runs/shopdesk/traces.jsonl`.

## 1. Build the dataset and read the coverage report

```bash
python build_dataset.py traces.jsonl --id-param order_id=id --min-outcome 1 --out runs/mine
```

Decisions are derived from your tool schemas:

| Kind | Candidates | From |
|---|---|---|
| `action` | every tool, plus `respond` (answer in text) | the `tools` list |
| `arg.<param>` for enum parameters | the enum values | the parameter schema |
| `arg.<param>` for ID parameters | IDs seen earlier in the conversation | `--id-param` |

`--id-param order_id=id` says: `order_id` is an ID, and its values also appear under the
key `id` in tool results and JSON in the system prompt. Each candidate carries the fields
next to where it was found, e.g. `W1234 (item: blue kettle, status: pending)`, so the
model can match "the kettle" to the right ID.

**Read the report before training.** It answers "how much of my agent can this model take over?":

```json
"parameter_coverage": {"order_id": {"decisions": 2051}},
"recall_miss_rate": {"action": 0.0, "arg.order_id": 0.0}
```

- `skipped_fewer_than_2_candidates`: the ID was never seen before it was used. Add the key
  it appears under (`--id-param order_id=id,orderId`), or accept that the parent handles it.
- `free_text_left_to_parent`: a parameter the agent writes (a search query, an email body).
  It isn't a decision, so it stays with your big model.
- `recall_miss_rate > 0`: the agent used a value that wasn't among the candidates.
  The model can never choose those.

On the ShopDesk traces (1,500 conversations), the coverage report (counted across all traces, before outcome filtering) says
`order_id` produced 2,369 decisions and nothing was skipped. Without `=id` it would say
`skipped_fewer_than_2_candidates`, because the IDs live under `id` in the profile.

## 2. Train

```bash
openjev tod train runs/mine/dataset --out runs/mine/model
```

Same four stages as cookbook 01, with one head per kind. On the ShopDesk traces:

```json
"action":       {"accuracy": 0.942, "mean_confidence": 0.93, "n": 326},
"arg.order_id": {"accuracy": 0.886, "mean_confidence": 0.77, "n": 175}
```

ID selection is the harder kind, and its confidence is lower. That's the point of
calibrating per kind: set a higher deferral threshold for it.

## 3. Decide the next step in a live conversation

```bash
python decide.py runs/mine conversation.json
```

`decide.py` rebuilds the spec from `runs/mine/spec.json`, renders the conversation exactly
as training did (`spec.state(messages)`), and asks for the next `action`. If the chosen
tool takes an ID, ask for `arg.<param>` the same way with `dict(state, next_call=tool)`.
When `defer` is true, let your big model take the turn.

With the included `conversation.json` (the customer wants to cancel a yoga mat, and the
lookup says it's pending):

```json
{"next": "cancel_order", "confidence": 0.999, "defer": false, ...}
```

## Getting good results

- **Volume:** a few thousand decision rows per kind is a good start. With 1,500
  conversations (~2,200 tool-choice and ~1,200 ID training rows) this cookbook reaches 94%
  and 89%.
- **Label quality beats quantity:** filter by outcome when you can. The model learns
  whatever your agent did, mistakes included.
- **When traces aren't enough:** if your agent is systematically wrong somewhere, no
  amount of imitation fixes it. Build an `Environment` and use RL (cookbook 02, step 5).
