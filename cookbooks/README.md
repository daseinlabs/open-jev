# Cookbooks

Each cookbook is a notebook plus the same steps as short scripts. Together they cover the
whole `openjev.tod` workflow, from raw data to a model that knows when to defer.

| Cookbook | You learn | Time (heads, M-series Mac) |
|---|---|---|
| [01-ticket-routing](01-ticket-routing/) ([notebook](01-ticket-routing/cookbook.ipynb)) | The core loop: labelled records → `DecisionSpec` → dataset → `openjev tod train` → confident-or-defer serving. | ~10 min |
| [02-shopdesk-traces-to-rl](02-shopdesk-traces-to-rl/) ([notebook](02-shopdesk-traces-to-rl/cookbook.ipynb)) | The full spectrum on a simulated support desk: chat traces → decision dataset → supervised fine-tuning → evaluation with deferral → RL that fixes a mistake your traces can't teach. | ~45 min |
| [03-your-own-chat-traces](03-your-own-chat-traces/) ([notebook](03-your-own-chat-traces/cookbook.ipynb)) | Running the same pipeline on your own OpenAI-style tool-calling logs, with the spec derived from your tool schemas. | depends on your data |
| [04-sql-assistant](04-sql-assistant/) ([notebook](04-sql-assistant/cookbook.ipynb)) | A chat assistant that writes SQL: the model decides when to query or answer, and which tables each query needs; your LLM writes the SQL. Labels come from your logs plus the database schema. | ~20 min |

## Setup (once)

You need Python 3.12+ and a GPU. Each cookbook trains with one of two methods:

- **`--method tod`** fine-tunes [TOD](https://huggingface.co/parsecai/tod) (Gemma 4 12B + LoRA):
  an NVIDIA GPU with 40 GB+ (A100, H100, L40S, A6000), or a 64 GB Apple silicon Mac. TOD and its
  base (`google/gemma-4-12B-it`, 24 GB, not gated) download on first use; no login needed.
- **`--method head`** (the default) trains small heads on a frozen Gemma 3 4B: an NVIDIA/AMD GPU with
  about 9 GB, or any Apple silicon Mac (MLX). See [hardware and backends](../docs/tod-sdk.md#hardware-and-backends).

```bash
pip install openjev-sdk      # imports as `openjev`, command `openjev`
hf auth login                # head method only: after accepting the Gemma terms at huggingface.co/parsecai/gemma-3-4b-it
```

For the head method, the backbone (Gemma 3 4B, 8 GB) downloads from
[`parsecai/gemma-3-4b-it`](https://huggingface.co/parsecai/gemma-3-4b-it) the first time a
cookbook needs it, into the Hugging Face cache; later runs reuse it. To use a copy you
already have, pass `--model DIR` or set `OPENJEV_MODEL=DIR`. Each cookbook writes under
`runs/` in its own folder.

Each cookbook is a notebook (`cookbook.ipynb`, with outputs, runnable on Colab) and a set of
scripts that do the same from a terminal.

Start with 01 even if your goal is 03: it takes ten minutes and shows every moving part.
