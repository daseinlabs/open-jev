# Cookbooks

Each cookbook is a folder of short scripts you run in order. Together they cover the
whole `openjev.tod` workflow, from raw data to a model that knows when to defer.

| Cookbook | You learn | Time on an M-series Mac |
|---|---|---|
| [01-ticket-routing](01-ticket-routing/) | The core loop: labelled records → `DecisionSpec` → dataset → `openjev tod train` → confident-or-defer serving. | ~10 min |
| [02-shopdesk-traces-to-rl](02-shopdesk-traces-to-rl/) | The full spectrum on a simulated support desk: chat traces → decision dataset → supervised fine-tuning → evaluation with deferral → RL that fixes a mistake your traces can't teach. | ~45 min |
| [03-your-own-chat-traces](03-your-own-chat-traces/) | Running the same pipeline on your own OpenAI-style tool-calling logs, with the spec derived from your tool schemas. | depends on your data |

## Setup (once)

You need Python 3.12+, the Gemma 3 4B weights, and either an NVIDIA/AMD GPU with about 9 GB
of memory (PyTorch) or an Apple silicon Mac (MLX). The backend is picked automatically;
see [hardware and backends](../docs/tod-sdk.md#hardware-and-backends).

```bash
pip install openjev-sdk      # imports as `openjev`, command `openjev`
hf auth login                # accept the Gemma license on Hugging Face first
hf download google/gemma-3-4b-it --local-dir ~/models/gemma-3-4b-it
export OPENJEV_MODEL=~/models/gemma-3-4b-it
```

`openjev tod train` uses `$OPENJEV_MODEL` (or `--model PATH`). A trained model remembers
its backbone's location, so later steps find it on their own. Each cookbook writes under
`runs/` in its own folder.

Start with 01 even if your goal is 03: it takes ten minutes and shows every moving part.
