"""Step 1: collect traces. ShopDesk's imperfect teacher writes them here; in production you
export your agent's logs (OpenAI-style messages with tool calls, plus an outcome per conversation)."""
import json
from pathlib import Path

from openjev.tod.envs.shopdesk import record_trace

out = Path('runs/shopdesk/traces.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
traces = [record_trace(f'conv-{i:05d}', seed=i * 2) for i in range(1500)]
with out.open('x') as stream:
    stream.writelines(json.dumps(t) + '\n' for t in traces)
print(f'{len(traces)} traces, teacher success {sum(t["outcome"] for t in traces) / len(traces):.1%} -> {out}')
