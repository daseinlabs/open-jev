"""Step 1: collect traces. SqlDesk's assistant writes 1,500 here; in production, export your
assistant's logs (OpenAI-style messages with run_sql tool calls, plus an outcome per conversation)."""
import json
from pathlib import Path

from openjev.tod.envs.sqldesk import record_trace

out = Path('runs/sql/traces.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
traces = [record_trace(f'chat-{i:05d}', seed=i * 2) for i in range(1500)]
with out.open('x') as stream:
    stream.writelines(json.dumps(t) + '\n' for t in traces)
queries = sum(1 for t in traces for m in t['messages'] if m.get('tool_calls')) / len(traces)
print(f'{len(traces)} traces, {queries:.1f} queries per conversation, '
      f'assistant success {sum(t["outcome"] for t in traces) / len(traces):.1%} -> {out}')
