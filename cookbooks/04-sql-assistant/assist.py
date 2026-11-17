"""Where the model sits in your assistant loop: before each LLM turn.

    python assist.py runs/sql conversation.json

It answers two questions with a confidence: run another query or answer now, and if a
query, which tables it needs. Your LLM then writes the SQL with only those tables' schema
in its prompt. When `defer` is true, let your LLM decide as it does today."""
import json
import sys
from pathlib import Path

from openjev.tod.chat_traces import ChatTraceSpec
from openjev.tod.rl import load_policy

run, conversation = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
spec = ChatTraceSpec.load(run/'spec.json')
policy = load_policy(run/'model', spec)
state = spec.state(conversation['messages'])
step = policy.decide(state, 'action', threshold=0.8)
print(json.dumps(dict(next=step.choice_id, confidence=round(step.confidence, 3), defer=step.defer)))
if step.choice_id == 'run_sql' and not step.defer:
    tables = policy.decide(dict(state, next_call='run_sql'), 'tables.sql', threshold=0.8)
    schema = spec.sql_params['sql']['schema']
    print(json.dumps(dict(tables=tables.payload, confidence=round(tables.confidence, 3), defer=tables.defer,
                          schema_for_the_llm={t: schema[t]['columns'] for t in (tables.payload if tables.payload != 'schema' else [])})))
