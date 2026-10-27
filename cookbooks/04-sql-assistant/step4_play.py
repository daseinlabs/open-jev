"""Step 4: let the model drive the assistant on fresh questions, with and without deferral.

The model decides run-another-query vs answer, and which tables each query uses; a simulated
LLM writes the SQL for those tables. Deferred decisions go to the parent (an oracle here)."""
import json
import random
import sys
from pathlib import Path

from openjev.tod.chat_traces import ChatTraceSpec
from openjev.tod.envs.sqldesk import SqlDesk, teacher
from openjev.tod.rl import load_policy

run = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/sql')
model = sys.argv[2] if len(sys.argv) > 2 else run/'model'
seeds = range(10_000_000, 10_000_200)  # never used for traces


def teacher_success(seed):
    desk = SqlDesk().reset(seed)
    teacher(desk, random.Random(seed + 1))
    return desk.success


policy = load_policy(model, ChatTraceSpec.load(run/'spec.json'))
report = {'assistant that wrote the traces': sum(map(teacher_success, seeds)) / len(seeds)}
for threshold in (None, 0.7, 0.9):
    report[f'model, defer below {threshold}' if threshold else 'model, never defers'] = policy.evaluate(
        SqlDesk, seeds, threshold=threshold)
print(json.dumps(report, indent=2))
