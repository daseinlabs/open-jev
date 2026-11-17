"""Step 2: traces -> decision dataset, keeping only conversations that succeeded."""
import json

from openjev.tod import build_dataset

from adapter import SPEC, trace_to_episode

with open('runs/shopdesk/traces.jsonl') as stream:
    episodes = [trace_to_episode(json.loads(line)) for line in stream]
dataset = build_dataset(SPEC, episodes, min_outcome=1.0)
dataset.write('runs/shopdesk/dataset')
print(json.dumps(dataset.report, indent=2))
