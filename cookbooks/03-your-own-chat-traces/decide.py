"""Ask the trained model what to do next in a live conversation.

    python decide.py runs/mine conversation.json

`conversation.json` holds the messages so far, in the same format as your traces."""
import json
import sys
from pathlib import Path

from openjev.tod.chat_traces import ChatTraceSpec
from openjev.tod.rl import load_policy

run, conversation = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
spec = ChatTraceSpec.load(run/'spec.json')
policy = load_policy(run/'model', spec)  # refuses a model trained with a different spec
state = spec.state(conversation['messages'] if isinstance(conversation, dict) else conversation)
p = policy.decide(state, 'action', threshold=0.8)
print(json.dumps(dict(next=p.choice_id, confidence=round(p.confidence, 3), defer=p.defer,
                      probabilities={k: round(v, 3) for k, v in p.probabilities.items()}), indent=2))
