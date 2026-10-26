"""Route a new ticket: act when confident, hand off to a person (or a bigger model) otherwise."""
import sys

from openjev.tod.rl import HeadPolicy

from build_dataset import SPEC

policy = HeadPolicy('runs/tickets-model', SPEC)  # checks the artifact was trained with this SPEC
state = {'message': ' '.join(sys.argv[1:]) or 'My parcel never showed up and I need it for tomorrow!'}
for kind in SPEC.kinds:
    p = policy.decide(state, kind, threshold=0.8)
    print(f'{kind:9} -> {p.choice_id:10} confidence {p.confidence:.2f}' + ('  (defer)' if p.defer else ''))
