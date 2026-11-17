"""Your chat traces -> decision dataset, with the spec derived from your tool schemas.

    python build_dataset.py traces.jsonl --id-param order_id=id --out runs/mine

Read the coverage report before training: it says which tool parameters became decisions
and which were skipped (and why)."""
import argparse
import json
from pathlib import Path

from openjev.tod import build_dataset
from openjev.tod.chat_traces import episodes_from_traces, spec_from_tools

p = argparse.ArgumentParser()
p.add_argument('traces', type=Path, help='JSONL of {id, tools, messages, outcome}')
p.add_argument('--out', type=Path, default=Path('runs/mine'))
p.add_argument('--name', default='my-agent')
p.add_argument('--id-param', action='append', default=[], metavar='PARAM[=ALIAS,...]',
               help='a parameter whose values are IDs seen earlier in the conversation, plus the keys '
                    'they appear under in tool results (e.g. order_id=id)')
p.add_argument('--min-outcome', type=float, default=None, help='keep conversations with outcome >= this')
p.add_argument('--no-respond', action='store_true', help='do not model "answer in text" as an action')
args = p.parse_args()

traces = [json.loads(line) for line in args.traces.open() if line.strip()]
id_params = {}
for item in args.id_param:
    param, _, aliases = item.partition('=')
    id_params[param] = [a for a in aliases.split(',') if a]
spec = spec_from_tools(args.name, traces[0]['tools'], id_params=id_params, respond=not args.no_respond)
episodes, coverage = episodes_from_traces(traces, spec)
dataset = build_dataset(spec, episodes, min_outcome=args.min_outcome)
dataset.write(args.out/'dataset')
spec.save(args.out/'spec.json')
print(json.dumps(dict(kinds=list(spec.kinds), parameter_coverage=coverage, **dataset.report), indent=2))
