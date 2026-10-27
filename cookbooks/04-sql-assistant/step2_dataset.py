"""Step 2: traces + database schema -> decision dataset (and spec.json for serving).

    python step2_dataset.py                       # the SqlDesk traces and schema
    python step2_dataset.py my_traces.jsonl --sqlite my.db --sql-param query --min-outcome 1

The schema comes from your database (SQLite here; for Postgres/MySQL, build the same
{table: {columns: [...], references: [...]}} dict from information_schema)."""
import argparse
import json
import sqlite3
from pathlib import Path

from openjev.tod import build_dataset
from openjev.tod.chat_traces import episodes_from_traces, schema_from_sqlite, spec_from_tools

p = argparse.ArgumentParser()
p.add_argument('traces', type=Path, nargs='?', default=Path('runs/sql/traces.jsonl'))
p.add_argument('--sqlite', type=Path, help='your database, to read the schema from (default: the SqlDesk schema)')
p.add_argument('--sql-param', default='sql', help='the tool argument that holds the SQL')
p.add_argument('--max-tables', type=int, default=3, help='largest join to offer as one candidate')
p.add_argument('--min-outcome', type=float, default=1.0)
p.add_argument('--out', type=Path, default=Path('runs/sql'))
args = p.parse_args()

if args.sqlite:
    schema = schema_from_sqlite(sqlite3.connect(args.sqlite))
else:
    from openjev.tod.envs.sqldesk import SCHEMA as schema
traces = [json.loads(line) for line in args.traces.open() if line.strip()]
spec = spec_from_tools('sql-assistant', traces[0]['tools'],
                       sql_params={args.sql_param: dict(schema=schema, max_tables=args.max_tables)})
episodes, coverage = episodes_from_traces(traces, spec)
dataset = build_dataset(spec, episodes, min_outcome=args.min_outcome)
dataset.write(args.out/'dataset')
spec.save(args.out/'spec.json')
print(json.dumps(dict(kinds=list(spec.kinds), parameter_coverage=coverage, **{k: dataset.report[k] for k in (
    'splits', 'recall_miss_rate', 'duplicates_dropped', 'duplicates_with_different_label')}), indent=2))
