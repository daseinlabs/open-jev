"""Generic adapter: OpenAI-style tool-calling traces -> decisions, with no hand-written spec.

Expected trace shape (one JSON object per line):

    {"id": "...", "tools": [{"type": "function", "function": {"name", "description", "parameters"}}],
     "messages": [{"role": "system"|"user"|"assistant"|"tool", ...}], "outcome": 1.0}

Decisions derived from the tool schemas:
- `action`: which tool to call next, plus `respond` (answer in text) when enabled;
- `arg.<param>`: one kind per parameter that can be picked rather than written:
  enum parameters (candidates = the enum), and ID-like parameters you name in `id_params`
  (candidates = values seen earlier in the conversation under that key or its aliases);
- `tables.<param>` for SQL parameters you name in `sql_params` with your database schema:
  which tables the next query needs (candidates = table sets joinable through foreign keys,
  plus `schema` for catalog lookups). The model routes; your LLM still writes the SQL,
  given only those tables.

Free-text parameters are not decisions; calls that need them stay with the parent model.
`state_from_messages` renders the model's view and must also be used when serving.
"""
from __future__ import annotations

import json
import re
from itertools import combinations

from ..decision import digest
from .schema import Candidate, Episode
from .spec import DecisionSpec, Kind

RESPOND = 'respond'


def _tools(tools):
    return {t['function']['name']: t['function'] for t in tools}


def _scan(value, keys, found):
    """Collect scalar values stored under any of `keys`, anywhere in nested JSON, each with a
    short description from its sibling fields (so "W1" reads as "W1 (item: kettle)")."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in keys and isinstance(v, (str, int)) and str(v) not in found:
                context = [f'{sk}: {sv}' for sk, sv in value.items()
                           if sk != k and isinstance(sv, (str, int, float)) and len(str(sv)) <= 60][:3]
                found[str(v)] = f"{v} ({', '.join(context)})" if context else str(v)
            _scan(v, keys, found)
    elif isinstance(value, list):
        for v in value:
            _scan(v, keys, found)


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------- SQL
SCHEMA_SET = 'schema'  # a query that only reads the catalog (sqlite_master, information_schema, ...)


def schema_from_sqlite(conn):
    """{table: {columns: [...], references: [tables]}} read from a SQLite connection."""
    schema = {}
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for table in tables:
        columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        references = sorted({r[2] for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')})
        schema[table] = dict(columns=columns, references=references)
    return schema


def table_sets(schema, max_tables=3):
    """Every set of up to `max_tables` tables connected through foreign keys, smallest first."""
    edges = {t: set() for t in schema}
    for table, info in schema.items():
        for other in info.get('references', []):
            if other in edges:
                edges[table].add(other)
                edges[other].add(table)
    sets = []
    for size in range(1, max_tables + 1):
        for combo in combinations(sorted(schema), size):
            seen, todo = set(), [combo[0]]
            while todo:
                t = todo.pop()
                if t not in seen:
                    seen.add(t)
                    todo += [o for o in edges[t] if o in combo]
            if seen == set(combo):
                sets.append(frozenset(combo))
    return sets


def table_set_id(tables):
    return '+'.join(sorted(tables)) if tables else SCHEMA_SET


def tables_in_sql(sql, schema):
    """Schema tables a query reads (FROM / JOIN), case-insensitive; empty for catalog-only queries."""
    names = {t.lower(): t for t in schema}
    found = set()
    for match in re.finditer(r'\b(?:from|join)\s+([`"\[]?[\w.]+[`"\]]?)', sql, flags=re.IGNORECASE):
        name = match.group(1).strip('`"[]').split('.')[-1].lower()
        if name in names:
            found.add(names[name])
    return frozenset(found)


def _table_candidates(schema, max_tables):
    candidates = [Candidate(SCHEMA_SET, 'schema: look up the database catalog (which tables and columns exist)',
                            payload=SCHEMA_SET)]
    for tables in table_sets(schema, max_tables):
        described = '; '.join(f"{t}({', '.join(schema[t]['columns'])})" for t in sorted(tables))
        candidates.append(Candidate(table_set_id(tables), f"{' + '.join(sorted(tables))}: {described}",
                                    payload=sorted(tables)))
    return candidates


# --------------------------------------------------------------------------- states
def _result(content, max_chars):
    value = _json(content)
    if value is None:
        return content[:max_chars]
    text = json.dumps(value, ensure_ascii=False)
    return value if len(text) <= max_chars else text[:max_chars] + ' ...'


def state_from_messages(messages, id_params, max_chars=600):
    """Compact transcript plus the ID values a picker may choose from."""
    transcript, known = [], {p: {} for p in id_params}
    for m in messages:
        role, content = m['role'], m.get('content')
        if role == 'system' and content:
            transcript.append(dict(system=content[:max_chars]))
        elif role == 'user':
            transcript.append(dict(user=content[:max_chars]))
        elif role == 'assistant':
            for call in m.get('tool_calls') or []:
                transcript.append(dict(call=call['function']['name'], args=_json(call['function']['arguments'])))
            if content:
                transcript.append(dict(assistant=content[:max_chars]))
        elif role == 'tool':
            transcript.append(dict(result=_result(content, max_chars)))
        blobs = [_json(c['function']['arguments']) for c in m.get('tool_calls') or []]
        if isinstance(content, str):
            # Whole JSON contents, or JSON embedded after prose (e.g. "Customer profile: {...}").
            blobs.append(_json(content) if role == 'tool' or '{' not in content else _json(content[content.index('{'):]))
        for blob in blobs:
            for param, aliases in id_params.items():
                _scan(blob, {param, *aliases}, known[param])
    return dict(transcript=transcript, known=known)


class ChatTraceSpec(DecisionSpec):
    """A DecisionSpec derived from tool schemas; remembers how to render chat states."""

    def __init__(self, name, kinds, tools, id_params, respond, source, sql_params=None):
        super().__init__(name, kinds)
        self.tools, self.id_params, self.respond, self.source = tools, id_params, respond, source
        self.sql_params = sql_params or {}

    def save(self, path):
        """Write what is needed to rebuild this spec at serving time."""
        with open(path, 'x') as stream:
            json.dump(self.source, stream, indent=2)
            stream.write('\n')

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            return spec_from_tools(**json.load(stream))

    def state(self, messages):
        return state_from_messages(messages, self.id_params)


def spec_from_tools(name, tools, *, id_params=None, sql_params=None, respond=True, version='1'):
    """Build a DecisionSpec from tool schemas. `id_params` maps an ID parameter to the keys
    its values appear under in tool results, e.g. {'order_id': ['id']}. `sql_params` maps a
    SQL parameter to the database schema, {table: {columns, references}} (see
    `schema_from_sqlite`), optionally as {'schema': ..., 'max_tables': 3}."""
    id_params = {p: list(a) for p, a in (id_params or {}).items()}
    sql_params = {p: v if 'schema' in v else dict(schema=v) for p, v in (sql_params or {}).items()}
    source = dict(name=name, tools=tools, id_params=id_params, respond=respond, version=version)
    if sql_params:
        source['sql_params'] = sql_params
    functions = _tools(tools)
    # Candidates come from the schemas, so they are part of the spec's identity.
    identity = dict(tools=tools, id_params=id_params, respond=respond)
    if sql_params:
        identity['sql_params'] = sql_params
    version = f"{version}+{digest(identity)[:12]}"
    actions = [Candidate(n, f"{n}: {f.get('description', '')}".strip(': '), payload=n) for n, f in functions.items()]
    if respond:
        actions.append(Candidate(RESPOND, 'respond: answer the user in text', payload=RESPOND))
    kinds = [Kind('action', 'Which tool should the assistant call next?', lambda s: actions, version)]
    enums = {}
    for f in functions.values():
        for param, schema in (f.get('parameters') or {}).get('properties', {}).items():
            if 'enum' in schema:
                enums.setdefault(param, [])
                enums[param] += [str(v) for v in schema['enum'] if str(v) not in enums[param]]
    for param, values in sorted(enums.items()):
        kinds.append(Kind(f'arg.{param}', f'Which value should {param} take?',
                          lambda s, values=values: [Candidate(v, v, payload=v) for v in values], version))
    for param in sorted(id_params):
        kinds.append(Kind(f'arg.{param}', f'Which {param} should the call use?',
                          lambda s, param=param: [Candidate(v, text, payload=v) for v, text in s['known'][param].items()],
                          version))
    for param, sql in sorted(sql_params.items()):
        tables = _table_candidates(sql['schema'], sql.get('max_tables', 3))
        kinds.append(Kind(f'tables.{param}', 'Which tables should the next SQL query use?',
                          lambda s, tables=tables: tables, version))
    return ChatTraceSpec(name, kinds, functions, id_params, respond, source, sql_params)


def trace_to_episode(trace, spec, *, group_id=None, coverage=None):
    """Cut one trace into action and argument decisions labelled with what was actually done.

    Argument decisions with fewer than two candidates are skipped (nothing to choose). A
    taken value that was never seen earlier becomes NoneCorrect and shows up as a recall miss."""
    messages, decisions = trace['messages'], []
    for i, m in enumerate(messages):
        if m['role'] != 'assistant':
            continue
        calls = m.get('tool_calls') or []
        taken = [c['function']['name'] for c in calls] or ([RESPOND] if m.get('content') and spec.respond else [])
        for j, tool in enumerate(taken):
            state = spec.state(messages[:i])
            row = f"{trace['id']}:{i}:{j}"
            decisions.append(spec.decision(f'{row}:action', 'action', state, taken=tool, group_id=group_id))
            if tool == RESPOND:
                continue
            args = _json(calls[j]['function']['arguments']) or {}
            for param, value in args.items():
                if param in spec.sql_params:
                    tables = tables_in_sql(str(value), spec.sql_params[param]['schema'])
                    _count(coverage, param, 'decisions')
                    decisions.append(spec.decision(f'{row}:{param}', f'tables.{param}', dict(state, next_call=tool),
                                                   taken=sorted(tables) or SCHEMA_SET, group_id=group_id))
                    continue
                kind = f'arg.{param}'
                if kind not in spec.kinds:
                    _count(coverage, param, 'free_text_left_to_parent')
                    continue
                arg_state = dict(state, next_call=tool)
                if len(spec.enumerate(kind, arg_state)) < 2:
                    _count(coverage, param, 'skipped_fewer_than_2_candidates')
                    continue
                _count(coverage, param, 'decisions')
                decisions.append(spec.decision(f'{row}:{param}', kind, arg_state, taken=str(value), group_id=group_id))
    return Episode(trace['id'], tuple(decisions), outcome=trace.get('outcome'))


def _count(coverage, param, what):
    if coverage is not None:
        coverage.setdefault(param, {}).setdefault(what, 0)
        coverage[param][what] += 1


def episodes_from_traces(traces, spec):
    """Adapt many traces; also report, per tool parameter, how many calls became decisions."""
    coverage = {}
    episodes = [trace_to_episode(t, spec, coverage=coverage) for t in traces]
    return episodes, coverage
