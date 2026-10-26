"""Generic adapter: OpenAI-style tool-calling traces -> decisions, with no hand-written spec.

Expected trace shape (one JSON object per line):

    {"id": "...", "tools": [{"type": "function", "function": {"name", "description", "parameters"}}],
     "messages": [{"role": "system"|"user"|"assistant"|"tool", ...}], "outcome": 1.0}

Decisions derived from the tool schemas:
- `action`: which tool to call next, plus `respond` (answer in text) when enabled;
- `arg.<param>`: one kind per parameter that can be picked rather than written:
  enum parameters (candidates = the enum), and ID-like parameters you name in `id_params`
  (candidates = values seen earlier in the conversation under that key or its aliases).

Free-text parameters are not decisions; calls that need them stay with the parent model.
`state_from_messages` renders the model's view and must also be used when serving.
"""
from __future__ import annotations

import json

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
            transcript.append(dict(result=_json(content) if _json(content) is not None else content[:max_chars]))
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

    def __init__(self, name, kinds, tools, id_params, respond, source):
        super().__init__(name, kinds)
        self.tools, self.id_params, self.respond, self.source = tools, id_params, respond, source

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


def spec_from_tools(name, tools, *, id_params=None, respond=True, version='1'):
    """Build a DecisionSpec from tool schemas. `id_params` maps an ID parameter to the keys
    its values appear under in tool results, e.g. {'order_id': ['id']}."""
    id_params = {p: list(a) for p, a in (id_params or {}).items()}
    source = dict(name=name, tools=tools, id_params=id_params, respond=respond, version=version)
    functions = _tools(tools)
    # Candidates come from the schemas, so they are part of the spec's identity.
    version = f"{version}+{digest(dict(tools=tools, id_params=id_params, respond=respond))[:12]}"
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
    return ChatTraceSpec(name, kinds, functions, id_params, respond, source)


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
