"""Traces -> decisions. This is the file you rewrite for your own traces.

SPEC says which decisions exist and how their candidates are enumerated. `render_state`
(shared with the live environment) says what the model sees. Training and serving both
go through these, so the model never trains on a view it will not get live.
"""
import json

from openjev.tod import Candidate, DecisionSpec, Episode, Kind
from openjev.tod.envs.shopdesk import TAKES_ORDER, TOOLS, render_state

ACTIONS = [Candidate(name, description, payload=name) for name, description in TOOLS.items()]


def order_candidates(state):
    return [Candidate(o['id'], f"Order {o['id']}: {o['item']}", payload=o['id']) for o in state['orders']]


SPEC = DecisionSpec('shopdesk', [
    Kind('action', 'Which tool should the agent call next?', lambda state: ACTIONS, version='1'),
    Kind('arg.order_id', 'Which order should this tool call use?', order_candidates, version='1'),
])


def trace_to_episode(trace):
    """One conversation -> flat decision rows, labelled with what the agent actually did."""
    messages = trace['messages']
    customer = json.loads(messages[0]['content'].split('Customer profile: ', 1)[1])
    message = next(m['content'] for m in messages if m['role'] == 'user')
    results = {m['tool_call_id']: json.loads(m['content']) for m in messages if m['role'] == 'tool'}
    history, decisions = [], []
    for m in messages:
        for call in m.get('tool_calls') or []:
            tool = call['function']['name']
            order_id = json.loads(call['function']['arguments']).get('order_id')
            row = f"{trace['id']}:{len(history)}"
            decisions.append(SPEC.decision(f'{row}:action', 'action',
                                           render_state(customer, message, history), taken=tool))
            if tool in TAKES_ORDER:
                decisions.append(SPEC.decision(f'{row}:order', 'arg.order_id',
                                               render_state(customer, message, history, next_tool=tool),
                                               taken=order_id))
            history.append((tool, order_id, results[call['id']]))
    return Episode(trace['id'], tuple(decisions), outcome=trace['outcome'])
