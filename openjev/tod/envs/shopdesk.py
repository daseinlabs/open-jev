"""ShopDesk: a tiny simulated support desk that stands in for "your production agent".

It produces two things a real deployment would give you:
- traces: OpenAI-style chat logs (system / user / assistant tool calls / tool results),
  written by a scripted, imperfect teacher agent, each with an outcome;
- an environment: the same desk, steppable by any policy, with a success reward.

Rules the agent has to learn (status and return window are only visible after `get_order`):

    intent  \\ status   pending          shipped             delivered (window open / closed)
    cancel            cancel_order     transfer_to_human   return_order / transfer_to_human
    return            cancel_order     transfer_to_human   return_order / transfer_to_human
    track             send_tracking    send_tracking       send_tracking

The teacher has a systematic blind spot: it ignores the return window and starts returns
that are no longer allowed. Those conversations always fail, so successful traces never
show what to do with a closed window. Supervised fine-tuning cannot learn it; RL against
the desk's reward can.
"""
from __future__ import annotations

import json
import random

ITEMS = ['blue kettle', 'desk lamp', 'running shoes', 'yoga mat', 'coffee grinder', 'wool scarf',
         'phone case', 'water bottle', 'office chair', 'backpack', 'headphones', 'cast-iron pan']
STATUSES = ['pending', 'shipped', 'delivered']
NAMES = ['Ana', 'Ben', 'Chen', 'Dana', 'Eli', 'Fatima', 'Gus', 'Hana', 'Ivan', 'Jo', 'Kofi', 'Lena']
MESSAGES = {
    'cancel': ["Please cancel order {id}.", "I need to cancel my {item} order.",
               "Changed my mind about the {item}, can you cancel it?", "Cancel {id} please, I ordered it by mistake."],
    'return': ["I want to return the {item}.", "Order {id} isn't what I expected, I'd like to send it back.",
               "How do I return my {item}? I don't need it anymore.", "Please start a return for {id}."],
    'track': ["Where is my {item}?", "Any update on order {id}?", "Can you tell me when the {item} arrives?",
              "I'm waiting on {id}, what's the status?"],
}
TOOLS = {
    'get_order': 'Look up an order to see its status',
    'cancel_order': 'Cancel an order that has not shipped yet',
    'return_order': 'Start a return for a delivered order',
    'send_tracking': 'Send the customer tracking details for an order',
    'transfer_to_human': 'Transfer the conversation to a human agent',
}
TAKES_ORDER = {'get_order', 'cancel_order', 'return_order', 'send_tracking'}
TERMINAL = set(TOOLS) - {'get_order'}
MAX_CALLS = 4
SYSTEM = 'You are ShopDesk, a retail support agent. Look orders up before acting on them.'


def correct_action(intent, order):
    if intent == 'track':
        return 'send_tracking'
    if order['status'] == 'delivered' and order['return_window'] == 'closed':
        return 'transfer_to_human'
    return {'pending': 'cancel_order', 'shipped': 'transfer_to_human', 'delivered': 'return_order'}[order['status']]


def render_state(customer, message, history, next_tool=None):
    """What a decision model sees. `history` is [(tool, order_id, result)].

    Your adapter (traces -> decisions) and your serving code must build the state with the
    same function, so the model is never trained on a view it will not get live."""
    state = dict(customer_message=message, orders=customer['orders'],
                 tool_calls=[dict(tool=t, order_id=o, result=r) for t, o, r in history])
    if next_tool is not None:
        state['next_tool'] = next_tool
    return state


class ShopDesk:
    """One support conversation, playable through the `openjev.tod.rl.Environment` protocol.

    `observe()` returns the next decision as (kind, state): first `action` (which tool),
    then `arg.order_id` if that tool takes an order. `act(payload)` answers it, `defer()`
    lets the parent (an oracle here) answer instead, and `reward()` is 1.0 on success.
    """

    def reset(self, seed):
        rng = random.Random(seed)
        items = rng.sample(ITEMS, rng.randint(2, 3))
        self.orders = [dict(id=f'W{rng.randint(1000, 9999)}', item=item, status=rng.choice(STATUSES))
                       for item in items]
        for o in self.orders:
            if o['status'] == 'delivered':
                o['return_window'] = rng.choice(['open', 'closed'])
        self.target = rng.choice(self.orders)
        self.intent = rng.choice(list(MESSAGES))
        self.customer = dict(name=rng.choice(NAMES), orders=[dict(id=o['id'], item=o['item']) for o in self.orders])
        self.message = rng.choice(MESSAGES[self.intent]).format(**self.target)
        self.history, self.pending, self.done, self.success = [], None, False, False
        return self

    # --- Environment protocol -------------------------------------------------------
    def observe(self):
        if self.done:
            return None
        if self.pending is not None:
            return 'arg.order_id', render_state(self.customer, self.message, self.history, self.pending)
        return 'action', render_state(self.customer, self.message, self.history)

    def act(self, payload):
        if self.pending is None and payload in TAKES_ORDER:
            self.pending = payload
        elif self.pending is None:
            self.call(payload)
        else:
            tool, self.pending = self.pending, None
            self.call(tool, payload)

    def defer(self):
        if self.pending is not None:
            tool, self.pending = self.pending, None
            self.call(tool, self.target['id'])
        else:
            self.call(*self.oracle())

    def reward(self):
        return float(self.success)

    # --- the desk itself ------------------------------------------------------------
    def call(self, tool, order_id=None):
        """Execute one tool call and return the result the agent sees."""
        if self.done:
            raise ValueError('episode is over')
        order = next((o for o in self.orders if o['id'] == order_id), None)
        if tool in TAKES_ORDER and order is None:
            result = {'error': f'unknown order {order_id}'}
        elif tool == 'get_order':
            result = dict(order)
        else:
            result = {'ok': True}
        self.history.append((tool, order_id, result))
        if tool in TERMINAL:
            expected = correct_action(self.intent, self.target)
            self.success = tool == expected and (tool == 'transfer_to_human' or order_id == self.target['id'])
            self.done = True
        elif len(self.history) >= MAX_CALLS:
            self.done = True
        return result

    def oracle(self):
        """The right next call: look the requested order up, then resolve it."""
        if not any(t == 'get_order' and o == self.target['id'] for t, o, _ in self.history):
            return 'get_order', self.target['id']
        action = correct_action(self.intent, self.target)
        return action, None if action == 'transfer_to_human' else self.target['id']


def teacher(desk, rng):
    """Planned tool calls of a plausible but imperfect production agent: right most of the time, with real failure modes."""
    roll = rng.random()
    target = desk.target
    if roll < .15:      # skips the lookup and acts on the literal request
        default = {'cancel': 'cancel_order', 'return': 'return_order', 'track': 'send_tracking'}[desk.intent]
        return [(default, target['id'])]
    if roll < .25:      # looks up the wrong order and acts on it
        wrong = rng.choice([o for o in desk.orders if o is not target])
        action = correct_action(desk.intent, {**wrong, 'return_window': 'open'})
        return [('get_order', wrong['id']), (action, None if action == 'transfer_to_human' else wrong['id'])]
    if roll < .30:      # gives up straight away
        return [('transfer_to_human', None)]
    action = correct_action(desk.intent, {**target, 'return_window': 'open'})  # the blind spot
    return [('get_order', target['id']), (action, None if action == 'transfer_to_human' else target['id'])]


def tool_schemas():
    schema = []
    for name, description in TOOLS.items():
        params = {'type': 'object', 'properties': {}, 'required': []}
        if name in TAKES_ORDER:
            params = {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}
        schema.append({'type': 'function', 'function': {'name': name, 'description': description, 'parameters': params}})
    return schema


def record_trace(trace_id, seed):
    """Run the teacher on one conversation and log it the way a chat API would."""
    desk = ShopDesk().reset(seed)
    messages = [{'role': 'system', 'content': SYSTEM + '\nCustomer profile: ' + json.dumps(desk.customer)},
                {'role': 'user', 'content': desk.message}]
    for i, (tool, order_id) in enumerate(teacher(desk, random.Random(seed + 1))):
        args = {} if order_id is None else {'order_id': order_id}
        call_id = f'call_{i}'
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': tool, 'arguments': json.dumps(args)}}]})
        result = desk.call(tool, order_id)
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(result)})
        if desk.done:
            break
    messages.append({'role': 'assistant', 'content': 'Thanks, is there anything else I can help with?'})
    return dict(id=trace_id, tools=tool_schemas(), messages=messages, outcome=float(desk.success))
