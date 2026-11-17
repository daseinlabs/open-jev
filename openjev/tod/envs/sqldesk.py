"""SqlDesk: a simulated analytics assistant that answers questions by running SQL on a shop database.

A user asks a question ("What was total revenue from West customers in 2026-03?"). The
assistant looks around (lists tables, peeks at rows), runs the queries the question needs,
sometimes double-checks, and answers: typically 3 to 6 `run_sql` calls per conversation.
Traces are OpenAI tool-calling logs with an outcome (1 when the answer is right).

The **teacher** that writes the traces is a plausible production assistant with real failure
modes: it sometimes forgets a join (querying `orders` alone for a per-region question) and
sometimes answers before running the query that answers the question.

`SqlDesk` replays the same desk as an `Environment` for a decision model. The model takes the
decisions around the SQL, the same two kinds `chat_traces` derives from the traces:
  action      run another query (`run_sql`) or answer now (`respond`)
  tables.sql  which tables the next query should use
and a simulated LLM writes the SQL for the chosen tables: correct when they cover what the
step needs, a plausible wrong query when a needed table is missing, a peek otherwise.
"""
from __future__ import annotations

import json
import random
import sqlite3
from functools import lru_cache

from ..chat_traces import SCHEMA_SET, state_from_messages, table_set_id

SCHEMA = {
    'customers': dict(columns=['id', 'name', 'region', 'signup_date'], references=[]),
    'orders': dict(columns=['id', 'customer_id', 'order_date', 'total', 'status'], references=['customers']),
    'order_items': dict(columns=['order_id', 'product_id', 'qty', 'price'], references=['orders', 'products']),
    'products': dict(columns=['id', 'name', 'category', 'price'], references=[]),
    'refunds': dict(columns=['id', 'order_id', 'amount', 'reason', 'refund_date'], references=['orders']),
}
REGIONS = ['North', 'South', 'East', 'West']
CATEGORIES = {'audio': ['earbuds', 'speaker', 'headphones', 'soundbar'], 'kitchen': ['kettle', 'blender', 'toaster', 'coffee grinder'],
              'outdoor': ['tent', 'backpack', 'camp stove', 'sleeping bag'], 'home': ['lamp', 'rug', 'desk chair', 'shelf'],
              'fitness': ['yoga mat', 'dumbbells', 'jump rope', 'foam roller']}
REASONS = ['damaged', 'late delivery', 'wrong item', 'changed mind', 'defective']
MONTHS = [f'2026-0{m}' for m in range(1, 7)]
FIRST = ['Ana', 'Ben', 'Chen', 'Dara', 'Eli', 'Femi', 'Gita', 'Hugo', 'Iris', 'Jon', 'Kemi', 'Lars', 'Mina', 'Noor', 'Omar', 'Pia']
LAST = ['Silva', 'Okafor', 'Novak', 'Kim', 'Haddad', 'Berg', 'Rossi', 'Tanaka', 'Moreau', 'Patel']
SYSTEM = ('You are an analytics assistant for a shop. Use run_sql to query the SQLite database, '
          'then answer the question with the number or name.')
MAX_QUERIES = 7


def tool_schemas():
    return [{'type': 'function', 'function': {
        'name': 'run_sql', 'description': 'Run a read-only SQLite query and return the first rows',
        'parameters': {'type': 'object', 'properties': {'sql': {'type': 'string', 'description': 'SQLite SELECT statement'}},
                       'required': ['sql']}}}]


@lru_cache(maxsize=1)
def database():
    """The shop database: the same contents for every conversation (seeded)."""
    rng = random.Random(0)
    db = sqlite3.connect(':memory:', check_same_thread=False)
    db.executescript('''
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT, signup_date TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id),
                             order_date TEXT, total REAL, status TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id),
                                  qty INTEGER, price REAL);
        CREATE TABLE refunds (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), amount REAL,
                              reason TEXT, refund_date TEXT);''')
    names = rng.sample([f'{f} {l}' for f in FIRST for l in LAST], 90)
    db.executemany('INSERT INTO customers VALUES (?, ?, ?, ?)',
                   [(i + 1, n, rng.choice(REGIONS), f'2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}')
                    for i, n in enumerate(names)])
    products = [(i + 1, name, cat, round(rng.uniform(8, 240), 2))
                for i, (cat, name) in enumerate((c, n) for c, items in CATEGORIES.items() for n in items)]
    db.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', products)
    refund_id = 0
    for order_id in range(1, 1201):
        lines = [(order_id, p[0], rng.randint(1, 3), p[3]) for p in rng.sample(products, rng.randint(1, 3))]
        total = round(sum(q * price for _, _, q, price in lines), 2)
        date = f'{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}'
        db.execute('INSERT INTO orders VALUES (?, ?, ?, ?, ?)',
                   (order_id, rng.randint(1, 90), date, total, rng.choice(['delivered'] * 4 + ['shipped', 'pending'])))
        db.executemany('INSERT INTO order_items VALUES (?, ?, ?, ?)', lines)
        if rng.random() < .15:
            refund_id += 1
            db.execute('INSERT INTO refunds VALUES (?, ?, ?, ?, ?)',
                       (refund_id, order_id, round(total * rng.choice([.25, .5, 1.0]), 2), rng.choice(REASONS),
                        f'{date[:7]}-{min(28, int(date[8:]) + rng.randint(0, 3)):02d}'))
    db.commit()
    return db


def run(sql):
    """Execute a query the way the assistant's tool does: first 5 rows, or the error."""
    try:
        cursor = database().execute(sql)
        rows = cursor.fetchall()
        return dict(columns=[d[0] for d in cursor.description or []], rows=[list(r) for r in rows[:5]],
                    row_count=len(rows))
    except sqlite3.Error as exc:
        return dict(error=str(exc))


# ------------------------------------------------------------------- questions
# Each intent: a question, how to sample its parameters, and its plan: the queries that answer
# it, in order. A step is (tables it needs, correct SQL, wrong SQL an assistant writes when it
# leaves out a needed table, or None). SQL builders get the parameters and the previous answer.
MONTH = "strftime('%Y-%m', {col}) = '{month}'"
INTENTS = {
    'customer_orders': dict(
        question='How many orders has {name} placed since {date}?',
        steps=[(('customers',), lambda p, r: f"SELECT id FROM customers WHERE name = '{p['name']}'", None),
               (('orders',), lambda p, r: f"SELECT COUNT(*) FROM orders WHERE customer_id = {r} AND order_date >= '{p['date']}'",
                None)]),
    'customer_refunds': dict(
        question='How much has {name} been refunded in total?',
        steps=[(('customers',), lambda p, r: f"SELECT id FROM customers WHERE name = '{p['name']}'", None),
               (('orders', 'refunds'), lambda p, r: 'SELECT ROUND(COALESCE(SUM(r.amount), 0), 2) FROM refunds r '
                f'JOIN orders o ON r.order_id = o.id WHERE o.customer_id = {r}',
                (('refunds',), lambda p, r: f'SELECT ROUND(COALESCE(SUM(amount), 0), 2) FROM refunds WHERE order_id = {r}'))]),
    'region_revenue': dict(
        question='What was total revenue from {region} customers in {month}?',
        steps=[(('customers', 'orders'), lambda p, r: 'SELECT ROUND(SUM(o.total), 2) FROM orders o JOIN customers c '
                f"ON o.customer_id = c.id WHERE c.region = '{p['region']}' AND " + MONTH.format(col='o.order_date', month=p['month']),
                (('orders',), lambda p, r: 'SELECT ROUND(SUM(total), 2) FROM orders WHERE ' + MONTH.format(col='order_date', month=p['month'])))]),
    'region_average': dict(
        question='What is the average order value for {region} customers?',
        steps=[(('customers', 'orders'), lambda p, r: 'SELECT ROUND(AVG(o.total), 2) FROM orders o JOIN customers c '
                f"ON o.customer_id = c.id WHERE c.region = '{p['region']}'",
                (('orders',), lambda p, r: 'SELECT ROUND(AVG(total), 2) FROM orders'))]),
    'top_category': dict(
        question='Which product category sold the most units in {month}?',
        steps=[(('order_items', 'orders', 'products'), lambda p, r: 'SELECT p.category FROM order_items i '
                'JOIN orders o ON i.order_id = o.id JOIN products p ON i.product_id = p.id WHERE '
                + MONTH.format(col='o.order_date', month=p['month']) + ' GROUP BY p.category ORDER BY SUM(i.qty) DESC LIMIT 1',
                (('order_items', 'products'), lambda p, r: 'SELECT p.category FROM order_items i JOIN products p '
                 'ON i.product_id = p.id GROUP BY p.category ORDER BY SUM(i.qty) DESC LIMIT 1'))]),
    'product_units': dict(
        question='How many {product} units were sold in {month}?',
        steps=[(('products',), lambda p, r: f"SELECT id FROM products WHERE name = '{p['product']}'", None),
               (('order_items', 'orders'), lambda p, r: 'SELECT COALESCE(SUM(i.qty), 0) FROM order_items i JOIN orders o '
                f'ON i.order_id = o.id WHERE i.product_id = {r} AND ' + MONTH.format(col='o.order_date', month=p['month']),
                (('order_items',), lambda p, r: f'SELECT COALESCE(SUM(qty), 0) FROM order_items WHERE product_id = {r}'))]),
    'refund_reason': dict(
        question='What was the most common refund reason in {month}?',
        steps=[(('refunds',), lambda p, r: 'SELECT reason FROM refunds WHERE ' + MONTH.format(col='refund_date', month=p['month'])
                + ' GROUP BY reason ORDER BY COUNT(*) DESC, reason LIMIT 1', None)]),
}


def _sample(intent, rng):
    names = [r[0] for r in database().execute('SELECT name FROM customers ORDER BY id')]
    products = [n for items in CATEGORIES.values() for n in items]
    return dict(name=rng.choice(names), region=rng.choice(REGIONS), month=rng.choice(MONTHS),
                product=rng.choice(products), date=f'{rng.choice(MONTHS)}-01')


def _first(result):
    rows = result.get('rows') or []
    return rows[0][0] if rows and rows[0] else None


def _peek(tables):
    return f'SELECT * FROM {sorted(tables)[0]} LIMIT 3' if tables else "SELECT name FROM sqlite_master WHERE type = 'table'"


class SqlDesk:
    """One conversation: the user's question, the database, and the assistant's messages so far."""

    def reset(self, seed):
        rng = random.Random(seed)
        self.intent = rng.choice(list(INTENTS))
        self.params = _sample(self.intent, rng)
        spec = INTENTS[self.intent]
        self.question = spec['question'].format(**self.params)
        self.steps = spec['steps']
        self.messages = [dict(role='system', content=SYSTEM), dict(role='user', content=self.question)]
        self.progress, self.last, self.answer = 0, None, None   # last = result of the latest correct step
        self.queries, self.pending, self.done = 0, False, False
        self.expected = self._expected()
        return self

    def _expected(self):
        r = None
        for _, sql, _ in self.steps:
            r = _first(run(sql(self.params, r)))
        return r

    # The assistant's moves (shared by the teacher and the environment).
    def query(self, sql):
        call = dict(id=f'call_{self.queries}', type='function',
                    function=dict(name='run_sql', arguments=json.dumps(dict(sql=sql))))
        result = run(sql)
        self.messages += [dict(role='assistant', content=None, tool_calls=[call]),
                          dict(role='tool', tool_call_id=call['id'], content=json.dumps(result))]
        self.queries += 1
        if self.queries >= MAX_QUERIES:
            self.done = True
        return result

    def respond(self, value):
        self.answer = value
        self.messages.append(dict(role='assistant', content=f'The answer is {value}.'))
        self.done = True

    @property
    def success(self):
        return self.answer is not None and self.answer == self.expected

    # Environment protocol, for a decision model.
    def observe(self):
        if self.done:
            return None
        state = state_from_messages(self.messages, {})
        return ('tables.sql', dict(state, next_call='run_sql')) if self.pending else ('action', state)

    def act(self, payload):
        if not self.pending:
            if payload == 'respond':
                self.respond(self.last)
            else:
                self.pending = True
            return
        self.pending = False
        chosen = frozenset(payload) if payload != SCHEMA_SET else frozenset()
        if self.progress >= len(self.steps):
            self.query(_peek(chosen))   # nothing left to compute: a check
            return
        tables, sql, wrong = self.steps[self.progress]
        if chosen and chosen >= set(tables):
            self.last = _first(self.query(sql(self.params, self.last)))
            self.progress += 1
        elif wrong is not None and chosen == set(wrong[0]):
            self.query(wrong[1](self.params, self.last))   # a plausible query that misses a filter
        else:
            self.query(_peek(chosen))   # looks around without progress

    def defer(self):
        """The parent (here an oracle) takes the decision."""
        if self.pending:
            self.act(sorted(self.steps[self.progress][0]) if self.progress < len(self.steps) else SCHEMA_SET)
        else:
            self.act('run_sql' if self.progress < len(self.steps) else 'respond')

    def reward(self):
        return 1.0 if self.success else 0.0


def teacher(desk, rng):
    """A plausible production assistant: looks around first, runs the plan, sometimes checks,
    then answers. It forgets a join 12% of the time and stops a query early 6% of the time."""
    if rng.random() < .55:
        desk.query(_peek(()))
    for t in sorted(set(desk.steps[0][0])):
        if rng.random() < .5:
            desk.query(_peek({t}))
    last = None
    for i, (tables, sql, wrong) in enumerate(desk.steps):
        if i == len(desk.steps) - 1 and i > 0 and rng.random() < .06:
            return desk.respond(last)                       # answers before the final query
        if wrong is not None and rng.random() < .12:
            last = _first(desk.query(wrong[1](desk.params, last)))   # forgets a join or filter
        else:
            last = _first(desk.query(sql(desk.params, last)))
    if rng.random() < .45 and not desk.done:
        desk.query(f'SELECT COUNT(*) FROM {sorted(desk.steps[-1][0])[0]}')   # sanity check
    desk.respond(last)


def record_trace(trace_id, seed):
    desk = SqlDesk().reset(seed)
    teacher(desk, random.Random(seed + 1))
    return dict(id=trace_id, tools=tool_schemas(), messages=desk.messages, outcome=1.0 if desk.success else 0.0,
                intent=desk.intent)


__all__ = ['SCHEMA', 'SqlDesk', 'database', 'record_trace', 'run', 'teacher', 'tool_schemas', 'table_set_id']
