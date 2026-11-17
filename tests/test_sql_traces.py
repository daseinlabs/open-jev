import sqlite3
import unittest

from openjev.tod.chat_traces import (SCHEMA_SET, episodes_from_traces, schema_from_sqlite, spec_from_tools,
                                     table_sets, tables_in_sql)
from openjev.tod.envs.sqldesk import SCHEMA, SqlDesk, record_trace
from openjev.tod.schema import Choice


class SqlTracesTests(unittest.TestCase):
    def test_tables_in_sql(self):
        self.assertEqual(tables_in_sql('SELECT SUM(o.total) FROM Orders o JOIN "customers" c ON o.customer_id = c.id', SCHEMA),
                         {'orders', 'customers'})
        self.assertEqual(tables_in_sql("SELECT name FROM sqlite_master WHERE type = 'table'", SCHEMA), set())
        self.assertEqual(tables_in_sql('select * from main.refunds', SCHEMA), {'refunds'})

    def test_table_sets_are_connected(self):
        sets = table_sets(SCHEMA, max_tables=3)
        self.assertIn(frozenset({'order_items', 'orders', 'products'}), sets)
        self.assertNotIn(frozenset({'customers', 'products'}), sets)   # no foreign-key path in two tables
        self.assertEqual(len([s for s in sets if len(s) == 1]), len(SCHEMA))

    def test_schema_from_sqlite(self):
        db = sqlite3.connect(':memory:')
        db.executescript('CREATE TABLE a (id INTEGER PRIMARY KEY, x TEXT);'
                         'CREATE TABLE b (id INTEGER, a_id INTEGER REFERENCES a(id));')
        self.assertEqual(schema_from_sqlite(db), {'a': {'columns': ['id', 'x'], 'references': []},
                                                  'b': {'columns': ['id', 'a_id'], 'references': ['a']}})

    def test_traces_become_action_and_table_decisions(self):
        traces = [record_trace(f'c{i}', i) for i in range(40)]
        spec = spec_from_tools('sql', traces[0]['tools'], sql_params={'sql': SCHEMA})
        episodes, coverage = episodes_from_traces(traces, spec)
        tables = [d for e in episodes for d in e.decisions if d.kind == 'tables.sql']
        self.assertTrue(tables and all(isinstance(d.target, Choice) for d in tables))
        self.assertIn(SCHEMA_SET, {d.target.choice_id for d in tables})
        self.assertEqual(coverage['sql']['decisions'], len(tables))
        # The spec's identity depends on the schema.
        other = dict(SCHEMA, extra=dict(columns=['id'], references=[]))
        self.assertNotEqual(spec.fingerprint, spec_from_tools('sql', traces[0]['tools'], sql_params={'sql': other}).fingerprint)

    def test_oracle_through_the_environment_always_succeeds(self):
        for seed in range(100):
            desk = SqlDesk().reset(seed)
            while desk.observe() is not None:
                desk.defer()
            self.assertEqual(desk.reward(), 1.0)

    def test_missing_join_gives_a_wrong_answer(self):
        desk = SqlDesk()
        seed = next(s for s in range(1000) if desk.reset(s).intent == 'region_revenue')
        desk.reset(seed)
        for payload in ('run_sql', ['orders'], 'respond'):
            desk.act(payload)
        self.assertEqual(desk.reward(), 0.0)


if __name__ == '__main__':
    unittest.main()
