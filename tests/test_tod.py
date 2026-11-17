"""TOD SDK primitives: spec-built rows, grouped splits, recall misses, confident-or-defer serving."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from openjev.decision import choice_result, load_tasks
from openjev.tod import (Candidate, Choice, Decision, DecisionSpec, Episode, Kind, NoneCorrect,
                         Preference, build_dataset, load_decisions)

TOOLS = [Candidate('lookup', 'Look up an order', {'tool': 'get_order'}),
         Candidate('cancel', 'Cancel an order', {'tool': 'cancel_order'}),
         Candidate('reply', 'Reply to the customer', {'tool': 'reply'}),
         Candidate('human', 'Transfer to a human agent', {'tool': 'transfer'})]


def orders(state):
    return [Candidate(o, f'Order {o}', {'order_id': o}) for o in state['orders']]


def spec(version='1'):
    return DecisionSpec('retail', [
        Kind('action', 'What should the agent do next?', lambda s: TOOLS, version,
             mandatory_defer=frozenset({'human'})),
        Kind('arg.order_id', 'Which order is this about?', orders, version)])


def episode(i, s=None):
    s = s or spec()
    state = {'orders': [f'W{i}', f'X{i}'], 'message': f'cancel W{i} please ({i})'}
    return Episode(f'conv-{i}', (
        s.decision(f'{i}-0', 'action', state, taken={'tool': 'cancel_order'}),
        s.decision(f'{i}-1', 'arg.order_id', {**state, 'action': 'cancel'}, choice=f'W{i}')), outcome=1.)


class Scorer:
    def __init__(self, utilities):
        self.utilities = utilities

    def score(self, task):
        return choice_result(task, [self.utilities.get(c.id, 0.) for c in task.candidates])


class TODTests(unittest.TestCase):
    def test_taken_payload_labels_or_records_recall_miss(self):
        s = spec()
        state = {'orders': ['W1']}
        self.assertEqual(s.decision('a', 'action', state, taken={'tool': 'cancel_order'}).target, Choice('cancel'))
        miss = s.decision('b', 'action', state, taken={'tool': 'refund'})
        self.assertIsInstance(miss.target, NoneCorrect)
        self.assertEqual(Decision.parse(json.loads(json.dumps(miss.to_dict()))), miss)
        with self.assertRaises(ValueError):
            miss.to_choice_task()

    def test_roundtrip_and_payload_never_reaches_v1_task(self):
        d = episode(1).decisions[1]
        parsed = Decision.parse(json.loads(json.dumps(d.to_dict())))
        self.assertEqual(parsed, d)
        task = d.to_choice_task()
        self.assertNotIn('payload', json.dumps(task.to_dict()))
        self.assertEqual(task.candidates[task.label].id, 'W1')

    def test_typed_targets_validate_ids(self):
        value = episode(1).decisions[0].to_dict()
        value['target'] = {'type': 'preference', 'better': 'cancel', 'worse': 'reply'}
        self.assertIsInstance(Decision.parse(value).target, Preference)
        value['target'] = {'type': 'choice', 'choice_id': 'nope'}
        with self.assertRaises(ValueError):
            Decision.parse(value)

    def test_spec_drift_is_rejected(self):
        d = episode(1).decisions[0]
        with self.assertRaisesRegex(ValueError, 'different DecisionSpec'):
            spec(version='2').check(d)
        stale = replace(d, candidates=d.candidates[:3])
        with self.assertRaisesRegex(ValueError, 'differ'):
            spec().check(stale)

    def test_dataset_groups_split_reports_and_writes_v1_bridge(self):
        s = spec()
        episodes = [episode(i, s) for i in range(40)]
        episodes.append(Episode('conv-miss', (s.decision('m-0', 'action', {'orders': ['Z']},
                                                         taken={'tool': 'refund'}),)))
        episodes.append(Episode('conv-failed', (s.decision('f-0', 'action', {'orders': ['Q']}, choice='reply'),),
                                outcome=0.))
        ds = build_dataset(s, episodes, min_outcome=.5)
        self.assertEqual(ds.report['filtered_episodes'], 2)  # missing and failing outcomes
        for rows in ds.splits.values():
            for d in rows:
                self.assertTrue(d.group_id.startswith('conv-'))
        placement = {d.group_id: name for name, rows in ds.splits.items() for d in rows}
        self.assertEqual(len(placement), 40)
        self.assertEqual(ds.report['recall_miss_rate'], {'action': 0., 'arg.order_id': 0.})
        ds = build_dataset(s, episodes)
        self.assertGreater(ds.report['recall_miss_rate']['action'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            ds.write(Path(tmp)/'out')
            train = load_decisions(Path(tmp)/'out'/'train.jsonl', s)
            self.assertEqual(len(train), len(ds.splits['train']))
            v1 = load_tasks(Path(tmp)/'out'/'train.choice.jsonl', labelled=True)
            self.assertEqual(len(v1), sum(isinstance(d.target, Choice) for d in train))
            self.assertEqual(json.loads((Path(tmp)/'out'/'manifest.json').read_text())['spec'], s.fingerprint)

    def test_decide_executes_confident_and_defers_otherwise(self):
        s = spec()
        state = {'orders': ['W1']}
        p = s.decide(Scorer({'cancel': 8.}), state, 'action', threshold=.9)
        self.assertTrue(p.execute)
        self.assertEqual(p.payload, {'tool': 'cancel_order'})
        p = s.decide(Scorer({'cancel': 1.}), state, 'action', threshold={'action': .9})
        self.assertEqual((p.defer, p.reason), (True, 'low_confidence'))
        p = s.decide(Scorer({'human': 20.}), state, 'action', threshold=.5)
        self.assertEqual((p.defer, p.reason), (True, 'mandatory'))


if __name__ == '__main__':
    unittest.main()


class DedupTests(unittest.TestCase):
    def test_identical_inputs_are_dropped_and_conflicts_counted(self):
        s = spec()
        state = {'orders': ['W1', 'X1'], 'message': 'same words'}
        episodes = [Episode(f'conv-{i}', (s.decision(f'{i}', 'action', state, choice=choice),))
                    for i, choice in enumerate(['cancel', 'cancel', 'reply'])]
        ds = build_dataset(s, episodes)
        self.assertEqual(sum(len(rows) for rows in ds.splits.values()), 1)
        self.assertEqual((ds.report['duplicates_dropped'], ds.report['duplicates_with_different_label']), (2, 1))


class ShopDeskTests(unittest.TestCase):
    def test_oracle_through_protocol_always_succeeds(self):
        from openjev.tod.envs.shopdesk import ShopDesk
        for seed in range(200):
            env = ShopDesk().reset(seed)
            while env.observe() is not None:
                env.defer()
            self.assertEqual(env.reward(), 1.0)

    def test_teacher_traces_never_resolve_a_closed_window(self):
        from openjev.tod.envs.shopdesk import record_trace
        for i in range(300):
            trace = record_trace(str(i), i)
            calls = [m['tool_calls'][0]['function']['name'] for m in trace['messages'] if m.get('tool_calls')]
            results = [json.loads(m['content']) for m in trace['messages'] if m['role'] == 'tool']
            if trace['outcome'] and any(r.get('return_window') == 'closed' for r in results):
                self.assertIn(calls[-1], ('send_tracking', 'cancel_order', 'transfer_to_human'))

    def test_rollout_acts_defers_and_records_steps(self):
        import numpy as np
        from openjev.tod.envs.shopdesk import ShopDesk
        from openjev.tod.rl import HeadPolicy
        shop = DecisionSpec('shop', [
            Kind('action', 'Next tool?', lambda s: [Candidate(t, t, payload=t) for t in
                 ('get_order', 'cancel_order', 'return_order', 'send_tracking', 'transfer_to_human')], '1'),
            Kind('arg.order_id', 'Which order?', lambda s: [Candidate(o['id'], o['item'], payload=o['id'])
                                                            for o in s['orders']], '1')])
        env = ShopDesk()
        policy = object.__new__(HeadPolicy)
        policy.spec = shop
        policy.heads = {k: {'temperature': 1.0} for k in shop.kinds}

        def logits(decision):  # confident oracle on actions, uniform (so deferred) on orders
            right = env.oracle()[0] if decision.kind == 'action' else None
            return None, None, np.array([9. if c.id == right else 0. for c in decision.candidates])
        policy.logits = logits
        run = policy.rollout(env, 3, threshold=0.9)
        self.assertEqual(run.reward, 1.0)
        self.assertEqual(run.decided, len(run.steps))
        self.assertGreater(run.decided, 0)
        self.assertGreater(run.deferred, 0)


class ChatTraceTests(unittest.TestCase):
    def test_spec_from_tool_schemas_and_coverage(self):
        from openjev.tod.chat_traces import episodes_from_traces, spec_from_tools
        from openjev.tod.envs.shopdesk import record_trace
        traces = [record_trace(f'c{i}', i * 2) for i in range(40)]
        bare = spec_from_tools('auto', traces[0]['tools'], id_params={'order_id': []})
        _, coverage = episodes_from_traces(traces, bare)
        self.assertNotIn('decisions', coverage['order_id'])
        aliased = spec_from_tools('auto', traces[0]['tools'], id_params={'order_id': ['id']})
        self.assertNotEqual(bare.fingerprint, aliased.fingerprint)
        episodes, coverage = episodes_from_traces(traces, aliased)
        self.assertGreater(coverage['order_id']['decisions'], 0)
        kinds = {d.kind for e in episodes for d in e.decisions}
        self.assertEqual(kinds, {'action', 'arg.order_id'})
        order = next(d for e in episodes for d in e.decisions if d.kind == 'arg.order_id')
        self.assertIn('item:', order.candidates[0].text)
        self.assertEqual(build_dataset(aliased, episodes).report['recall_miss_rate']['arg.order_id'], 0.0)


class TrainerWeightingTests(unittest.TestCase):
    def test_pooled_updates_weigh_label_and_anchor_rows_equally(self):
        import numpy as np
        from openjev.tod.rl import _Trainer
        trainer = _Trainer.__new__(_Trainer)
        trainer.rng, trainer.anchor_weight = np.random.default_rng(0), 1.0
        ctx, opt = np.zeros((4, 8), np.float16), np.zeros((3, 8), np.float16)
        anchor = type('Anchor', (), {'__len__': lambda self: 100,
                                      'batch': lambda self, idx: ('anchor', len(idx))})()
        trainer.anchors = {'action': anchor}
        captured = {}
        trainer.optimizers = {'action': type('Opt', (), {'step': lambda self, terms: captured.update(terms=terms)})()}
        batch = [(ctx, opt, 1, 1.0)] * 2
        trainer.update('action', batch, 6, pooled=True)
        self.assertEqual([t[2] for t in captured['terms']], [2 / 8, 6 / 8])
        trainer.update('action', batch, 6)
        self.assertEqual([t[2] for t in captured['terms']], [1.0, 1.0])
