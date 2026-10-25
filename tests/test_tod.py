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
