"""Subset targets, declarative specs, operating points, and the encoder method end to end
(a tiny BERT built locally, so no download)."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from openjev.tod import Decision, Episode, ListSpec, Subset, build_dataset, load_spec
from openjev.tod import operating

SPEC = ListSpec('filter', [dict(name='keep', question='Which passages mention the word?', candidates='passages')])
WORDS = ['apple', 'river', 'stone', 'cloud', 'tiger', 'piano', 'lemon', 'house', 'green', 'ocean', 'tower', 'paper']


def episodes(n, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        word = rng.choice(WORDS)
        passages, keep = [], []
        for j in range(rng.randint(3, 5)):
            words = rng.sample([w for w in WORDS if w != word], 3)
            if rng.random() < .35:
                words[rng.randrange(3)] = word
                keep.append(f'p{j + 1}')
            passages.append(' '.join(words) + f' {j}')
        state = dict(question=f'word {word}', passages=passages)
        d = SPEC.decision(f'src{i % 2}:{i}', 'keep', state, target=Subset(keep))
        out.append(Episode(f'e{i}', (d,)))
    return out


class SubsetSchemaTests(unittest.TestCase):
    def test_round_trip_and_empty(self):
        d = episodes(1)[0].decisions[0]
        for ids in (['p1', 'p3'], []):
            row = dict(d.to_dict(), target=dict(type='subset', ids=ids))
            parsed = Decision.parse(row)
            self.assertEqual(parsed.target, Subset(tuple(ids)))
            self.assertEqual(parsed.to_dict()['target'], dict(type='subset', ids=ids))

    def test_rejects_unknown_and_duplicate_ids(self):
        row = episodes(1)[0].decisions[0].to_dict()
        with self.assertRaises(ValueError):
            Decision.parse(dict(row, target=dict(type='subset', ids=['p9'])))
        with self.assertRaises(ValueError):
            Decision.parse(dict(row, target=dict(type='subset', ids=['p1', 'p1'])))


class ListSpecTests(unittest.TestCase):
    def test_candidates_context_and_definition(self):
        state = dict(question='q', passages=['a', dict(id='x', text='b', payload={'doc': 7})])
        candidates = SPEC.enumerate('keep', state)
        self.assertEqual([(c.id, c.text, c.payload) for c in candidates], [('p1', 'a', 0), ('x', 'b', {'doc': 7})])
        self.assertEqual(SPEC.context('keep', state), dict(question='q'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spec.json'
            path.write_text(json.dumps(SPEC.definition()))
            self.assertEqual(load_spec(path).fingerprint, SPEC.fingerprint)

    def test_dataset_carries_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(SPEC, episodes(20)).write(Path(tmp)/'d')
            self.assertEqual(load_spec(Path(tmp)/'d'/'spec.json').fingerprint, SPEC.fingerprint)


class OperatingPointTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.questions = []
        for _ in range(500):
            p = rng.random(8)
            relevant = rng.random(8) < .2
            self.questions.append((p, relevant))

    def survival(self, rule, value, survival='any'):
        hits = [(r & operating.kept(p, rule, value)).any() if survival == 'any' else
                (r & operating.kept(p, rule, value)).sum() == r.sum() for p, r in self.questions if r.any()]
        return np.mean(hits)

    def test_largest_value_reaching_target(self):
        for rule in operating.RULES:
            for survival in operating.SURVIVAL:
                value = operating.choose(self.questions, rule, .9, survival)
                self.assertGreaterEqual(self.survival(rule, value, survival), .9)
                self.assertLess(self.survival(rule, value * 1.01 + 1e-6, survival), .9 + 1e-9)

    def test_rho_keeps_the_top_candidate(self):
        self.assertEqual(operating.kept([.1, .4, .2], 'rho', 1.0).tolist(), [False, True, False])

    def test_metrics_and_top_k(self):
        questions = [dict(p=[.9, .1, .8], relevant=[False, False, True], tokens=[10, 10, 20]),
                     dict(p=[.2, .7, .1], relevant=[False, False, False], tokens=[5, 5, 5])]
        report = operating.metrics(questions, {'tau_50': ('tau', .5)})
        point = report['tau_50']
        self.assertEqual(point['answer_survival'], 1.0)
        self.assertAlmostEqual(point['tokens_kept'], 35 / 55)
        self.assertEqual(point['top_k']['k'], 3)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'needs torch and transformers')
class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_is_half_bce(self):
        import torch
        from openjev.tod.encoder import subset_loss
        logits, y = torch.tensor([2.0, -1.0, .3]), torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(subset_loss(logits, y, loss='focal', gamma=0, alpha=.5)),
                               float(subset_loss(logits, y)) / 2, places=6)


def tiny_bert(root):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '{', '}', '"', ':', 'question', 'word', 'which',
             'passages', 'mention', 'the', '?', *WORDS, *map(str, range(10))]
    root.mkdir()
    (root/'vocab.txt').write_text('\n'.join(vocab) + '\n')
    BertTokenizerFast(str(root/'vocab.txt')).save_pretrained(str(root))
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128)
    BertModel(config).save_pretrained(str(root))
    return root


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'needs torch and transformers')
class EncoderEndToEndTests(unittest.TestCase):
    def test_packed_layout(self):
        self.run_pipeline(layout=None, head=None)

    def test_pairs_layout_with_linear_head(self):
        self.run_pipeline(layout='pairs', head='linear')

    def run_pipeline(self, layout, head):
        from openjev import compute
        from openjev.tod.encoder import calibrate, evaluate, export, finetune
        from openjev.tod.rl import load_policy
        compute.select('torch', 'cpu')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tiny_bert(tmp/'bert')
            build_dataset(SPEC, episodes(240)).write(tmp/'data')
            result = finetune(tmp/'data', tmp/'model', source=base, epochs=2, batch_tokens=600, learning_rate=1e-3,
                              loss='focal', eval_every=10, checkpoint_every=10, keep_checkpoints=1, layout=layout,
                              head=head, log=lambda m: None)
            config = json.loads((tmp/'model'/'encoder.json').read_text())
            self.assertEqual((config['layout'], config['head']), (layout or 'packed', head or 'mlp'))
            self.assertIn('keep', json.loads((tmp/'model'/'calibration.json').read_text())['temperatures'])
            self.assertEqual(len(list((tmp/'model'/'checkpoints').iterdir())), 1)
            points = json.loads((tmp/'model'/'operating_points.json').read_text())
            self.assertEqual(points['default'], 'rho_99')
            self.assertEqual(set(points['kinds']['keep']), {'tau_95', 'tau_98', 'tau_99', 'rho_95', 'rho_98', 'rho_99'})
            test = result['test']['keep']
            self.assertIn('top_k', test['rho_99'])
            self.assertEqual(set(test['by_source']), {'src0', 'src1'})
            self.assertTrue(json.loads((tmp/'model'/'tod.json').read_text())['training']['history'])

            # Serving from the model alone, one decision per call or packed.
            policy, explicit = load_policy(tmp/'model'), load_policy(tmp/'model', SPEC)
            states = [d.state for e in episodes(6, seed=9) for d in e.decisions]
            single = [policy.decide(s, 'keep') for s in states]
            packed = policy.decide_many([(s, 'keep') for s in states])
            self.assertEqual([p.keep for p in single], [p.keep for p in packed])
            self.assertEqual([p.keep for p in single], [explicit.decide(s, 'keep').keep for s in states])
            self.assertTrue(all(p.keep and p.choice_id in p.keep and p.point == 'rho_99' for p in single))
            self.assertEqual(policy.decide(states[0], 'keep', threshold=0.0).keep,
                             tuple(c.id for c in SPEC.enumerate('keep', states[0])))

            # Re-choosing points, evaluation-only sets, and probabilities.
            again = calibrate(tmp/'model', tmp/'data', targets=(.9,), default_point='tau_90')
            self.assertEqual(set(again['operating_points']['kinds']['keep']), {'tau_90', 'rho_90'})
            build_dataset(SPEC, episodes(30, seed=5), fractions={'test': 1.0}).write(tmp/'external')
            report = evaluate(tmp/'model', tmp/'external', out=tmp/'ext.json', probabilities_path=tmp/'p.jsonl',
                              match_survival=0.9)
            keep = report['test']['keep']
            self.assertEqual(keep['questions'], 30)
            self.assertGreaterEqual(keep['rho_matched']['answer_survival'], 0.9)
            self.assertIn('calibration_error', keep)
            self.assertEqual(len((tmp/'p.jsonl').read_text().splitlines()), 30)

            # Export: bf16 trunk, no checkpoints, same decisions.
            shipped = export(tmp/'model', tmp/'shipped', check=16, log=lambda m: None)
            self.assertFalse((tmp/'shipped'/'checkpoints').exists())
            self.assertEqual(shipped['checked'], 16)
            self.assertLess(shipped['bytes'], sum(p.stat().st_size for p in (tmp/'model').rglob('*')
                                                  if p.is_file() and 'checkpoints' not in p.parts))
            self.assertEqual(load_policy(tmp/'shipped').decide(states[0], 'keep').choice_id is not None, True)

            # A self-contained Hugging Face folder: loads with AutoModel, same probabilities.
            hf = export(tmp/'model', tmp/'hf', check=8, hf=True, log=lambda m: None)
            self.assertTrue((tmp/'hf'/'modeling_openjev.py').exists())
            self.assertLess(hf['max_probability_difference'], 0.02)  # bf16 trunk: a boundary keep may flip
            self.assertGreaterEqual(hf['same_decisions'], 6)

            # Continued fine-tuning starts from the trained trunk and head.
            more = finetune(tmp/'data', tmp/'more', source=tmp/'model', max_steps=3, learning_rate=1e-5,
                            batch_size=4, log=lambda m: None)
            binding = json.loads((tmp/'more'/'tod.json').read_text())
            self.assertEqual(binding['fine_tuned_from'], str(tmp/'model'))
            self.assertIn('operating_points', more['calibration'])


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'needs torch and transformers')
class PackingTests(unittest.TestCase):
    def test_scores_ignore_siblings_and_order(self):
        from openjev import compute
        from openjev.tod.encoder import EncoderModel
        compute.select('torch', 'cpu')
        with tempfile.TemporaryDirectory() as tmp:
            model = EncoderModel(tiny_bert(Path(tmp)/'bert'))
            self.assertEqual(model.layout, 'packed')
            state = dict(question='word apple', passages=['apple river stone', 'cloud tiger', 'piano lemon house'])
            full = SPEC.decision('a', 'keep', state)
            fewer = SPEC.decision('b', 'keep', dict(state, passages=state['passages'][:2]))
            reordered = SPEC.decision('c', 'keep', dict(state, passages=state['passages'][::-1]))
            a, b, c = model.score([full, fewer, reordered], SPEC)
            self.assertAlmostEqual(a[0], b[0], places=4)
            np.testing.assert_allclose(a, c[::-1], atol=1e-4)


class StepPlanTests(unittest.TestCase):
    def test_token_budget_and_rank_balance(self):
        from openjev.tod.encoder import plan_steps
        lengths = np.random.default_rng(0).integers(40, 1000, size=400)
        steps = plan_steps(list(range(400)), 2, batch_tokens=4000, lengths=lengths)
        for ranks in steps:
            totals = [lengths[r].sum() for r in ranks]
            self.assertGreaterEqual(sum(totals), 8000)
            self.assertLess(sum(totals), 8000 + 1000)
            self.assertLess(abs(totals[0] - totals[1]), 1000)
        by_count = plan_steps(list(range(10)), 2, batch_size=2)
        self.assertEqual(by_count[0], [[0, 2], [1, 3]])

    def test_subset_temperature(self):
        from openjev.tod.encoder import fit_subset_temperature
        rng = np.random.default_rng(1)
        z = rng.normal(0, 4, size=20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-z / 2))).astype(np.float32)
        self.assertAlmostEqual(fit_subset_temperature([z], [y]), 2.0, delta=0.15)


if __name__ == '__main__':
    unittest.main()
