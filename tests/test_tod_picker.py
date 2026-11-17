"""TOD fine-tuning: the prompt must stay byte-identical to tod.eval.letter_logit (cygnet.v1)."""
import importlib.util
import unittest

from openjev.tod import Candidate, DecisionSpec, Kind
from openjev.tod.picker import build_user_message, record

SPEC = DecisionSpec('routing', [Kind('team', 'Which team should handle this ticket?', lambda s: [
    Candidate('billing', 'Billing: charges, invoices'), Candidate('shipping', 'Delivery and tracking'),
    Candidate('account', 'account')], version='1')])
DECISION = SPEC.decision('t1', 'team', {'message': 'I was charged twice'}, choice='billing')

EXPECTED = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question about it, "
    "and a fixed list of lettered options. You reply with that option's LETTER and nothing else - a single "
    "character, no words, no punctuation, no explanation.\n\n"
    '{\n "message": "I was charged twice"\n}\n\n'
    "Which team should handle this ticket?\n\n"
    "Options:\nA. Billing: charges, invoices\nB. shipping: Delivery and tracking\nC. account\n\n"
    "Answer with the letter of exactly one option, and nothing else:")


class PromptTests(unittest.TestCase):
    def test_record_and_prompt(self):
        rec = record(DECISION)
        self.assertEqual(rec['labels'], ['billing', 'shipping', 'account'])
        self.assertEqual(rec['question']['criteria']['shipping'], 'Delivery and tracking')
        self.assertEqual(build_user_message(rec), EXPECTED)

    @unittest.skipUnless(importlib.util.find_spec('tod'), 'tod (github.com/daseinlabs/tod) not installed')
    def test_matches_tod(self):
        from tod.eval.letter_logit import build_user_message as tod_message
        self.assertEqual(build_user_message(record(DECISION)), tod_message(record(DECISION)))


if __name__ == '__main__':
    unittest.main()
