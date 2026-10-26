"""tickets.csv -> decision dataset. The only code you write for your own data looks like this."""
import csv
import json

from openjev.tod import Candidate, DecisionSpec, Episode, Kind, build_dataset

TEAMS = {'billing': 'Billing: charges, invoices, refunds of fees',
         'shipping': 'Shipping: delivery, tracking, addresses',
         'returns': 'Returns: broken items, exchanges, send-backs',
         'account': 'Account: login, email, deletion',
         'technical': 'Technical: app or website errors'}
PRIORITIES = {'urgent': 'Urgent: handle today', 'normal': 'Normal', 'low': 'Low: no rush'}


def options(table):
    return lambda state: [Candidate(id, text, payload=id) for id, text in table.items()]


# One kind per decision. Bump `version` whenever a candidate list changes.
SPEC = DecisionSpec('ticket-routing', [
    Kind('team', 'Which team should handle this ticket?', options(TEAMS), version='1'),
    Kind('priority', 'How urgent is this ticket?', options(PRIORITIES), version='1'),
])

if __name__ == '__main__':
    episodes = []
    with open('tickets.csv') as stream:
        for row in csv.DictReader(stream):
            state = {'message': row['message']}
            episodes.append(Episode(row['ticket_id'], (
                SPEC.decision(f"{row['ticket_id']}:team", 'team', state, choice=row['team'],
                              group_id=row['customer_id']),
                SPEC.decision(f"{row['ticket_id']}:priority", 'priority', state, choice=row['priority'],
                              group_id=row['customer_id']))))
    dataset = build_dataset(SPEC, episodes)
    dataset.write('runs/tickets')
    print(json.dumps(dataset.report['splits'], indent=2))
