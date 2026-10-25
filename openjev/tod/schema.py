"""Decision data contract (schema v2). Pure Python; safe to import without a backend.

A Decision is one pick: a state, a typed kind, the candidates the spec enumerated for
that state, and an optional supervision target. Rows stay flat; earlier picks in a
turn belong in the next row's state, and episodes only group rows for splitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..decision import ChoiceTask, canonical

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Candidate:
    """`text` is what the model reads; `payload` is what the caller executes (never rendered)."""
    id: str
    text: str
    payload: object = None

    def to_dict(self):
        value = dict(id=self.id, text=self.text)
        if self.payload is not None:
            value['payload'] = self.payload
        return value


# Supervision is a tagged union. A missing target means "unlabelled", never "wrong".

@dataclass(frozen=True)
class Choice:
    choice_id: str


@dataclass(frozen=True)
class Preference:
    better: str
    worse: str


@dataclass(frozen=True)
class Outcome:
    candidate_id: str
    reward: float


@dataclass(frozen=True)
class NoneCorrect:
    """The taken action was not among the enumerated candidates (a recall miss, kept visible)."""
    note: str = ''


Target = Choice | Preference | Outcome | NoneCorrect
_TARGETS = {'choice': Choice, 'preference': Preference, 'outcome': Outcome, 'none_correct': NoneCorrect}
_TAGS = {v: k for k, v in _TARGETS.items()}


def target_to_dict(target):
    return {'type': _TAGS[type(target)], **target.__dict__}


def parse_target(value, ids):
    if not isinstance(value, dict) or value.get('type') not in _TARGETS:
        raise ValueError(f'target type must be one of {sorted(_TARGETS)}')
    fields = {k: v for k, v in value.items() if k != 'type'}
    target = _TARGETS[value['type']](**fields)
    referenced = {Choice: ('choice_id',), Preference: ('better', 'worse'),
                  Outcome: ('candidate_id',)}.get(type(target), ())
    for name in referenced:
        if getattr(target, name) not in ids:
            raise ValueError(f'target {name} is not in candidates')
    if isinstance(target, Preference) and target.better == target.worse:
        raise ValueError('preference needs two different candidates')
    if isinstance(target, Outcome) and not math.isfinite(target.reward):
        raise ValueError('outcome reward must be finite')
    return target


@dataclass(frozen=True)
class Decision:
    id: str
    kind: str
    state: object
    question: str
    candidates: tuple[Candidate, ...]
    target: Target | None = None
    group_id: str | None = None
    spec: str | None = None  # fingerprint of the DecisionSpec that enumerated the candidates

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or value.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'schema_version must be {SCHEMA_VERSION}')
        allowed = {'schema_version', 'id', 'kind', 'state', 'question', 'candidates',
                   'target', 'group_id', 'spec'}
        if set(value) - allowed:
            raise ValueError(f'unknown decision fields: {sorted(set(value)-allowed)}')
        for c in value.get('candidates') or []:
            if not isinstance(c, dict) or not {'id', 'text'} <= set(c) <= {'id', 'text', 'payload'}:
                raise ValueError('each candidate must contain id, text, and optional payload')
        candidates = tuple(Candidate(**c) for c in value.get('candidates') or [])
        target = value.get('target')
        if target is not None:
            target = parse_target(target, {c.id for c in candidates})
        decision = cls(value.get('id'), value.get('kind'), value.get('state'), value.get('question'),
                       candidates, target, value.get('group_id'), value.get('spec'))
        decision.validate()
        return decision

    def validate(self):
        for name in ('id', 'kind', 'question'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty string')
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError('state must be a string, object, or array')
        canonical(self.state)
        # Reuse the v1 candidate rules (2..255, unique nonempty ids and texts).
        self.to_choice_task(require_choice=False)
        for c in self.candidates:
            canonical(c.payload)
        if self.group_id is not None and (not isinstance(self.group_id, str) or not self.group_id.strip()):
            raise ValueError('group_id must be a nonempty string')

    def to_dict(self):
        value = dict(schema_version=SCHEMA_VERSION, id=self.id, kind=self.kind, state=self.state,
                     question=self.question, candidates=[c.to_dict() for c in self.candidates])
        for name in ('group_id', 'spec'):
            if getattr(self, name) is not None:
                value[name] = getattr(self, name)
        if self.target is not None:
            value['target'] = target_to_dict(self.target)
        return value

    def to_choice_task(self, require_choice=True):
        """Bridge to the v1 trainer. Payloads and kind stay out of the rendered context."""
        if require_choice and not isinstance(self.target, Choice):
            raise ValueError('only choice targets train the v1 candidate head')
        value = dict(schema_version=1, id=self.id, state=self.state, question=self.question,
                     candidates=[dict(id=c.id, text=c.text) for c in self.candidates])
        if isinstance(self.target, Choice):
            value['target'] = {'choice_id': self.target.choice_id}
        if self.group_id is not None:
            value['group_id'] = self.group_id
        return ChoiceTask.parse(value, labelled=require_choice)

    def candidate(self, candidate_id):
        return next(c for c in self.candidates if c.id == candidate_id)


@dataclass(frozen=True)
class Episode:
    """One conversation or rollout: decisions in order, sharing a split group."""
    id: str
    decisions: tuple[Decision, ...]
    outcome: float | None = None  # e.g. task success, for filtering or weighting labels


@dataclass(frozen=True)
class Prediction:
    decision_id: str
    kind: str
    choice_id: str
    confidence: float
    probabilities: dict[str, float]
    defer: bool
    reason: str | None = None  # 'low_confidence' | 'mandatory'
    payload: object = None

    @property
    def execute(self):
        """True when the caller should act on `payload` instead of handing off to the parent."""
        return not self.defer


__all__ = ['SCHEMA_VERSION', 'Candidate', 'Choice', 'Preference', 'Outcome',
           'NoneCorrect', 'Target', 'Decision', 'Episode', 'Prediction', 'parse_target',
           'target_to_dict']
