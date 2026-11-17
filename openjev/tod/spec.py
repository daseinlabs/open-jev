"""DecisionSpec: the user-supplied definition of which decisions exist and how candidates
are enumerated. The same spec builds training rows and serves predictions, and its
fingerprint travels with datasets and artifacts so the two cannot silently drift.

A spec that can describe itself as data (`ListSpec`, chat-trace specs) is also written to
`spec.json` beside the rows and copied into every model trained on them, so
`load_policy(MODEL)` serves without importing the code that defined it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Callable, Iterable

from ..decision import RECIPE, canonical, check_disjoint, digest, provenance
from .schema import Candidate, Choice, Decision, Episode, NoneCorrect, Prediction, Target

CandidateSource = Callable[[object], Iterable[Candidate]]
SPLITS = {'train': .7, 'validation': .1, 'calibration': .1, 'test': .1}


@dataclass(frozen=True)
class Kind:
    name: str
    question: str
    candidates: CandidateSource
    version: str  # bump whenever the candidate source's behaviour changes
    mandatory_defer: frozenset[str] = frozenset()  # candidate ids the model must never execute

    def identity(self):
        return dict(question=self.question, version=self.version, mandatory_defer=sorted(self.mandatory_defer))


class DecisionSpec:
    def __init__(self, name: str, kinds: Iterable[Kind], renderer: str = RECIPE['context_rendering']):
        self.name, self.renderer = name, renderer
        self.kinds = {}
        for kind in kinds:
            if kind.name in self.kinds:
                raise ValueError(f'duplicate kind {kind.name!r}')
            self.kinds[kind.name] = kind
        if not self.kinds:
            raise ValueError('a spec needs at least one kind')

    @property
    def fingerprint(self):
        return digest(dict(name=self.name, renderer=self.renderer,
                           kinds={k: v.identity() for k, v in self.kinds.items()}))

    def require(self, fingerprint):
        if fingerprint != self.fingerprint:
            raise ValueError('data or artifact was built with a different DecisionSpec; rebuild it')

    def definition(self) -> dict | None:
        """The spec as JSON for `load_spec`, or None when it is defined by code."""
        return None

    def context(self, kind, state):
        """The part of the state that every candidate is read against (encoder models)."""
        return state

    def kind(self, name):
        if name not in self.kinds:
            raise ValueError(f'unknown decision kind {name!r}')
        return self.kinds[name]

    def enumerate(self, kind, state):
        candidates = tuple(self.kind(kind).candidates(state))
        if not all(isinstance(c, Candidate) for c in candidates):
            raise ValueError(f'{kind}: candidate source must yield Candidate objects')
        return candidates

    def decision(self, id, kind, state, *, choice=None, taken=None, target: Target | None = None,
                 group_id=None):
        """Build a row through the spec. Label with `choice` (a candidate id), `taken` (the
        payload the teacher executed), or an explicit `target`. A taken payload missing from
        the enumerated candidates becomes NoneCorrect so recall misses stay countable."""
        if sum(x is not None for x in (choice, taken, target)) > 1:
            raise ValueError('pass at most one of choice, taken, target')
        candidates = self.enumerate(kind, state)
        if choice is not None:
            target = Choice(choice)
        elif taken is not None:
            key = canonical(taken)
            matches = [c.id for c in candidates if c.payload is not None and canonical(c.payload) == key]
            if len(matches) > 1:
                raise ValueError(f'{id}: taken payload matches several candidates')
            target = Choice(matches[0]) if matches else NoneCorrect(key)
        decision = Decision(id, kind, state, self.kind(kind).question, candidates, target, group_id,
                            self.fingerprint)
        decision.validate()
        if isinstance(target, Choice) and target.choice_id not in {c.id for c in candidates}:
            raise ValueError(f'{id}: choice {target.choice_id!r} is not in candidates')
        return decision

    def check(self, decision: Decision):
        """Reject rows that this spec would not produce today (stale or hand-edited data)."""
        self.require(decision.spec)
        kind = self.kind(decision.kind)
        if decision.question != kind.question:
            raise ValueError(f'{decision.id}: question differs from the spec')
        if self.enumerate(decision.kind, decision.state) != decision.candidates:
            raise ValueError(f'{decision.id}: candidates differ from what the spec enumerates now')

    def decide(self, scorer, state, kind, threshold: float | dict[str, float], id='request'):
        """Serve one pick. `scorer.score(ChoiceTask)` returns choice_result-shaped output,
        e.g. openjev.DecisionScorer. Defers below the kind's threshold or on mandatory ids."""
        decision = self.decision(id, kind, state)
        result = scorer.score(decision.to_choice_task(require_choice=False))
        probabilities = {o['id']: o['probability'] for o in result['candidates']}
        if set(probabilities) != {c.id for c in decision.candidates}:
            raise ValueError('scorer returned a different candidate set')
        choice = result['choice_id']
        limit = threshold[kind] if isinstance(threshold, dict) else threshold
        if not 0 <= limit <= 1:
            raise ValueError('threshold must be a probability')
        reason = ('mandatory' if choice in self.kind(kind).mandatory_defer else
                  'low_confidence' if probabilities[choice] < limit else None)
        return Prediction(id, kind, choice, probabilities[choice], probabilities, reason is not None,
                          reason, decision.candidate(choice).payload)


class ListSpec(DecisionSpec):
    """A declarative spec whose candidates are a list stored in the state.

    Each kind is {name, question, candidates, version}: `candidates` names the state key that
    holds the list. An item is a string (its text; id `p1`, `p2`, ... by position, payload its
    0-based index) or {id, text, payload}. For example a retrieval filter:

        ListSpec('ragzilla', [dict(name='keep', question='Which passages help answer the question?',
                                   candidates='passages')])
        state = {'question': 'who wrote ...', 'passages': ['...', '...']}
    """

    def __init__(self, name, kinds):
        self.declared = [dict(name=k['name'], question=k['question'], candidates=k['candidates'],
                              version=str(k.get('version', '1'))) for k in kinds]
        super().__init__(name, [Kind(k['name'], k['question'], self._enumerator(k['candidates']),
                                     f"{k['version']}+list:{k['candidates']}") for k in self.declared])

    @staticmethod
    def _enumerator(key):
        def candidates(state):
            if not isinstance(state, dict) or not isinstance(state.get(key), list):
                raise ValueError(f'state must be an object with a list under {key!r}')
            for i, item in enumerate(state[key]):
                if isinstance(item, str):
                    yield Candidate(f'p{i + 1}', item, i)
                else:
                    yield Candidate(str(item['id']), item['text'], item.get('payload'))
        return candidates

    def definition(self):
        return dict(type='list', name=self.name, kinds=self.declared)

    def context(self, kind, state):
        key = next(k['candidates'] for k in self.declared if k['name'] == kind)
        return {k: v for k, v in state.items() if k != key}


def load_spec(path) -> DecisionSpec:
    """Rebuild a spec from its `spec.json` (a ListSpec or a chat-trace spec)."""
    value = json.loads(Path(path).read_text())
    if value.get('type') == 'list':
        return ListSpec(value['name'], value['kinds'])
    if 'tools' in value:
        from .chat_traces import spec_from_tools
        return spec_from_tools(**value)
    raise ValueError(f'{path}: not a spec definition')


@dataclass
class Dataset:
    spec: str
    splits: dict[str, list[Decision]]
    report: dict = field(default_factory=dict)
    definition: dict | None = None  # written as spec.json when the spec can describe itself

    def write(self, out):
        """Write v2 rows per split, plus v1 `*.choice.jsonl` files the current trainer reads."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        for split, rows in self.splits.items():
            with (out/f'{split}.jsonl').open('x') as stream:
                for d in rows:
                    stream.write(canonical(d.to_dict()) + '\n')
            with (out/f'{split}.choice.jsonl').open('x') as stream:
                for d in rows:
                    if isinstance(d.target, Choice):
                        stream.write(canonical(d.to_choice_task().to_dict()) + '\n')
        (out/'manifest.json').write_text(json.dumps(dict(schema_version=2, spec=self.spec,
                                                         report=self.report), indent=2) + '\n')
        if self.definition is not None:
            (out/'spec.json').write_text(json.dumps(self.definition, indent=2) + '\n')


def split_of(group, seed, fractions):
    point = int(digest([seed, group]), 16) / 2**256
    for name, fraction in fractions.items():
        point -= fraction
        if point < 0:
            return name
    return name


def build_dataset(spec: DecisionSpec, episodes: Iterable[Episode], *, fractions=SPLITS,
                  seed='0', min_outcome: float | None = None) -> Dataset:
    """Validate every row against the spec, split by group, and report per-kind label coverage."""
    if not math.isclose(sum(fractions.values()), 1) or min(fractions.values()) < 0:
        raise ValueError('split fractions must be nonnegative and sum to 1')
    splits = {name: [] for name in fractions}
    ids, seen, filtered, duplicates, conflicts = set(), {}, 0, 0, 0
    for episode in episodes:
        if min_outcome is not None and (episode.outcome is None or episode.outcome < min_outcome):
            filtered += 1
            continue
        for d in episode.decisions:
            spec.check(d)
            if d.id in ids:
                raise ValueError(f'duplicate decision id {d.id!r}')
            ids.add(d.id)
            # An identical input in two splits would leak; keep the first occurrence only.
            key = d.to_choice_task(require_choice=False).input_fingerprint
            if key in seen:
                duplicates += 1
                conflicts += seen[key] != d.target
                continue
            seen[key] = d.target
            d = replace(d, group_id=d.group_id or episode.id)
            splits[split_of(d.group_id, seed, fractions)].append(d)
    names = list(splits)
    tasks = {n: [d.to_choice_task(require_choice=False) for d in splits[n]] for n in names}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if tasks[a] and tasks[b]:
                check_disjoint(provenance(tasks[a]), provenance(tasks[b]))
    report = dict(filtered_episodes=filtered, duplicates_dropped=duplicates,
                  duplicates_with_different_label=conflicts, splits={})
    for name, rows in splits.items():
        counts = defaultdict(Counter)
        for d in rows:
            counts[d.kind]['unlabelled' if d.target is None else type(d.target).__name__] += 1
        report['splits'][name] = {k: dict(v) for k, v in sorted(counts.items())}
    # Share of labelled rows whose taken action was not enumerated; serving cannot pick these.
    labelled, misses = Counter(), Counter()
    for d in (d for rows in splits.values() for d in rows if d.target is not None):
        labelled[d.kind] += 1
        misses[d.kind] += isinstance(d.target, NoneCorrect)
    report['recall_miss_rate'] = {k: misses[k] / n for k, n in sorted(labelled.items())}
    return Dataset(spec.fingerprint, splits, report, spec.definition())


def load_decisions(path, spec: DecisionSpec | None = None):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                d = Decision.parse(json.loads(line))
                if spec is not None:
                    spec.check(d)
                rows.append(d)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    return rows
