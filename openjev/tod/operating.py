"""Operating points for Subset kinds: which candidates to keep, chosen for a survival target.

A Subset model gives each candidate an independent probability p_i. Two keep rules:

    tau (absolute)   keep i when p_i >= tau
    rho (relative)   keep i when p_i >= rho * max_j p_j; the top candidate is always kept

A filter is judged per question, not per candidate. A question *survives* when the kept
set still holds a relevant candidate (`survival='any'`, answer survival) or all of them
(`survival='all'`). For each rule and target (e.g. 0.99), the operating point is the largest
value whose survival on the calibration split reaches the target: the fewest candidates
kept for that survival. Points are stored in `model/operating_points.json`:

    {"default": "rho_99", "survival": "any", "targets": [0.95, 0.98, 0.99],
     "kinds": {"keep": {"tau_95": 0.0863, ..., "rho_99": 0.1363}},
     "calibration": {"keep": {"questions": 81796, "with_relevant": 64948}}}

Pure numpy; safe to import without a backend.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TARGETS = (0.95, 0.98, 0.99)
RULES = ('tau', 'rho')
DEFAULT_POINT = 'rho_99'
SURVIVAL = ('any', 'all')
FILE = 'operating_points.json'


def point_name(rule, target):
    return f'{rule}_{round(target * 100):02d}'


def parse_point(name):
    rule, _, target = name.partition('_')
    if rule not in RULES or not target.isdigit():
        raise ValueError(f'operating point names look like tau_99 or rho_98, not {name!r}')
    return rule, int(target) / 100


def kept(probabilities, rule, value):
    """Boolean keep mask for one question's candidates."""
    p = np.asarray(probabilities, np.float64)
    if rule == 'tau':
        return p >= value
    if rule == 'rho':
        mask = p >= value * p.max()
        mask[int(p.argmax())] = True
        return mask
    raise ValueError(f'rule must be one of {RULES}')


def _survives(relevant, mask, survival):
    hits = relevant & mask
    return hits.any() if survival == 'any' else hits.sum() == relevant.sum()


def _statistic(p, relevant, rule, survival):
    """The largest rule value at which this question still survives."""
    r = p[relevant]
    s = r.max() if survival == 'any' else r.min()
    return s if rule == 'tau' else (1.0 if p.max() <= 0 else s / p.max())


def choose(questions, rule, target, survival='any'):
    """The largest `rule` value reaching `target` survival on questions [(p, relevant)].
    Questions with no relevant candidate do not count towards survival."""
    if survival not in SURVIVAL:
        raise ValueError(f'survival must be one of {SURVIVAL}')
    if not 0 < target <= 1:
        raise ValueError('targets must be in (0, 1]')
    stats = sorted((_statistic(np.asarray(p, np.float64), np.asarray(r, bool), rule, survival)
                    for p, r in questions if np.any(r)), reverse=True)
    if not stats:
        raise ValueError('no question has a relevant candidate; cannot choose an operating point')
    k = max(1, math.ceil(target * len(stats) - 1e-9))
    # A hair lower so rounding in p >= value * max cannot drop the boundary question.
    return float(stats[k - 1] * (1 - 1e-12))


def choose_points(questions, targets=TARGETS, survival='any'):
    return {point_name(rule, t): choose(questions, rule, t, survival) for rule in RULES for t in targets}


def save(model, points_by_kind, *, default=DEFAULT_POINT, survival='any', targets=TARGETS, counts=None):
    for kind, points in points_by_kind.items():
        if default not in points:
            raise ValueError(f'{kind}: default point {default!r} was not chosen (targets {list(targets)})')
    value = dict(default=default, survival=survival, targets=list(targets), kinds=points_by_kind,
                 calibration=counts or {})
    (Path(model)/FILE).write_text(json.dumps(value, indent=2) + '\n')
    return value


def load(model):
    path = Path(model)/FILE
    return json.loads(path.read_text()) if path.exists() else None


def metrics(questions, points, survival='any'):
    """Set metrics for questions [dict(p, relevant, tokens)] at each point {name: (rule, value)}.

    answer_survival / all_survival: over questions with a relevant candidate, the share keeping
    one / all of them. tokens_kept: kept candidate tokens over all candidate tokens, every
    question. candidates_kept: mean per question. top_k: the smallest fixed prefix (the
    candidates' given order) whose survival reaches the point's, and its tokens_kept."""
    rows = [(np.asarray(q['p'], np.float64), np.asarray(q['relevant'], bool), np.asarray(q['tokens'], np.float64))
            for q in questions]
    with_relevant = [i for i, (_, r, _) in enumerate(rows) if r.any()]
    total_tokens = sum(t.sum() for _, _, t in rows) or 1.0

    def summarize(masks):
        any_ = [(rows[i][1] & masks[i]).any() for i in with_relevant]
        all_ = [(rows[i][1] & masks[i]).sum() == rows[i][1].sum() for i in with_relevant]
        return dict(answer_survival=float(np.mean(any_)) if any_ else None,
                    all_survival=float(np.mean(all_)) if all_ else None,
                    tokens_kept=float(sum(t[m].sum() for (_, _, t), m in zip(rows, masks)) / total_tokens),
                    candidates_kept=float(np.mean([m.sum() for m in masks])) if masks else 0.0)

    width = max((len(p) for p, _, _ in rows), default=0)
    prefixes = []
    for k in range(1, width + 1):
        masks = [np.arange(len(p)) < k for p, _, _ in rows]
        prefixes.append(dict(k=k, **summarize(masks)))
    key = 'answer_survival' if survival == 'any' else 'all_survival'
    report = dict(questions=len(rows), with_relevant=len(with_relevant))
    for name, (rule, value) in points.items():
        entry = dict(rule=rule, value=value, **summarize([kept(p, rule, value) for p, _, _ in rows]))
        if entry[key] is not None:
            prefix = next((x for x in prefixes if x[key] >= entry[key] - 1e-12), prefixes[-1] if prefixes else None)
            if prefix:
                entry['top_k'] = dict(k=prefix['k'], survival=prefix[key], tokens_kept=prefix['tokens_kept'])
        report[name] = entry
    return report


__all__ = ['TARGETS', 'RULES', 'DEFAULT_POINT', 'kept', 'choose', 'choose_points', 'metrics', 'point_name',
           'parse_point', 'load', 'save']
