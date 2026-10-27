"""Serve a trained decision model, play it inside an environment, and improve it from reward (MLX or PyTorch).

Any simulator, replay harness, or sandboxed copy of your product can be an Environment:

    reset(seed)                 start an episode (deterministic per seed for `improve`)
    observe() -> (kind, state)  the next decision, or None when the episode is over
    act(payload)                execute the chosen candidate's payload
    defer()                     let the parent (your big model) make this decision
    reward() -> float           episode return, read once the episode is over

Candidates come from the DecisionSpec, exactly as in training. The backbone stays frozen;
`improve` and `reinforce` update only the per-kind heads and write a new model directory.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import time
from typing import Callable, Protocol

import numpy as np

from .. import compute
from ..decision import file_digest, read_artifact
from .schema import Prediction
from .spec import DecisionSpec
from .train import BINDING, bind_spec, read_binding


class Environment(Protocol):
    def reset(self, seed: int): ...
    def observe(self) -> tuple[str, object] | None: ...
    def act(self, payload) -> None: ...
    def defer(self) -> None: ...
    def reward(self) -> float: ...


@dataclass
class Rollout:
    reward: float
    steps: list = field(default_factory=list)  # (sample, _, chosen index) per decision the model made; the sample is
    # what the policy trains on: (ctx, opt) features for heads, the Decision itself for TOD
    decisions: list = field(default_factory=list)  # the Decision behind each step
    actions: list = field(default_factory=list)  # every payload sent to env.act, in order (for replay)
    decided: int = 0
    deferred: int = 0


def softmax(logits, temperature=1.0):
    z = np.exp((logits - logits.max()) / temperature)
    return z / z.sum()


class HeadPolicy:
    """A trained decision model (one head per kind, one shared backbone) acting through a spec.

    Kinds without a head (too little data at training time) always defer."""

    def __init__(self, model, spec: DecisionSpec | None, backbone=None, calibrated=True):
        from ..decision_mlx import DecisionScorer
        from ..decision_probability import TemperatureCalibrator
        self.model, self.spec = Path(model), spec
        self.binding = read_binding(model)
        if spec is not None:
            spec.require(self.binding['spec'])
        self.heads = {}
        scorer = None
        for kind in self.binding['kinds']:
            artifact = self.model/'heads'/kind
            if scorer is None:  # loads and verifies the backbone once
                scorer = DecisionScorer(artifact, backbone)
                self.fx, self.limits = scorer.fx, scorer.manifest['limits']
                head, fingerprint = scorer.head, scorer.manifest['backbone']['fingerprint']
            else:
                manifest = read_artifact(artifact)
                if manifest['backbone']['fingerprint'] != fingerprint or manifest['limits'] != self.limits:
                    raise ValueError(f'{kind}: head was trained on a different backbone')
                head, _ = scorer.ops.AttentionHead.load(str(artifact/'head.safetensors'))
            artifact_id = file_digest(artifact/'manifest.json')
            temperature = 1.0
            if calibrated and (artifact/'calibrator.json').exists():
                temperature = TemperatureCalibrator.load(artifact/'calibrator.json', artifact_id).metadata['temperature']
            self.heads[kind] = dict(head=head, temperature=temperature, artifact_id=artifact_id)

    # RL hooks (improve / reinforce). Each kind has its own head; TOD shares one adapter.
    shared_weights = False
    rl_defaults = dict(steps=64, learning_rate=1e-5, minibatch=32, anchor_size=256)

    def trainer(self, anchor, anchor_weight, learning_rate, rng):
        return _Trainer(self, anchor, anchor_weight, learning_rate, rng)

    def validation_accuracy(self, kinds):
        """{kind: accuracy on its SFT validation split} for kinds that have one (the RL guard)."""
        if not hasattr(self, '_validation'):
            self._validation = _validation_sets(self)
        return {k: _accuracy(self.heads[k]['head'], self._validation[k]) for k in kinds if k in self._validation}

    def snapshot(self, kind):
        return _snapshot(self.heads[kind]['head'])

    def restore(self, kind, state):
        _restore(self.heads[kind]['head'], state)

    def save(self, out, info):
        _save(self, out, info)

    def logits(self, decision):
        from ..decision_mlx import extract_task, head_logits
        ctx, opt = extract_task(self.fx, decision.to_choice_task(require_choice=False), self.limits)
        return ctx, opt, head_logits(self.heads[decision.kind]['head'], ctx, opt)

    def decide(self, state, kind, threshold: float | dict[str, float] = 0.0, id='request') -> Prediction:
        """One calibrated pick. Defers below the kind's threshold, on the kind's mandatory-defer
        candidates, and for kinds that have no trained head."""
        decision = self.spec.decision(id, kind, state)
        if kind not in self.heads:
            return Prediction(id, kind, decision.candidates[0].id, 0.0, {}, True, 'no_model')
        _, _, logits = self.logits(decision)
        probs = softmax(logits, self.heads[kind]['temperature'])
        index = int(probs.argmax())
        choice = decision.candidates[index]
        limit = threshold[kind] if isinstance(threshold, dict) else threshold
        reason = ('mandatory' if choice.id in self.spec.kind(kind).mandatory_defer else
                  'low_confidence' if probs[index] < limit else None)
        return Prediction(id, kind, choice.id, float(probs[index]),
                          {c.id: float(p) for c, p in zip(decision.candidates, probs)},
                          reason is not None, reason, choice.payload)

    def rollout(self, env: Environment, seed, *, rng=None, threshold=None, replay=()) -> Rollout:
        """Play one episode. `rng` samples choices (exploration); otherwise take the argmax.
        With `threshold`, decisions whose probability falls below it are deferred.
        `replay` payloads are acted first, without the model (to branch from a point)."""
        env.reset(seed)
        run = Rollout(0.)
        for payload in replay:
            if env.observe() is None:
                break
            run.actions.append(payload)
            env.act(payload)
        while (observation := env.observe()) is not None:
            kind, state = observation
            if kind not in self.heads:
                run.deferred += 1
                env.defer()
                continue
            decision = self.spec.decision('live', kind, state)
            ctx, opt, logits = self.logits(decision)
            probs = softmax(logits, self.heads[kind]['temperature'])
            index = int(rng.choice(len(probs), p=probs)) if rng is not None else int(probs.argmax())
            if threshold is not None and probs[index] < threshold:
                run.deferred += 1
                env.defer()
                continue
            run.decided += 1
            run.steps.append((ctx, opt, index))
            run.decisions.append(decision)
            run.actions.append(decision.candidates[index].payload)
            env.act(decision.candidates[index].payload)
        run.reward = float(env.reward())
        return run

    def evaluate(self, make_env: Callable[[], Environment], seeds, threshold=None):
        runs = [self.rollout(make_env(), int(s), threshold=threshold) for s in seeds]
        decided = sum(r.decided for r in runs)
        return dict(mean_reward=float(np.mean([r.reward for r in runs])),
                    decisions_handled=decided / max(1, decided + sum(r.deferred for r in runs)),
                    fully_autonomous=float(np.mean([r.deferred == 0 for r in runs])), episodes=len(runs))


def load_policy(model, spec, **kwargs):
    """HeadPolicy for a head model, TOD policy for a model from `openjev tod train --method tod`."""
    from .picker import is_tod_model, policy
    return policy(model, spec, **kwargs) if is_tod_model(model) else HeadPolicy(model, spec, **kwargs)


def _pad(arrays):
    width = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), width, arrays[0].shape[1]), np.float16)
    mask = np.zeros((len(arrays), width), np.float32)
    for i, a in enumerate(arrays):
        out[i, :len(a)], mask[i, :len(a)] = a, 1
    return out, mask


class _Trainer:
    """Per-kind optimizers and loss, with an optional supervised anchor per kind."""

    def __init__(self, policy, anchor, anchor_weight, learning_rate, rng):
        self.policy, self.rng, self.anchor_weight = policy, rng, anchor_weight
        self.optimizers = {k: compute.ops(e['head'].backend).Optimizer(e['head'], learning_rate)
                           for k, e in policy.heads.items()}
        self.anchors = {}
        if anchor:
            from ..features import FeatureSet
            dataset = Path(policy.binding['dataset'])
            for kind in policy.heads:
                path = dataset/'kinds'/kind/'features'/'train.npz'
                if path.exists():
                    self.anchors[kind] = FeatureSet(str(path))

    def update(self, kind, batch, anchor_size, pooled=False):
        """batch: [(ctx, opt, index, weight)]; weight 1 = cross-entropy toward index,
        an advantage = policy-gradient step.

        pooled: average over batch and anchor rows together, so a handful of new labels
        weigh as many rows rather than as much as the whole anchor. Without it, the
        batch mean and the anchor mean count equally."""
        ctx, cm = _pad([b[0] for b in batch])
        opt, om = _pad([b[1] for b in batch])
        anchor = self.anchors.get(kind)
        m = min(anchor_size, len(anchor)) if anchor is not None else 0
        n = len(batch)
        terms = [((ctx, cm, opt, om, np.array([b[2] for b in batch], np.int32)),
                  np.array([b[3] for b in batch], np.float32), n / (n + m) if pooled else 1.0, 0.)]
        if m:
            indices = self.rng.choice(len(anchor), m, replace=False).tolist()
            terms.append((anchor.batch(indices), None, self.anchor_weight * (m / (n + m) if pooled else 1.0), 0.))
        self.optimizers[kind].step(terms)


def _validation_sets(policy):
    from ..features import FeatureSet
    dataset = Path(policy.binding['dataset'])
    sets = {}
    for kind in policy.heads:
        path = dataset/'kinds'/kind/'features'/'validation.npz'
        if path.exists():
            sets[kind] = FeatureSet(str(path))
    return sets


def _accuracy(head, features):
    from ..train import evaluate
    return evaluate(head, features)['top1']


def _snapshot(head):
    return compute.ops(head.backend).snapshot(head)


def _restore(head, snapshot):
    compute.ops(head.backend).restore(head, snapshot)


def _by_kind(samples):
    groups = defaultdict(list)
    for kind, sample in samples:
        groups[kind].append(sample)
    return groups


def improve(policy: HeadPolicy, make_env: Callable[[], Environment], out, *, anchor=True, anchor_weight=1.0,
            iterations=3, episodes=128, steps=None, learning_rate=None, minibatch=None, anchor_size=None,
            max_reward=None, tolerance=0.02, seed=30_000_000, log=None):
    """Rollout-based policy improvement: learn from reward by trying the alternatives.

    Each iteration plays `episodes` greedy episodes. In every episode that did not reach
    `max_reward`, decisions are revisited from the last one backwards: the environment is
    reset to the same seed, replayed up to that decision, every other candidate is tried,
    and the policy finishes the episode. The latest decision where an alternative earns more
    reward becomes a training label. (Earlier branches replay the same seed, so they can
    "succeed" on information the model could not have had yet.) Labels accumulate across
    iterations. Each iteration takes `steps` updates per kind: up to `minibatch` labels plus
    `anchor_size` SFT training rows of the same kind (`anchor`), every row weighted equally.
    The anchor rows show the contrasting cases (e.g. open return windows), so a correction
    stays specific to where it applies instead of spreading across similar states.

    Guard: after training each kind, if its accuracy on its SFT validation split (every
    kind's, when kinds share weights as in TOD) fell more than `tolerance` below the starting
    model, that update is rolled back. Some failures are hard cases the model cannot fit
    without forgetting others; RL keeps only what it can learn safely.

    `steps`, `learning_rate`, `minibatch` and `anchor_size` default to `policy.rl_defaults`
    (smaller for TOD, where every row is a 12B forward and backward pass).

    Unlike REINFORCE this tries every option at the decisions that went wrong, so it finds
    actions the policy gives ~0 probability. It needs `reset(seed)` to be deterministic.
    """
    log = log or (lambda message: print(message, flush=True))
    out = Path(out)
    if out.exists():
        raise ValueError(f'{out} already exists')
    settings = dict(policy.rl_defaults)
    settings.update({k: v for k, v in dict(steps=steps, learning_rate=learning_rate, minibatch=minibatch,
                                            anchor_size=anchor_size).items() if v is not None})
    steps, learning_rate, minibatch, anchor_size = (settings[k] for k in ('steps', 'learning_rate', 'minibatch', 'anchor_size'))
    rng = np.random.default_rng(seed)
    trainer = policy.trainer(anchor, anchor_weight, learning_rate, rng)
    floor = {k: v - tolerance for k, v in policy.validation_accuracy(list(policy.heads)).items()}
    labels, curve = [], []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        seeds = [int(s) for s in seed + (iteration - 1) * episodes + np.arange(episodes)]
        runs = [policy.rollout(make_env(), s) for s in seeds]
        found = 0
        for s, run in zip(seeds, runs):
            if max_reward is not None and run.reward >= max_reward:
                continue
            for i in reversed(range(len(run.steps))):
                (a, b, chosen), decision = run.steps[i], run.decisions[i]
                best, best_reward = chosen, run.reward
                for alt, candidate in enumerate(decision.candidates):
                    if alt != chosen:
                        reward = policy.rollout(make_env(), s, replay=[*run.actions[:i], candidate.payload]).reward
                        if reward > best_reward:
                            best, best_reward = alt, reward
                if best != chosen:
                    labels.append((decision.kind, (a, b, best, 1.0)))
                    found += 1
                    break
        kept, rolled_back = [], []
        for kind, samples in _by_kind(labels).items():
            snapshot = policy.snapshot(kind)
            for _ in range(steps):
                pick = rng.choice(len(samples), min(minibatch, len(samples)), replace=False)
                trainer.update(kind, [samples[i] for i in pick], anchor_size, pooled=True)
            accuracy = policy.validation_accuracy(list(floor) if policy.shared_weights else [kind])
            if any(accuracy[k] < floor[k] for k in accuracy):
                policy.restore(kind, snapshot)
                rolled_back.append(kind)
            else:
                kept.append(kind)
        curve.append(dict(iteration=iteration, greedy_mean_reward=float(np.mean([r.reward for r in runs])),
                          new_labels=found, total_labels=len(labels), updated=kept, rolled_back=rolled_back,
                          seconds=round(time.perf_counter() - started, 1)))
        log(json.dumps(curve[-1]))
    policy.save(out, dict(algorithm='rollout_policy_improvement', learning_rate=learning_rate,
                            iterations=iterations, episodes_per_iteration=episodes, steps=steps, anchor_size=anchor_size,
                            max_reward=max_reward, tolerance=tolerance, seed=seed, anchor=bool(trainer.anchors),
                            anchor_weight=anchor_weight, curve=curve))
    return dict(model=str(out), curve=curve)


def reinforce(policy: HeadPolicy, make_env: Callable[[], Environment], out, *, anchor=True, anchor_weight=1.0,
              iterations=15, episodes=128, learning_rate=1e-5, minibatch=64, seed=20_000_000, log=None):
    """REINFORCE with a normalized batch baseline: each decision is pushed toward (or away
    from) its choice in proportion to how much better than average its episode did.

    Cheap per step, but it only learns about actions it happens to sample: an option the
    policy gives ~0 probability is never tried. Prefer `improve` when the environment is
    deterministic per seed. Build the policy with `calibrated=False`."""
    log = log or (lambda message: print(message, flush=True))
    out = Path(out)
    if out.exists():
        raise ValueError(f'{out} already exists')
    if any(entry['temperature'] != 1.0 for entry in policy.heads.values()):
        raise ValueError('sample from the raw heads: use HeadPolicy(..., calibrated=False) for RL')
    rng = np.random.default_rng(seed)
    trainer = policy.trainer(anchor, anchor_weight, learning_rate, rng)
    curve = []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        seeds = seed + (iteration - 1) * episodes + np.arange(episodes)
        runs = [policy.rollout(make_env(), int(s), rng=rng) for s in seeds]
        rewards = np.array([r.reward for r in runs])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        samples = [(d.kind, (x, y, index, float(a)))
                   for r, a in zip(runs, advantages) for (x, y, index), d in zip(r.steps, r.decisions)]
        for kind, group in _by_kind(samples).items():
            order = rng.permutation(len(group))
            for start in range(0, len(group), minibatch):
                trainer.update(kind, [group[i] for i in order[start:start + minibatch]], minibatch)
        curve.append(dict(iteration=iteration, mean_reward=float(rewards.mean()), decisions=len(samples),
                          seconds=round(time.perf_counter() - started, 1)))
        log(json.dumps(curve[-1]))
    policy.save(out, dict(algorithm='reinforce_normalized_baseline', learning_rate=learning_rate,
                            iterations=iterations, episodes_per_iteration=episodes, seed=seed,
                            anchor=bool(trainer.anchors), anchor_weight=anchor_weight, curve=curve))
    return dict(model=str(out), curve=curve)


def _save(policy, out, info):
    """Write the updated heads as a new model; calibration must be refit for it."""
    source = policy.model
    for kind, entry in policy.heads.items():
        src, dst = source/'heads'/kind, out/'heads'/kind
        dst.mkdir(parents=True)
        for name in ('head.json', 'manifest.json'):
            shutil.copy(src/name, dst/name)
        entry['head'].save(str(dst/'head.safetensors'), json.loads((src/'head.json').read_text()))
        manifest = json.loads((dst/'manifest.json').read_text())
        manifest['files'] = {name: file_digest(dst/name) for name in ('head.safetensors', 'head.json')}
        manifest['training']['rl'] = dict(initial_artifact_id=entry['artifact_id'], **info)
        (dst/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    binding = dict(policy.binding)
    binding.pop('rl_from', None)
    bind_spec(out, binding.pop('spec'), **binding, rl_from=str(source))
