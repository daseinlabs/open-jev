"""Reproducible four-split probability-quality experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import tomllib

from .decision import check_disjoint, choice_result, digest, file_digest, load_tasks, provenance, read_artifact
from .decision_probability import TemperatureCalibrator, baseline_predictions, probability_report

SPLITS = ('train', 'validation', 'calibration', 'test')


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    out: str
    datasets: dict[str, str]
    caches: dict[str, str]
    brier_weights: tuple[float, ...] = (0.0, 0.1, 1.0)
    rank: int = 256
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-4
    seed: int = 7
    selection: str = 'nll'
    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 7
    feature_batch_size: int = 8
    context_tokens: int = 4096
    candidate_tokens: int = 256

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        value = tomllib.loads(path.read_text())
        allowed = {'schema_version', 'model', 'out', 'datasets', 'caches', 'training', 'calibration', 'evaluation', 'features'}
        if set(value)-allowed or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError('expected experiment schema_version=1 and known keys')
        def resolve(v):
            if not isinstance(v, str) or not v.strip():
                raise ValueError('paths must be nonempty strings')
            return str((path.parent/v).resolve())
        def section(name, keys):
            v = value.get(name, {})
            if not isinstance(v, dict) or set(v)-set(keys):
                raise ValueError(f'unknown or invalid {name} settings')
            return v
        datasets = section('datasets', SPLITS)
        caches = section('caches', SPLITS)
        if set(datasets) != set(SPLITS) or (caches and set(caches) != set(SPLITS)):
            raise ValueError('four dataset splits required; caches must supply all four or none')
        tr = section('training', ('brier_weights', 'rank', 'epochs', 'batch_size', 'lr', 'seed', 'selection'))
        ca = section('calibration', ('temperature_bounds',))
        ev = section('evaluation', ('bootstrap_samples', 'bootstrap_seed'))
        fe = section('features', ('feature_batch_size', 'context_tokens', 'candidate_tokens'))
        if 'model' not in value or 'out' not in value:
            raise ValueError('model and out paths are required')
        config = cls(model=resolve(value['model']), out=resolve(value['out']),
                     datasets={k: resolve(v) for k, v in datasets.items()},
                     caches={k: resolve(v) for k, v in caches.items()}, **tr, **ca, **ev, **fe)
        for name in ('rank', 'epochs', 'batch_size', 'bootstrap_samples', 'feature_batch_size',
                     'context_tokens', 'candidate_tokens'):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'bootstrap_seed'):
            if type(getattr(config, name)) is not int or not 0 <= getattr(config, name) < 2**32:
                raise ValueError(f'{name} must be an integer in [0, 2**32)')
        def finite(v):
            return type(v) in (int, float) and math.isfinite(v)
        if not finite(config.lr) or config.lr <= 0 or config.selection not in ('nll', 'brier'):
            raise ValueError('positive finite lr and selection=nll or brier required')
        weights = config.brier_weights
        if (not isinstance(weights, (list, tuple)) or not weights or
                not all(finite(w) and w >= 0 for w in weights) or
                len(set(weights)) != len(weights) or 0 not in weights or not any(w > 0 for w in weights)):
            raise ValueError('brier_weights must be unique, finite, nonnegative, and include zero and a positive weight')
        bounds = config.temperature_bounds
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(finite(b) for b in bounds) or
                not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]):
            raise ValueError('temperature_bounds must be positive, ordered, finite, and contain 1')
        return config


def validate_splits(config):
    tasks = {split: load_tasks(path, labelled=True) for split, path in config.datasets.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        check_disjoint(provenance(tasks[a]), provenance(tasks[b]))
    return tasks


def validate_cache(path, data, tasks, split):
    import numpy as np
    from .decision_mlx import inspect_features
    meta = inspect_features(path)
    if (meta['split'] != split or meta['source_sha256'] != file_digest(data) or
            meta['provenance'] != provenance(tasks)):
        raise ValueError(f'{split}: cache does not match dataset content, order, provenance, or role')
    with np.load(path, allow_pickle=False) as z:
        if z['labels'].tolist() != [t.label for t in tasks]:
            raise ValueError(f'{split}: cached labels differ from dataset')
    return meta


def score_cached(artifact, cache, tasks, batch_size=64):
    """Score checked frozen features using the artifact's normal attention head."""
    from . import compute
    from .decision_mlx import inspect_features
    meta, manifest = inspect_features(cache), read_artifact(artifact)
    if meta['signature_sha256'] != digest(manifest['feature_signature']) or meta['provenance'] != provenance(tasks):
        raise ValueError('cache/artifact/task feature contract mismatch')
    from .features import FeatureSet
    ops = compute.ops()
    fs = FeatureSet(str(cache))
    head, _ = ops.AttentionHead.load(str(Path(artifact)/'head.safetensors'))
    artifact_id = file_digest(Path(artifact)/'manifest.json')
    results = []
    for start in range(0, len(tasks), batch_size):
        indices = list(range(start, min(start+batch_size, len(tasks))))
        ctx, cm, opt, om, _ = fs.batch(indices)
        logits = ops.forward(head, ctx, cm, opt, om)
        for i, row in zip(indices, logits.tolist()):
            result = choice_result(tasks[i], row[:len(tasks[i].candidates)])
            result['artifact_id'] = artifact_id
            results.append(result)
    return results


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _predictions(path, tasks, results):
    with Path(path).open('x') as stream:
        for task, result in zip(tasks, results):
            stream.write(json.dumps(dict(target=task.choice_id, **result), allow_nan=False)+'\n')


def run_experiment(config_path):
    """Train all configured losses, lock validation selection, then evaluate test."""
    from .decision_mlx import extract_features, model_identity, train_artifact
    config = ExperimentConfig.load(config_path)
    tasks = validate_splits(config)  # Fail leakage checks before model/GPU initialization.
    out = Path(config.out)
    if out.exists():
        raise ValueError('experiment output directory already exists')
    caches = config.caches or {s: str(out/f'{s}.npz') for s in SPLITS}
    if config.caches:
        metas = {s: validate_cache(caches[s], config.datasets[s], tasks[s], s) for s in SPLITS}
        if len({m['signature_sha256'] for m in metas.values()}) != 1:
            raise ValueError('feature contracts differ across splits')
        identity = model_identity(config.model)
        if identity['fingerprint'] != metas['train']['backbone']['fingerprint']:
            raise ValueError('configured model differs from cached backbone')
    out.mkdir(parents=True, exist_ok=False)
    _save(out/'config.json', dict(schema_version=1, source_sha256=file_digest(config_path), **asdict(config)))
    if not config.caches:
        for split in SPLITS:
            extract_features(config.datasets[split], config.model, caches[split], split,
                             config.feature_batch_size, config.context_tokens, config.candidate_tokens)
        metas = {s: validate_cache(caches[s], config.datasets[s], tasks[s], s) for s in SPLITS}
        if len({m['signature_sha256'] for m in metas.values()}) != 1:
            raise ValueError('feature contracts differ across splits')
    _save(out/'inputs.json', {s: dict(data_sha256=file_digest(config.datasets[s]),
        cache_sha256=file_digest(caches[s]), feature_signature=metas[s]['signature'],
        examples=len(tasks[s]), provenance=provenance(tasks[s])) for s in SPLITS})
    trials = []
    for index, weight in enumerate(config.brier_weights):
        trial = out/f'trial-{index:02d}'
        trial.mkdir()
        artifact = trial/'artifact'
        train_artifact(caches['train'], caches['validation'], artifact,
                       rank=config.rank, epochs=config.epochs, batch_size=config.batch_size,
                       lr=config.lr, seed=config.seed, brier_weight=weight, selection=config.selection)
        val = score_cached(artifact, caches['validation'], tasks['validation'], config.batch_size)
        _predictions(trial/'validation.jsonl', tasks['validation'], val)
        metrics = probability_report(tasks['validation'], val, bootstrap_samples=0)
        trials.append(dict(name=trial.name, brier_weight=weight, validation=metrics,
                           artifact_id=file_digest(artifact/'manifest.json')))
    selected = min(trials, key=lambda t: t['validation'][config.selection])
    # Persist all choices before calibration fitting or test predictions exist.
    _save(out/'selection.json', dict(metric=f'validation_{config.selection}', tie_break='first_configured_weight',
        selected=selected['name'], trials=trials, checkpoint_selection=f'validation_{config.selection}',
        calibration_comparison=['raw', 'nll', 'brier'], test_used_for_selection=False))
    for trial in trials:
        root = out/trial['name']
        calibration = score_cached(root/'artifact', caches['calibration'], tasks['calibration'], config.batch_size)
        _predictions(root/'calibration-raw.jsonl', tasks['calibration'], calibration)
        calibrators = {}
        for objective in ('nll', 'brier'):
            fitted = TemperatureCalibrator.fit(tasks['calibration'], calibration, trial['artifact_id'],
                                               objective, config.temperature_bounds)
            fitted.save(root/f'calibrator-{objective}.json')
            calibrators[objective] = fitted
        raw = score_cached(root/'artifact', caches['test'], tasks['test'], config.batch_size)
        variants = dict(raw=raw, **{k: [c.apply(r) for r in raw] for k, c in calibrators.items()})
        trial['calibrators'] = {k: c.metadata for k, c in calibrators.items()}
        trial['test'] = {}
        for name, predictions in variants.items():
            trial['test'][name] = probability_report(tasks['test'], predictions,
                                                     config.bootstrap_samples, config.bootstrap_seed)
            _predictions(root/f'test-{name}.jsonl', tasks['test'], predictions)
    baselines = {}
    for name, predictions in baseline_predictions(tasks['train'], tasks['test']).items():
        baselines[name] = probability_report(tasks['test'], predictions, config.bootstrap_samples, config.bootstrap_seed)
        _predictions(out/f'test-{name}.jsonl', tasks['test'], predictions)
    report = dict(schema_version=1, selected_trial=selected['name'], trials=trials, baselines=baselines,
        brier_convention='sum_over_valid_candidates', split_sizes={s: len(tasks[s]) for s in SPLITS},
        probability_semantics='distribution over offered choices, not outcome-success probability',
        selection='head checkpoint and loss weight selected on validation; temperatures fit only on calibration',
        stratification='candidate count and question; explicit task-family and difficulty metadata unavailable',
        uncertainty='95% cluster percentile intervals; group IDs or singleton examples; one training seed')
    _save(out/'metrics.json', report)
    lines = ['# Phase 2 probability-quality ablation', '',
             f"Selected head: **{selected['name']}**, by validation {config.selection} (first tie).",
             'Temperatures are fitted only on the separate calibration split. All test variants are reported.', '',
             '| Loss weight | Scaling | Temperature | Accuracy | NLL | Summed Brier | ECE |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for trial in trials:
        for name, metrics in trial['test'].items():
            temperature = 1.0 if name == 'raw' else trial['calibrators'][name]['temperature']
            lines.append(f"| {trial['brier_weight']:g} | {name} | {temperature:.5g} | {metrics['accuracy']:.4f} | {metrics['nll']:.6g} | {metrics['brier']:.6g} | {metrics['ece']:.6g} |")
    for name, metrics in baselines.items():
        lines.append(f"| — | {name} | 1 | {metrics['accuracy']:.4f} | {metrics['nll']:.6g} | {metrics['brier']:.6g} | {metrics['ece']:.6g} |")
    lines += ['', '## Interpretation and reproducibility', '',
              'Each loss uses the same initialization seed, data, optimizer, and epoch budget. '
              'Checkpoint and loss-weight selection use validation data. Temperature fitting uses calibration data. '
              'Test metrics do not choose a loss or a scaling objective.', '',
              'A positive scalar temperature preserves the chosen candidate. Metric tradeoffs should be read separately; '
              'low ECE alone does not establish a useful decision model. Bounds and boundary optima are recorded in each calibrator.', '',
              'Uniform and add-one-smoothed training chosen-text priors are conditioned on the offered candidates. '
              'Uniform ties follow candidate-ID ordering; chance accuracy can differ from observed tie-broken accuracy.', '',
              'See `metrics.json` for reliability bins, selective error/coverage (confidence ties retained), '
              'candidate-count/question strata, and 95% group-bootstrap intervals. '
              'Intervals measure dataset sampling uncertainty, not variability across training seeds. '
              'Task-family and difficulty strata require additional dataset metadata.', '',
              'The resolved configuration, input hashes, selection record, artifacts, calibrators, and per-example '
              'predictions are saved alongside this report. Model/tokenizer weights remain external.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return dict(out=str(out), selected_trial=selected['name'], report=str(out/'report.md'))
