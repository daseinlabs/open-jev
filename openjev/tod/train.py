"""Dataset directory -> calibrated, evaluated decision model, bound to its DecisionSpec (MLX or PyTorch).

`train_dataset` runs the whole supervised path on a directory written by `Dataset.write`.
Each decision kind gets its own head (choosing a tool and matching an ID are different
tasks; one shared head lets the larger kind crowd out the smaller), all on one frozen
backbone. Per kind: cache features per split, train on train (selecting on validation),
fit a temperature on calibration, and report accuracy on test.

Model layout:

    model/tod.json               spec fingerprint, dataset, trained kinds
    model/heads/<kind>/          a standard decision artifact (+ calibrator.json)
    model/evaluation.json        per-kind test report
"""
from __future__ import annotations

import json
from pathlib import Path

from ..decision import canonical, choice_result, file_digest
from .schema import Choice
from .spec import load_decisions

SPLITS = ('train', 'validation', 'calibration', 'test')
BINDING = 'tod.json'


def dataset_spec(dataset):
    manifest = json.loads((Path(dataset)/'manifest.json').read_text())
    if manifest.get('schema_version') != 2 or not manifest.get('spec'):
        raise ValueError(f'{dataset}: not a decision dataset written by Dataset.write')
    return manifest['spec']


def bind_spec(model, spec_fingerprint, **extra):
    """Record which DecisionSpec built the model's training data."""
    (Path(model)/BINDING).write_text(json.dumps(dict(spec=spec_fingerprint, **extra), indent=2) + '\n')


def read_binding(model):
    path = Path(model)/BINDING
    if not path.exists():
        raise ValueError(f'{model}: no {BINDING}; train it with `openjev tod train`')
    return json.loads(path.read_text())


def artifact_spec(model):
    return read_binding(model)['spec']


def kind_rows(dataset, kind, split):
    return [d for d in load_decisions(Path(dataset)/f'{split}.jsonl')
            if d.kind == kind and isinstance(d.target, Choice)]


def kind_features(dataset, kind, split, model, batch_size=8):
    """Write this kind's v1 rows and cache their features; reuse while the rows are unchanged."""
    from ..decision_mlx import extract_features, inspect_features, versions
    root = Path(dataset)/'kinds'/kind
    root.mkdir(parents=True, exist_ok=True)
    source, out = root/f'{split}.choice.jsonl', root/'features'/f'{split}.npz'
    text = ''.join(canonical(d.to_choice_task().to_dict()) + '\n' for d in kind_rows(dataset, kind, split))
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    if out.exists():
        meta = inspect_features(out)
        if meta['source_sha256'] == file_digest(source) and meta['signature']['runtime'] == versions():
            return out
        out.unlink()
    extract_features(source, model, out, split, batch_size)
    return out


def raw_result(head, fx, limits, artifact_id, task):
    from ..decision_mlx import extract_task, head_logits
    ctx, opt = extract_task(fx, task, limits)
    result = choice_result(task, [float(x) for x in head_logits(head, ctx, opt)])
    result['artifact_id'] = artifact_id
    return result


def evaluate_dataset(model, dataset, refit=False):
    """Per kind: fit (or reuse) a temperature on calibration, then score test."""
    from ..decision_probability import TemperatureCalibrator
    from .picker import evaluate, is_tod_model
    from .rl import HeadPolicy
    if is_tod_model(model):
        return evaluate(model, dataset, refit)
    model, dataset = Path(model), Path(dataset)
    if artifact_spec(model) != dataset_spec(dataset):
        raise ValueError('model and dataset were built with different DecisionSpecs')
    policy = HeadPolicy(model, None, calibrated=False)
    report = {}
    for kind, entry in policy.heads.items():
        path = model/'heads'/kind/'calibrator.json'
        if refit and path.exists():
            path.unlink()
        if not path.exists():
            tasks = [d.to_choice_task() for d in kind_rows(dataset, kind, 'calibration')]
            results = [raw_result(entry['head'], policy.fx, policy.limits, entry['artifact_id'], t) for t in tasks]
            TemperatureCalibrator.fit(tasks, results, entry['artifact_id']).save(path)
        calibrator = TemperatureCalibrator.load(path, entry['artifact_id'])
        hits = confidence = 0.
        rows = kind_rows(dataset, kind, 'test')
        for d in rows:
            result = calibrator.apply(raw_result(entry['head'], policy.fx, policy.limits, entry['artifact_id'],
                                                 d.to_choice_task()))
            hits += result['choice_id'] == d.target.choice_id
            confidence += max(c['probability'] for c in result['candidates'])
        report[kind] = dict(accuracy=hits / len(rows), mean_confidence=confidence / len(rows), n=len(rows),
                            temperature=calibrator.metadata['temperature'])
    (model/'evaluation.json').write_text(json.dumps(dict(test=report), indent=2) + '\n')
    return dict(test=report)


def train_dataset(dataset, model, out, *, epochs=20, learning_rate=5e-5, rank=256, batch_size=64,
                  seed=7, log=None):
    from ..decision_mlx import train_artifact
    from ..backbone import resolve
    log = log or (lambda message: print(message, flush=True))
    dataset, out = Path(dataset), Path(out)
    spec = dataset_spec(dataset)
    model = resolve(model)  # None: $OPENJEV_MODEL, else the default Hugging Face repo
    if out.exists():
        raise ValueError(f'{out} already exists')
    report = json.loads((dataset/'manifest.json').read_text())['report']['splits']
    kinds = sorted({k for split in report.values() for k in split})
    trained, skipped = [], {}
    for kind in kinds:
        missing = [s for s in SPLITS if not report.get(s, {}).get(kind, {}).get('Choice')]
        if missing:
            skipped[kind] = f'no labelled rows in {", ".join(missing)}'
            log(f'{kind}: skipped ({skipped[kind]}); it will always defer')
            continue
        features = {}
        for split in SPLITS:
            log(f'{kind}: features ({split})')
            features[split] = kind_features(dataset, kind, split, model)
        log(f'{kind}: training head')
        out.mkdir(parents=True, exist_ok=True)
        train_artifact(features['train'], features['validation'], out/'heads'/kind, rank=rank, epochs=epochs,
                       batch_size=batch_size, lr=learning_rate, seed=seed)
        trained.append(kind)
    if not trained:
        raise ValueError(f'no kind had labelled rows in every split: {skipped}')
    bind_spec(out, spec, dataset=str(dataset.resolve()), kinds=trained, skipped=skipped)
    log('calibrating and evaluating')
    return dict(model=str(out), skipped=skipped, **evaluate_dataset(out, dataset))
