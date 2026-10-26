"""Native candidate-head training and inference on MLX or PyTorch (see `openjev.compute`),
imported only when requested. (The module name predates the PyTorch backend.)"""
from __future__ import annotations

import json
from pathlib import Path
import time
from collections.abc import Mapping

import numpy as np

from . import compute
from .decision import (RECIPE, ChoiceTask, check_disjoint, choice_result, digest,
                       file_digest, load_tasks, model_identity, provenance,
                       read_artifact, render_context)


def versions():
    return compute.versions()


def extractor(model, batch_size):
    from .scorer import OptionScorer
    from .features import FeatureExtractor
    backend = compute.backend()
    scorer = OptionScorer(model, backend=backend, device='auto' if backend == 'mlx' else compute.device(),
                          batch_size=batch_size, chat=False, sep='')
    if scorer.bos_id is None:
        raise ValueError('independent candidate features require a BOS token')
    scorer.model.eval()
    return FeatureExtractor(scorer, contextual=False)


def extract_task(fx, task, limits):
    context = render_context(task)
    texts = [c.text for c in task.candidates]
    if len(fx.s.context_ids(context)) > limits['context_tokens']:
        raise ValueError(f'{task.id}: context exceeds token limit; truncation is disabled')
    if any(len(fx.s.option_ids(text)) > limits['candidate_tokens'] for text in texts):
        raise ValueError(f'{task.id}: candidate exceeds token limit; truncation is disabled')
    return fx.extract(context, texts, chat=False, sep='')


def extract_features(data, model, out, split, batch_size=8, context_tokens=4096, candidate_tokens=256):
    if batch_size < 1 or context_tokens < 1 or candidate_tokens < 1:
        raise ValueError('batch size and token limits must be positive')
    if split not in ('train', 'validation', 'calibration', 'test'):
        raise ValueError('split must be train, validation, calibration, or test')
    out = Path(out)
    if out.exists() or out.suffix != '.npz':
        raise ValueError('feature output must be a new .npz file')
    tasks = load_tasks(data, labelled=True)
    identity = model_identity(model)
    fx = extractor(model, batch_size)
    limits = dict(context_tokens=context_tokens, candidate_tokens=candidate_tokens)
    signature = dict(model_fingerprint=identity['fingerprint'], recipe=RECIPE,
                     hidden=fx.hidden, limits=limits, runtime=versions())
    arrays = {}
    started = time.perf_counter()
    for i, task in enumerate(tasks):
        arrays[f'ctx_{i}'], arrays[f'opt_{i}'] = extract_task(fx, task, limits)
        if (i+1) % 100 == 0:
            print(f'Extracted {i+1}/{len(tasks)}', flush=True)
    meta = dict(schema_version=1, strategy='candidate_head', hidden=fx.hidden, n=len(tasks),
                split=split, source=str(Path(data).resolve()), source_sha256=file_digest(data),
                backbone=identity, recipe=RECIPE, limits=limits, signature=signature,
                signature_sha256=digest(signature), provenance=provenance(tasks),
                seconds=time.perf_counter()-started)
    arrays['labels'] = np.array([t.label for t in tasks], dtype=np.int32)
    arrays['meta'] = np.array(json.dumps(meta))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez(stream, **arrays)
    return meta


def inspect_features(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta']))
        if meta.get('schema_version') != 1 or meta.get('strategy') != 'candidate_head' or meta.get('recipe') != RECIPE:
            raise ValueError('expected native decision features; regenerate legacy caches with decision features')
        if digest(meta['signature']) != meta['signature_sha256']:
            raise ValueError('invalid feature signature')
        if (meta['signature']['model_fingerprint'] != meta['backbone']['fingerprint'] or
                digest(meta['backbone']['files']) != meta['backbone']['fingerprint'] or
                meta['signature']['recipe'] != RECIPE or meta['signature']['hidden'] != meta['hidden'] or
                meta['signature']['limits'] != meta['limits']):
            raise ValueError('inconsistent feature metadata')
        labels = archive['labels']
        if labels.ndim != 1 or labels.dtype.kind not in 'iu' or not len(labels) or len(labels) != meta['n']:
            raise ValueError('invalid feature labels/count')
        if len(meta['provenance']['ids']) != len(labels) or len(meta['provenance']['inputs']) != len(labels):
            raise ValueError('invalid feature provenance')
        for i, label in enumerate(labels):
            c, o = archive[f'ctx_{i}'], archive[f'opt_{i}']
            if (c.ndim != 2 or o.ndim != 2 or c.shape[0] < 1 or not 2 <= o.shape[0] <= 255 or
                    c.shape[1] != meta['hidden'] or o.shape[1] != meta['hidden'] or
                    c.dtype != np.float16 or o.dtype != np.float16 or not 0 <= label < len(o) or
                    not np.isfinite(c).all() or not np.isfinite(o).all()):
                raise ValueError(f'invalid feature tensors or label at example {i}')
    return meta


def train_artifact(train_path, validation_path, out, rank=256, epochs=8, batch_size=64, lr=1e-4, seed=7,
                   brier_weight=0.0, selection='top1'):
    if min(rank, epochs, batch_size) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive rank, epochs, batch size, and finite learning rate required')
    if not np.isfinite(brier_weight) or brier_weight < 0 or selection not in ('top1', 'nll', 'brier'):
        raise ValueError('invalid Brier weight or checkpoint selection metric')
    tr, va = inspect_features(train_path), inspect_features(validation_path)
    if tr['split'] != 'train' or va['split'] != 'validation':
        raise ValueError('training needs train and validation feature splits; test cannot select a checkpoint')
    if tr['signature_sha256'] != va['signature_sha256']:
        raise ValueError('training/validation feature contracts differ')
    check_disjoint(tr['provenance'], va['provenance'])
    out = Path(out)
    if out.exists():
        raise ValueError('artifact output directory already exists')
    from .train import train
    out.mkdir(parents=True, exist_ok=False)
    result = train(str(train_path), str(validation_path), str(out/'head.safetensors'),
                   rank=rank, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                   brier_weight=brier_weight, selection=selection)
    manifest = dict(schema_version=1, strategy='candidate_head', head='attention_head_v1',
                    recipe=RECIPE, backbone=tr['backbone'], limits=tr['limits'],
                    feature_signature=tr['signature'], runtime=versions(),
                    files={name: file_digest(out/name) for name in ('head.safetensors', 'head.json')},
                    training=dict(objective='categorical_cross_entropy' if brier_weight == 0 else 'categorical_cross_entropy_plus_brier',
                                  brier_weight=brier_weight, brier_convention='sum_over_valid_candidates', frozen_backbone=True,
                                  selection=f'validation_{selection}_first_tie', seed=seed,
                                  features={str(Path(p).resolve()): file_digest(p) for p in (train_path, validation_path)},
                                  train=tr['provenance'], validation=va['provenance']))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return dict(artifact=str(out), best_val_top1=result['best_val_top1'])


def runtime_mismatch(expected, current=None):
    """Differences between the runtime that extracted an artifact's features and this one.

    The backend may differ: heads load on both, and the backbone's weights must match.
    Library versions within the same backend must match, since they change the features."""
    current = current or versions()
    if expected.get('backend', 'mlx') != current['backend']:
        return {}
    return {k: (v, current.get(k)) for k, v in expected.items() if k != 'backend' and current.get(k) != v}


class DecisionScorer:
    """Load a trained local decision artifact and its frozen backbone (MLX or PyTorch).

    ``model`` overrides the recorded backbone path when weights have moved; their
    contents must still match. Reuse an instance across requests to load once.
    """
    def __init__(self, artifact, model=None, batch_size=8, calibrator=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.manifest = read_artifact(artifact)
        self.artifact_id = file_digest(Path(artifact)/'manifest.json')
        self.calibrator = None
        if calibrator is not None:
            from .decision_probability import TemperatureCalibrator
            self.calibrator = TemperatureCalibrator.load(calibrator, self.artifact_id)
            for split in ('train', 'validation'):
                check_disjoint(self.calibrator.metadata['provenance'], self.manifest['training'][split])
        model = model or self.manifest['backbone']['path']
        actual = model_identity(model)
        if actual['fingerprint'] != self.manifest['backbone']['fingerprint']:
            raise ValueError('backbone/tokenizer identity differs from the trained artifact')
        if mismatch := runtime_mismatch(self.manifest['feature_signature']['runtime']):
            raise ValueError(f'feature extraction library versions differ from artifact {mismatch}; restore its pinned runtime or regenerate features and retrain')
        self.fx = extractor(model, batch_size)
        self.ops = compute.ops()
        self.head, config = self.ops.AttentionHead.load(str(Path(artifact)/'head.safetensors'))
        if self.head.hidden != self.fx.hidden:
            raise ValueError('head and backbone hidden sizes differ')
        self.artifact_id = file_digest(Path(artifact)/'manifest.json')

    def predict(self, *, state: str | dict | list, question: str,
                candidates: Mapping[str, str], request_id: str = 'request') -> dict:
        """Choose among candidate ID → text pairs; return a JSON-compatible dict.

        Probabilities are normalized over this request's candidates. They are
        not automatically calibrated. Candidate order is preserved in output.
        """
        if not isinstance(candidates, Mapping):
            raise ValueError('candidates must be a mapping of candidate IDs to texts')
        task = ChoiceTask.parse(dict(schema_version=1, id=request_id, state=state,
                                    question=question, candidates=[
                                        dict(id=k, text=v) for k, v in candidates.items()]))
        return self.score(task)

    def score(self, task: ChoiceTask | dict) -> dict:
        """Score a ChoiceTask or a native schema-version-1 request dictionary."""
        # Validate before loading the accelerator, including directly constructed
        # dataclasses whose constructors do not enforce the serialized contract.
        task = ChoiceTask.parse(task.to_dict() if isinstance(task, ChoiceTask) else task)
        started = time.perf_counter()
        ctx, opt = extract_task(self.fx, task, self.manifest['limits'])
        result = choice_result(task, head_logits(self.head, ctx, opt).tolist())
        result.update(artifact_id=self.artifact_id, seconds=time.perf_counter()-started)
        if self.calibrator is not None:
            result = self.calibrator.apply(result)
        return result


def head_logits(head, ctx, opt):
    """One example's logits (N,) from its cached (Lc, H) context and (N, H) option features."""
    return compute.ops(head.backend).forward(head, ctx[None], np.ones((1, len(ctx)), np.float32),
                                            opt[None], np.ones((1, len(opt)), np.float32))[0]
