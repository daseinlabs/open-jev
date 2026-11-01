"""Fine-tune any Hugging Face model end to end as a per-candidate decision model (PyTorch).

`openjev tod train DATASET --method encoder --model HF_ID_OR_DIR` trains every weight of the
named model, encoders included (default answerdotai/ModernBERT-large). Each candidate is read
as a pair, (question + state, candidate text), so it attends to the question and state but
never to the other candidates; a linear head on the masked mean of the final hidden states
gives one logit per candidate. Kinds labelled with `Choice` train a softmax over their
candidates (cross-entropy). Kinds labelled with `Subset` train an independent sigmoid per
candidate: BCE by default, or focal loss (`--loss focal`), which keeps rare positives from
being drowned out without rescaling the probabilities the way class weights do.

After training, `calibrate` fits a temperature per Choice kind and chooses the operating
points of every Subset kind (see `operating`) on the calibration split, and `evaluate`
reports the test split. `--model` may also be a model trained here: training then continues
from its trunk and head (a few thousand of your own rows, a short run, a low learning rate)
and the points are re-chosen on your calibration split.

CUDA trains in float32 weights with bfloat16 autocast, and several GPUs train under
`torchrun --nproc_per_node N -m openjev tod train ...`. MPS and CPU train in float32.

Model layout:

    model/tod.json                spec fingerprint, dataset, kinds and their targets, method "encoder"
    model/encoder.json            base model, pooling, length limits, kind targets
    model/trunk/                  the fine-tuned model and its tokenizer (save_pretrained)
    model/head.safetensors        the per-candidate head (float32)
    model/spec.json               the spec, when the dataset carried one
    model/calibration.json        temperature per Choice kind
    model/operating_points.json   keep thresholds per Subset kind
    model/evaluation.json         test report
    model/checkpoints/step-N/     rotating checkpoints, each a loadable model (`export` leaves them behind)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from . import operating
from .schema import Choice, Prediction, Subset
from .spec import load_decisions, load_spec
from .train import bind_spec, dataset_spec, read_binding

DEFAULT_MODEL = 'answerdotai/ModernBERT-large'
CONFIG = 'encoder.json'
MAX_CHARS = 4000  # candidate texts are cut to this many characters
MAX_TOKENS = 16384  # padded tokens per forward


# ------------------------------------------------------------------------ rendering
def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)


def query_text(decision, spec=None) -> str:
    """What every candidate of a decision is read against: the question, then the state
    (the spec's `context`: a ListSpec leaves its candidate list out)."""
    state = spec.context(decision.kind, decision.state) if spec is not None else decision.state
    return decision.question.strip() + '\n\n' + _text(state).strip()


def target_of(decision):
    return 'subset' if isinstance(decision.target, Subset) else 'choice' if isinstance(decision.target, Choice) else None


def labels(decision) -> np.ndarray:
    """Subset: 0/1 per candidate. Choice: the index of the chosen candidate."""
    ids = [c.id for c in decision.candidates]
    if isinstance(decision.target, Subset):
        keep = set(decision.target.ids)
        return np.array([i in keep for i in ids], np.float32)
    return np.array(ids.index(decision.target.choice_id))


# ---------------------------------------------------------------------------- model
def is_encoder_model(model) -> bool:
    return (Path(model)/CONFIG).exists()


class EncoderModel:
    """A Hugging Face trunk plus a per-candidate linear head.

    `source` is a Hugging Face id or directory (a fresh head is created), or a model or
    checkpoint written here (trunk and head are loaded). Inference runs bfloat16 on CUDA
    and float32 elsewhere; training keeps float32 weights (bfloat16 autocast on CUDA)."""

    def __init__(self, source, *, device=None, trainable=False, max_length=None, max_chars=MAX_CHARS, seed=7,
                 gradient_checkpointing=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from .. import compute
        self.torch = torch
        self.device = torch.device(device or compute.device())
        path = Path(str(source)).expanduser()
        self.config = json.loads((path/CONFIG).read_text()) if (path/CONFIG).exists() else None
        trunk = str(path/'trunk') if self.config else str(source)
        self.base = self.config['base'] if self.config else str(source)
        bf16 = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        self.autocast = bf16 and trainable
        dtype = torch.bfloat16 if bf16 and not trainable else torch.float32
        self.tok = AutoTokenizer.from_pretrained(trunk)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            model = AutoModel.from_pretrained(trunk, dtype=dtype, attn_implementation='sdpa')
        except (ValueError, ImportError):
            model = AutoModel.from_pretrained(trunk, dtype=dtype)
        hidden = model.config.get_text_config().hidden_size
        torch.manual_seed(seed)
        head = torch.nn.Linear(hidden, 1)
        if self.config:
            from safetensors.torch import load_file
            head.load_state_dict(load_file(str(path/'head.safetensors')))
        self.net = _Net(model, head.float()).to(self.device)
        limit = self.tok.model_max_length if self.tok.model_max_length < 10**6 else 8192
        self.max_length = int(max_length or (self.config or {}).get('max_length') or min(limit, 8192))
        self.max_chars = int((self.config or {}).get('max_chars') or max_chars)
        self.net.eval()
        if trainable:
            self.net.float()
            if gradient_checkpointing and getattr(model, 'supports_gradient_checkpointing', False):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))

    # -- tokenizing and batching
    def tokenize(self, decisions, spec=None):
        """Token ids per candidate, grouped by decision."""
        pairs = [(query_text(d, spec), c.text[:self.max_chars]) for d in decisions for c in d.candidates]
        if not pairs:
            return []
        encoded = self.tok([q for q, _ in pairs], [t for _, t in pairs], truncation='longest_first',
                           max_length=self.max_length)
        rows = [{k: encoded[k][i] for k in encoded.keys()} for i in range(len(pairs))]
        out, start = [], 0
        for d in decisions:
            out.append(rows[start:start + len(d.candidates)])
            start += len(d.candidates)
        return out

    def chunks(self, tokenized, max_tokens=MAX_TOKENS):
        """Group whole decisions into forwards of at most `max_tokens` padded tokens."""
        group, width = [], 0
        for i, rows in enumerate(tokenized):
            longest = max(len(r['input_ids']) for r in rows)
            size = sum(len(tokenized[j]) for j in group) + len(rows)
            if group and size * max(width, longest) > max_tokens:
                yield group
                group, width = [], 0
            group.append(i)
            width = max(width, longest)
        if group:
            yield group

    def forward(self, rows, net=None):
        """Logits (n,) float32 for a flat list of tokenized candidate pairs."""
        torch = self.torch
        batch = self.tok.pad(rows, return_tensors='pt')
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast):
            return (net or self.net)(**batch)

    def score(self, decisions, spec=None, max_tokens=MAX_TOKENS):
        """Logits per decision (numpy float64), many decisions packed per forward."""
        tokenized = self.tokenize(decisions, spec)
        out = [None] * len(decisions)
        self.net.eval()
        with self.torch.inference_mode():
            for group in self.chunks(tokenized, max_tokens):
                logits = self.forward([r for i in group for r in tokenized[i]]).cpu().numpy().astype(np.float64)
                start = 0
                for i in group:
                    out[i] = logits[start:start + len(tokenized[i])]
                    start += len(tokenized[i])
        return out

    def token_counts(self, decision):
        return [len(self.tok(c.text, add_special_tokens=False)['input_ids']) for c in decision.candidates]

    def save(self, out, *, dtype=None, extra=None):
        """Write trunk (+ tokenizer), head (float32), and encoder.json to `out`."""
        from safetensors.torch import save_file
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        trunk = self.net.trunk
        if dtype is not None:
            trunk = trunk.to(dtype)
        trunk.save_pretrained(str(out/'trunk'))
        self.tok.save_pretrained(str(out/'trunk'))
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.net.head.state_dict().items()},
                  str(out/'head.safetensors'))
        config = dict(base=self.base, pooling='masked-mean', max_length=self.max_length, max_chars=self.max_chars,
                      **(extra or {}))
        (out/CONFIG).write_text(json.dumps(config, indent=2) + '\n')


def _Net(trunk, head):
    import torch

    class Net(torch.nn.Module):
        def __init__(self, trunk, head):
            super().__init__()
            self.trunk, self.head = trunk, head

        def forward(self, attention_mask, **batch):
            hidden = self.trunk(attention_mask=attention_mask, **batch).last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    return Net(trunk, head)


def release():
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------- loss
def subset_loss(logits, y, *, loss='bce', gamma=2.0, alpha=0.5, brier=0.0):
    """Mean per-candidate loss. Focal: alpha_t (1 - p_t)^gamma BCE; gamma 0, alpha 0.5 is BCE / 2."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    if loss == 'focal':
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        bce = alpha_t * (1 - p_t) ** gamma * bce
    value = bce.mean()
    if brier:
        value = value + brier * ((torch.sigmoid(logits) - y) ** 2).mean()
    return value


# ------------------------------------------------------------------------- training
def _rows(dataset, split, kinds=None):
    path = Path(dataset)/f'{split}.jsonl'
    if not path.exists():
        return []
    return [d for d in load_decisions(path) if target_of(d) and (kinds is None or d.kind in kinds)]


def dataset_spec_object(dataset):
    path = Path(dataset)/'spec.json'
    return load_spec(path) if path.exists() else None


def _kind_targets(rows):
    targets = {}
    for d in rows:
        t = target_of(d)
        if targets.setdefault(d.kind, t) != t:
            raise ValueError(f'kind {d.kind!r} mixes Choice and Subset targets')
    return targets


def _distributed():
    """(rank, world, device) under torchrun; (0, 1, None) otherwise."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world == 1:
        return 0, 1, None
    import torch
    import torch.distributed as dist
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    return dist.get_rank(), world, f'cuda:{local}' if torch.cuda.is_available() else 'cpu'


def probe(model, rows, spec, kind_targets, max_tokens=MAX_TOKENS):
    """Mid-run counts at p = 0.5 for Subset kinds, accuracy for Choice kinds, and speed."""
    started = time.perf_counter()
    logits = model.score(rows, spec, max_tokens)
    seconds = time.perf_counter() - started
    report = dict(decisions=len(rows), decisions_per_second=round(len(rows) / max(seconds, 1e-9), 1))
    for kind, target in sorted(kind_targets.items()):
        pairs = [(d, l) for d, l in zip(rows, logits) if d.kind == kind]
        if not pairs:
            continue
        if target == 'subset':
            counts = dict(kept_right=0, wrongly_kept=0, wrongly_dropped=0, dropped_right=0)
            for d, l in pairs:
                y, keep = labels(d).astype(bool), l >= 0
                counts['kept_right'] += int((y & keep).sum())
                counts['wrongly_kept'] += int((~y & keep).sum())
                counts['wrongly_dropped'] += int((y & ~keep).sum())
                counts['dropped_right'] += int((~y & ~keep).sum())
            report[kind] = counts
        else:
            report[kind] = dict(accuracy=float(np.mean([int(l.argmax()) == int(labels(d)) for d, l in pairs])))
    return report


def finetune(dataset, out, *, source=DEFAULT_MODEL, epochs=1, learning_rate=2e-5, batch_size=8, max_steps=None,
             loss='bce', focal_gamma=2.0, focal_alpha=0.5, brier=0.0, warmup=0.05, weight_decay=0.01,
             max_tokens=MAX_TOKENS, max_length=None, max_chars=MAX_CHARS, eval_every=None, eval_split='validation',
             eval_rows=2000, checkpoint_every=None, keep_checkpoints=2, targets=operating.TARGETS, survival='any',
             default_point=operating.DEFAULT_POINT, gradient_checkpointing=True, seed=7, log=None):
    """Fine-tune `source` end to end on a dataset directory written by `Dataset.write`, then
    calibrate on its calibration split and report its test split. Under torchrun every rank
    trains on its share of each batch; rank 0 probes, checkpoints, calibrates and writes."""
    import torch
    if loss not in ('bce', 'focal'):
        raise ValueError("loss must be 'bce' or 'focal'")
    rank, world, rank_device = _distributed()
    log = log or (lambda message: print(message, flush=True))
    say = log if rank == 0 else (lambda message: None)
    dataset, out = Path(dataset), Path(out)
    if rank == 0 and out.exists():
        raise ValueError(f'{out} already exists')
    spec_fp, spec = dataset_spec(dataset), dataset_spec_object(dataset)
    train = _rows(dataset, 'train')
    if not train:
        raise ValueError(f'{dataset}: no labelled (Choice or Subset) rows in train')
    kind_targets = _kind_targets(train)
    if loss == 'focal' and 'subset' not in kind_targets.values():
        raise ValueError('--loss focal applies to Subset kinds; this dataset has none')
    if not _rows(dataset, 'calibration', set(kind_targets)):
        raise ValueError(f'{dataset}: no labelled rows in calibration (needed for temperatures and operating points)')
    rng = random.Random(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    model = EncoderModel(source, device=rank_device, trainable=True, max_length=max_length, max_chars=max_chars,
                         seed=seed, gradient_checkpointing=gradient_checkpointing)
    resumed = model.config is not None
    net = model.net
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model.net, device_ids=[model.device.index] if model.device.type == 'cuda' else None)
    params = [p for p in model.net.parameters() if p.requires_grad]
    say(f'{model.base} {"(continuing from " + str(source) + ") " if resumed else ""}loaded on {model.device} '
        f'x{world} in {time.perf_counter() - started:.0f}s; {sum(p.numel() for p in params) / 1e6:.1f}M parameters, '
        f'{len(train)} training rows, kinds {kind_targets}')
    probe_rows = _rows(dataset, eval_split, set(kind_targets))
    if len(probe_rows) > eval_rows:
        probe_rows = random.Random(seed).sample(probe_rows, eval_rows)
    per_step = batch_size * world
    steps_per_epoch = max(1, len(train) // per_step)
    total = min(max_steps, epochs * steps_per_epoch) if max_steps else epochs * steps_per_epoch
    warm = max(1, int(warmup * total)) if warmup else 0
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    settings = dict(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, world_size=world, steps=total,
                    loss=loss, focal_gamma=focal_gamma if loss == 'focal' else None,
                    focal_alpha=focal_alpha if loss == 'focal' else None, brier=brier, warmup=warmup,
                    weight_decay=weight_decay, max_length=model.max_length, max_chars=model.max_chars,
                    train_rows=len(train), seed=seed)
    say(f'{total} steps of {per_step} decisions ({steps_per_epoch} per epoch)')

    def write_model(path):
        model.save(path, extra=dict(kinds=kind_targets))
        bind_spec(path, spec_fp, method='encoder', source=str(source), fine_tuned_from=str(source) if resumed else None,
                  dataset=str(dataset.resolve()), kinds=sorted(kind_targets), kind_targets=kind_targets,
                  skipped={}, training=settings)

    history, step, seen = [], 0, 0
    clock = time.perf_counter()
    for epoch in range(epochs):
        order = list(train)
        random.Random(seed + epoch).shuffle(order)
        for s in range(steps_per_epoch):
            if step >= total:
                break
            mine = order[s * per_step:(s + 1) * per_step][rank::world]
            tokenized = model.tokenize(mine, spec)
            groups = list(model.chunks(tokenized, max_tokens))
            net.train()
            optimizer.zero_grad()
            step_loss = 0.
            for g, group in enumerate(groups):
                sync = g == len(groups) - 1 or world == 1
                context = net.no_sync() if not sync else _null()
                with context:
                    logits = model.forward([r for i in group for r in tokenized[i]], net)
                    value, start = 0., 0
                    for i in group:
                        d, n = mine[i], len(tokenized[i])
                        l = logits[start:start + n]
                        start += n
                        if target_of(d) == 'subset':
                            y = torch.tensor(labels(d), device=model.device)
                            value = value + subset_loss(l, y, loss=loss, gamma=focal_gamma, alpha=focal_alpha,
                                                        brier=brier)
                        else:
                            value = value + torch.nn.functional.cross_entropy(
                                l[None], torch.tensor([int(labels(d))], device=model.device))
                    (value / len(mine)).backward()
                    step_loss += float(value.detach()) / len(mine)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            seen += len(mine) * world
            if step % 50 == 0 or step == total:
                say(f'  step {step}/{total}: loss {step_loss:.4f}, lr {schedule.get_last_lr()[0]:.2e}, '
                    f'{(time.perf_counter() - clock) / step:.2f}s/step')
            if rank == 0 and eval_every and (step % eval_every == 0 or step == total) and probe_rows:
                entry = dict(step=step, epoch=epoch + 1, loss=step_loss,
                             **probe(model, probe_rows, spec, kind_targets, max_tokens))
                history.append(entry)
                say(json.dumps(entry))
            if rank == 0 and checkpoint_every and step % checkpoint_every == 0 and step < total:
                write_model(out/'checkpoints'/f'step-{step}')
                kept_ = sorted((out/'checkpoints').glob('step-*'), key=lambda p: int(p.name.split('-')[1]))
                for old in kept_[:-keep_checkpoints] if keep_checkpoints else kept_:
                    shutil.rmtree(old)
    settings['history'] = history
    settings['seconds'] = round(time.perf_counter() - clock, 1)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
        if rank != 0:
            return dict(rank=rank)
    write_model(out)
    model.net.eval()
    log('calibrating')
    report = dict(calibration=calibrate(out, dataset, targets=targets, survival=survival,
                                        default_point=default_point, encoder=model))
    if _rows(dataset, 'test', set(kind_targets)):
        log('evaluating on test')
        report.update(evaluate(out, dataset, encoder=model))
    del model, net, optimizer
    release()
    return dict(model=str(out), **report)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------- calibration and metrics
def _spec_for(model, dataset=None):
    for root in (Path(model), Path(dataset) if dataset else None):
        if root is not None and (root/'spec.json').exists():
            return load_spec(root/'spec.json')
    return None


def _questions(encoder, rows, spec, max_tokens=MAX_TOKENS, tokens=False):
    logits = encoder.score(rows, spec, max_tokens)
    out = []
    for d, l in zip(rows, logits):
        q = dict(decision=d, logits=l, p=1 / (1 + np.exp(-l)), relevant=labels(d).astype(bool))
        if tokens:
            q['tokens'] = encoder.token_counts(d)
        out.append(q)
    return out


def calibrate(model, dataset, *, split='calibration', targets=operating.TARGETS, survival='any',
              default_point=operating.DEFAULT_POINT, encoder=None, max_tokens=MAX_TOKENS):
    """Fit a temperature per Choice kind and choose each Subset kind's operating points on
    `split` of `dataset`, writing them into the model. Rerun on your own calibration split
    to re-choose the points without retraining."""
    from ..decision_probability import fit_temperature
    model = Path(model)
    binding = read_binding(model)
    if binding['spec'] != dataset_spec(dataset):
        raise ValueError('model and dataset were built with different DecisionSpecs')
    kind_targets = binding['kind_targets']
    rows = _rows(dataset, split, set(kind_targets))
    if not rows:
        raise ValueError(f'{dataset}: no labelled rows in {split} for kinds {sorted(kind_targets)}')
    own = encoder is None
    encoder = encoder or EncoderModel(model)
    spec = _spec_for(model, dataset)
    questions = _questions(encoder, rows, spec, max_tokens)
    temperatures, points, counts = {}, {}, {}
    for kind, target in sorted(kind_targets.items()):
        mine = [q for q in questions if q['decision'].kind == kind]
        if not mine:
            continue
        if target == 'choice':
            temperatures[kind] = fit_temperature([q['logits'] for q in mine],
                                                 [int(labels(q['decision'])) for q in mine])['temperature']
        else:
            points[kind] = operating.choose_points([(q['p'], q['relevant']) for q in mine], targets, survival)
            counts[kind] = dict(questions=len(mine), with_relevant=sum(bool(q['relevant'].any()) for q in mine),
                                split=split, dataset=str(Path(dataset).resolve()))
    (model/'calibration.json').write_text(json.dumps(dict(temperatures=temperatures), indent=2) + '\n')
    report = dict(temperatures=temperatures)
    if points:
        report['operating_points'] = operating.save(model, points, default=default_point, survival=survival,
                                                     targets=targets, counts=counts)
    if own:
        del encoder
        release()
    return report


def _source(decision, sep):
    return decision.id.split(sep, 1)[0] if sep and sep in decision.id else None


def evaluate(model, dataset, *, split='test', out=None, probabilities=None, refit=False, source_sep=':',
             encoder=None, max_tokens=MAX_TOKENS):
    """Report `split` of `dataset`: per Choice kind accuracy and confidence; per Subset kind the
    set metrics at every stored operating point (answer and all-relevant survival, share of
    candidate tokens kept, candidates kept, the fixed top-k prefix needed for the same
    survival), overall and per source (the decision id up to `source_sep`).

    `dataset` may be evaluation-only (e.g. an external set with only a test split). The report
    goes to `out` (default: the model's evaluation.json); `probabilities` saves per-candidate
    probabilities as JSON lines."""
    model = Path(model)
    binding = read_binding(model)
    if binding['spec'] != dataset_spec(dataset):
        raise ValueError('model and dataset were built with different DecisionSpecs')
    own = encoder is None
    encoder = encoder or EncoderModel(model)
    if refit:
        calibrate(model, dataset, encoder=encoder, max_tokens=max_tokens)
    kind_targets = binding['kind_targets']
    rows = _rows(dataset, split, set(kind_targets))
    if not rows:
        raise ValueError(f'{dataset}: no labelled rows in {split} for kinds {sorted(kind_targets)}')
    spec = _spec_for(model, dataset)
    questions = _questions(encoder, rows, spec, max_tokens, tokens=True)
    temperatures = json.loads((model/'calibration.json').read_text())['temperatures'] \
        if (model/'calibration.json').exists() else {}
    stored = operating.load(model) or {}
    report = {}
    for kind, target in sorted(kind_targets.items()):
        mine = [q for q in questions if q['decision'].kind == kind]
        if not mine:
            continue
        if target == 'choice':
            t = temperatures.get(kind, 1.0)
            probs = [_softmax(q['logits'], t) for q in mine]
            report[kind] = dict(accuracy=float(np.mean([int(p.argmax()) == int(labels(q['decision']))
                                                        for p, q in zip(probs, mine)])),
                                mean_confidence=float(np.mean([p.max() for p in probs])), n=len(mine), temperature=t)
            continue
        points = {name: operating.parse_point(name)[:1] + (value,)
                  for name, value in stored.get('kinds', {}).get(kind, {}).items()}
        survival = stored.get('survival', 'any')
        entry = dict(default=stored.get('default'), survival=survival, **operating.metrics(mine, points, survival))
        sources = sorted({s for s in (_source(q['decision'], source_sep) for q in mine) if s})
        if len(sources) > 1:
            entry['by_source'] = {s: operating.metrics([q for q in mine if _source(q['decision'], source_sep) == s],
                                                       points, survival) for s in sources}
        report[kind] = entry
    if probabilities:
        with Path(probabilities).open('w') as stream:
            for q in questions:
                d = q['decision']
                p = q['p'] if kind_targets[d.kind] == 'subset' else _softmax(q['logits'], temperatures.get(d.kind, 1.0))
                stream.write(json.dumps(dict(id=d.id, kind=d.kind, probabilities={c.id: float(x) for c, x in
                                                                                  zip(d.candidates, p)},
                                             tokens=q['tokens'], target=_target_dict(d))) + '\n')
    result = {split: report, 'dataset': str(Path(dataset).resolve())}
    path = Path(out) if out else model/'evaluation.json'
    previous = json.loads(path.read_text()) if path.exists() and not out else {}
    path.write_text(json.dumps(dict(previous, **result), indent=2) + '\n')
    if own:
        del encoder
        release()
    return result


def _target_dict(decision):
    from .schema import target_to_dict
    return target_to_dict(decision.target)


def _softmax(x, t=1.0):
    z = np.exp((x - x.max()) / t)
    return z / z.sum()


# --------------------------------------------------------------------------- export
def export(model, out, *, check=64, tolerance=0.02, device=None, log=None):
    """Write a shippable copy: bfloat16 trunk, float32 head, with tod.json, spec.json,
    calibration, operating points and evaluation, and no checkpoints. The copy is reloaded
    and must give the same probabilities (within `tolerance`) and keep-sets on up to `check`
    rows of the model's dataset."""
    import torch
    log = log or (lambda message: print(message, flush=True))
    model, out = Path(model), Path(out)
    if not is_encoder_model(model):
        raise ValueError(f'{model}: export applies to models trained with --method encoder')
    if out.exists():
        raise ValueError(f'{out} already exists')
    binding = read_binding(model)
    sample = []
    if check and binding.get('dataset') and Path(binding['dataset']).exists():
        for split in ('test', 'validation', 'calibration'):
            sample = _rows(binding['dataset'], split, set(binding['kind_targets']))
            if sample:
                sample = random.Random(0).sample(sample, min(check, len(sample)))
                break
    if check and not sample:
        log('no rows to check against (the training dataset is not here); exporting without a check')
    source = EncoderModel(model, device=device)
    spec = _spec_for(model)
    before = source.score(sample, spec) if sample else []
    source.save(out, dtype=torch.bfloat16, extra=dict(exported_from=str(model)))
    del source
    for name in ('spec.json', 'calibration.json', operating.FILE, 'evaluation.json'):
        if (model/name).exists():
            shutil.copy(model/name, out/name)
    (out/'tod.json').write_text(json.dumps(dict(binding, exported_from=str(model)), indent=2) + '\n')
    report = dict(model=str(out), bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    if sample:
        copy = EncoderModel(out, device=device)
        after = copy.score(sample, spec)
        targets = binding['kind_targets']
        diff, same = 0.0, 0
        stored = operating.load(model)
        for d, a, b in zip(sample, before, after):
            if targets[d.kind] == 'subset':
                pa, pb = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
                if stored and d.kind in stored['kinds']:
                    rule, _ = operating.parse_point(stored['default'])
                    value = stored['kinds'][d.kind][stored['default']]
                    same += bool((operating.kept(pa, rule, value) == operating.kept(pb, rule, value)).all())
                else:
                    same += bool(((pa >= .5) == (pb >= .5)).all())
            else:
                pa, pb = _softmax(a), _softmax(b)
                same += int(pa.argmax()) == int(pb.argmax())
            diff = max(diff, float(np.abs(pa - pb).max()))
        report.update(checked=len(sample), max_probability_difference=diff, same_decisions=same)
        del copy
        if diff > tolerance:
            raise ValueError(f'exported model differs from the source by {diff:.4f} (> {tolerance}); see {out}')
    release()
    log(json.dumps(report))
    return report


# -------------------------------------------------------------------------- serving
def policy(model, spec, *, device=None, calibrated=True, encoder=None):
    """An encoder model acting through a spec (the HeadPolicy interface, plus Subset kinds
    and `decide_many`)."""
    from .rl import HeadPolicy

    class EncoderPolicy(HeadPolicy):
        def __init__(self):
            self.model, self.spec = Path(model), spec
            self.binding = read_binding(model)
            if spec is not None:
                spec.require(self.binding['spec'])
            self.encoder = encoder or EncoderModel(model, device=device)
            self.targets = self.binding['kind_targets']
            temperatures = {}
            if calibrated and (self.model/'calibration.json').exists():
                temperatures = json.loads((self.model/'calibration.json').read_text())['temperatures']
            self.points = operating.load(model) or {}
            self.heads = {kind: dict(temperature=temperatures.get(kind, 1.0), target=target)
                          for kind, target in self.targets.items()}

        def logits(self, decision):
            if self.targets.get(decision.kind) == 'subset':
                raise ValueError(f'{decision.kind} is a Subset kind; use decide or decide_many')
            return decision, None, self.encoder.score([decision], self.spec)[0]

        def decide(self, state, kind, threshold=None, id='request', point=None) -> Prediction:
            """One decision in one forward. Subset kinds keep the candidates the model's default
            operating point keeps, or `point` ('rho_98', ...), or p >= `threshold` when given."""
            return self.decide_many([(state, kind)], threshold=threshold, point=point, ids=[id])[0]

        def decide_many(self, requests, *, threshold=None, point=None, ids=None, max_tokens=MAX_TOKENS):
            """[(state, kind), ...] -> Predictions, packing many decisions per forward (GPU)."""
            ids = ids or [f'request-{i}' for i in range(len(requests))]
            decisions = [self.spec.decision(i, kind, state) for i, (state, kind) in zip(ids, requests)]
            scored = [d for d in decisions if d.kind in self.heads]
            logits = dict(zip((d.id for d in scored), self.encoder.score(scored, self.spec, max_tokens)))
            return [self._prediction(d, logits.get(d.id), threshold, point) for d in decisions]

        def _prediction(self, decision, logits, threshold, point):
            kind = decision.kind
            if logits is None:
                return Prediction(decision.id, kind, decision.candidates[0].id, 0.0, {}, True, 'no_model')
            limit = threshold[kind] if isinstance(threshold, dict) else threshold
            if self.targets[kind] == 'subset':
                probs = 1 / (1 + np.exp(-logits))
                if limit is not None:
                    rule, value, name = 'tau', float(limit), None
                else:
                    name = point or self.points.get('default')
                    values = self.points.get('kinds', {}).get(kind, {})
                    if name not in values:
                        raise ValueError(f'{kind}: no operating point {name!r} in the model '
                                         f'(have {sorted(values)}); pass point= or threshold=')
                    rule, value = operating.parse_point(name)[0], values[name]
                mask = operating.kept(probs, rule, value)
                top = int(probs.argmax())
                kept = [c for c, m in zip(decision.candidates, mask) if m]
                return Prediction(decision.id, kind, decision.candidates[top].id, float(probs[top]),
                                  {c.id: float(p) for c, p in zip(decision.candidates, probs)}, False, None,
                                  decision.candidates[top].payload, tuple(c.id for c in kept),
                                  tuple(c.payload for c in kept), name or f'tau={value}')
            probs = _softmax(logits, self.heads[kind]['temperature'])
            index = int(probs.argmax())
            choice = decision.candidates[index]
            reason = ('mandatory' if choice.id in self.spec.kind(kind).mandatory_defer else
                      'low_confidence' if probs[index] < (limit or 0.0) else None)
            return Prediction(decision.id, kind, choice.id, float(probs[index]),
                              {c.id: float(p) for c, p in zip(decision.candidates, probs)},
                              reason is not None, reason, choice.payload)

        def trainer(self, anchor, anchor_weight, learning_rate, rng):
            raise NotImplementedError('improve/reinforce do not support encoder models yet; '
                                      'fine-tune on new rows with `openjev tod train --model MODEL`')

    return EncoderPolicy()


__all__ = ['DEFAULT_MODEL', 'EncoderModel', 'finetune', 'calibrate', 'evaluate', 'export', 'policy',
           'subset_loss', 'is_encoder_model', 'query_text']
