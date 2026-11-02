"""Fine-tune any Hugging Face model end to end as a per-candidate decision model (PyTorch).

`openjev tod train DATASET --method encoder --model HF_ID_OR_DIR` trains every weight of the
named model, encoders included (default answerdotai/ModernBERT-large), with a head that
gives one logit per candidate. A candidate attends to the question and state but never to
the other candidates. Two input layouts (see `packing`):

    packed (default)  one sequence per decision: the question + state is encoded once, each
                      candidate span attends to it and to itself (a block mask, positions
                      restarting per candidate) and is pooled over its own span. Needs
                      `sdpa` attention and a model that takes position ids and a 4D mask;
                      this is checked when the model loads, and training falls back to
                      pairs when the check fails.
    pairs             one (question + state, candidate) sequence per candidate, pooled over
                      the pair. Works with any model; with flash-attn installed on CUDA,
                      padding costs nothing (`--attention auto` picks flash_attention_2).

Kinds labelled with `Choice` train a softmax over their candidates (cross-entropy). Kinds
labelled with `Subset` train an independent sigmoid per candidate: BCE by default, or focal
loss (`--loss focal`), which keeps rare positives from being drowned out without rescaling
the probabilities the way class weights do.

Steps hold `--batch-tokens` tokens per GPU (default 32,768) or `--batch-size` decisions.
Within a step decisions are sorted by length before they are cut into forwards of at most
16,384 padded tokens, and the next steps are tokenized on a background thread while the
GPU trains. The trunk and the freshly initialised head have separate learning rates.

After training, `calibrate` fits a temperature per kind (Choice: NLL; Subset: Brier, over
candidates) and chooses the operating points of every Subset kind (see `operating`) on the
calibrated probabilities of the calibration split, and `evaluate` reports the test split.
`--model` may also be a model trained here: training then continues from its trunk and head
and the points are re-chosen on your calibration split.

CUDA trains in float32 weights with bfloat16 autocast, and several GPUs train under
`torchrun --nproc_per_node N -m openjev tod train ...`. MPS and CPU train in float32.

Model layout:

    model/tod.json                spec fingerprint, dataset, kinds and their targets, method "encoder"
    model/encoder.json            base model, layout, head, length limits, kind targets
    model/trunk/                  the fine-tuned model and its tokenizer (save_pretrained)
    model/head.safetensors        the per-candidate head (float32)
    model/spec.json               the spec, when the dataset carried one
    model/calibration.json        temperature per kind
    model/operating_points.json   keep thresholds per Subset kind
    model/evaluation.json         test report
    model/checkpoints/step-N/     rotating checkpoints, each a loadable model (`export` leaves them behind)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import random
import shutil
import sys
import threading
import time

import numpy as np

from . import operating, packing
from ..decision import digest, file_digest
from .schema import Choice, Prediction, Subset
from .spec import load_decisions, load_spec
from .train import bind_spec, dataset_spec, read_binding

DEFAULT_MODEL = 'answerdotai/ModernBERT-large'
CONFIG = 'encoder.json'
MAX_CHARS = 4000  # candidate texts are cut to this many characters
MAX_TOKENS = 16384  # padded tokens per forward
BATCH_TOKENS = 32768  # tokens per GPU per step, unless --batch-size is given


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


def probabilities(target, logits, temperature=1.0):
    """Calibrated probabilities: a sigmoid per candidate (Subset) or a softmax (Choice)."""
    z = np.asarray(logits, np.float64) / temperature
    if target == 'subset':
        return 1 / (1 + np.exp(-z))
    z = np.exp(z - z.max())
    return z / z.sum()


# ---------------------------------------------------------------------------- model
def is_encoder_model(model) -> bool:
    return (Path(model)/CONFIG).exists()


def _attention(attention, layout, device):
    if attention not in (None, 'auto'):
        return attention
    flash = device.type == 'cuda' and importlib.util.find_spec('flash_attn') is not None
    return 'flash_attention_2' if layout == 'pairs' and flash else 'sdpa'


class EncoderModel:
    """A Hugging Face trunk plus a per-candidate head.

    `source` is a Hugging Face id or directory (a new head is created; `layout` and `head`
    choose the input layout and head), or a model or checkpoint written here (trunk, head,
    layout and head type are loaded; models from 0.4.0 are pairs + linear). Inference runs
    bfloat16 on CUDA and float32 elsewhere; training keeps float32 weights (bfloat16
    autocast on CUDA)."""

    def __init__(self, source, *, device=None, trainable=False, max_length=None, max_chars=MAX_CHARS, seed=7,
                 gradient_checkpointing=False, layout=None, head=None, attention='auto', log=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from .. import compute
        log = log or (lambda message: None)
        self.torch = torch
        self.device = torch.device(device or compute.device())
        path = Path(str(source)).expanduser()
        self.config = json.loads((path/CONFIG).read_text()) if (path/CONFIG).exists() else None
        trunk = str(path/'trunk') if self.config else str(source)
        self.base = self.config['base'] if self.config else str(source)
        if self.config:
            saved = dict(layout=self.config.get('layout', 'pairs'), head=self.config.get('head', 'linear'))
            for name, value in dict(layout=layout, head=head).items():
                if value is not None and value != saved[name]:
                    raise ValueError(f'{source} was trained with {name} {saved[name]!r}; it cannot continue as {value!r}')
            layout, head = saved['layout'], saved['head']
        layout, self.head_type = layout or 'packed', head or 'mlp'
        if layout not in packing.LAYOUTS:
            raise ValueError(f'layout must be one of {packing.LAYOUTS}')
        bf16 = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        self.autocast = bf16 and trainable
        dtype = torch.bfloat16 if bf16 and not trainable else torch.float32
        self.tok = AutoTokenizer.from_pretrained(trunk)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        implementation = _attention(attention, layout, self.device)
        if layout == 'packed' and implementation != 'sdpa':
            raise ValueError(f'the packed layout needs sdpa attention, not {implementation}')
        try:
            model = AutoModel.from_pretrained(trunk, dtype=dtype, attn_implementation=implementation)
        except (ValueError, ImportError) as exc:
            if attention not in (None, 'auto'):
                raise
            log(f'{implementation} attention unavailable ({exc}); using the default')
            implementation = None
            model = AutoModel.from_pretrained(trunk, dtype=dtype)
        self.attention = implementation or model.config._attn_implementation
        if layout == 'packed' and not packing.packing_supported(model.to(self.device)):
            if self.config:
                raise ValueError(f'{source} uses the packed layout, which this model or transformers version '
                                 f'does not support')
            log(f'{self.base} does not support the packed layout (position ids and a 4D mask); using pairs')
            layout = 'pairs'
        self.layout = layout
        hidden = model.config.get_text_config().hidden_size
        torch.manual_seed(seed)
        head_module = packing.make_head(self.head_type, hidden)
        if self.config:
            from safetensors.torch import load_file
            head_module.load_state_dict(load_file(str(path/'head.safetensors')))
        self.net = packing.Scorer(model, head_module.float()).to(self.device)
        limit = self.tok.model_max_length if self.tok.model_max_length < 10**6 else 8192
        self.max_length = int(max_length or (self.config or {}).get('max_length') or min(limit, 8192))
        self.max_chars = int((self.config or {}).get('max_chars') or max_chars)
        self.net.eval()
        if trainable:
            self.net.float()
            if gradient_checkpointing and getattr(model, 'supports_gradient_checkpointing', False):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))

    # -- tokenizing and batching
    def encode(self, decisions, spec=None, tok=None):
        """One encoded item per decision (a list of pairs, or one packed sequence). Pass `tok`
        (a copy of the tokenizer) when encoding on another thread."""
        return [packing.encode(tok or self.tok, self.layout, query_text(d, spec), [c.text for c in d.candidates],
                               self.max_length, self.max_chars) for d in decisions]

    def chunks(self, items, max_tokens=MAX_TOKENS):
        """Whole decisions, sorted by length, grouped into forwards of at most `max_tokens` padded tokens."""
        sizes = [packing.size(self.layout, item) for item in items]
        group, rows, width = [], 0, 0
        for i in sorted(range(len(items)), key=lambda i: sizes[i][1]):
            n, longest = sizes[i]
            if group and (rows + n) * max(width, longest) > max_tokens:
                yield group
                group, rows, width = [], 0, 0
            group.append(i)
            rows, width = rows + n, max(width, longest)
        if group:
            yield group

    def batches(self, items, max_tokens=MAX_TOKENS, tok=None):
        """[(indices, model inputs, pooling inputs, useful tokens, padded tokens)] on the CPU."""
        out = []
        for group in self.chunks(items, max_tokens):
            batch, pool, padded = packing.collate(tok or self.tok, self.layout, [items[i] for i in group])
            out.append((group, batch, pool, sum(packing.useful(self.layout, items[i]) for i in group), padded))
        return out

    def forward(self, batch, pool, net=None):
        """Logits (n,) float32 for one collated batch."""
        torch = self.torch
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        pool = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in pool.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast):
            return (net or self.net)(**pool, **batch)

    def score(self, decisions, spec=None, max_tokens=MAX_TOKENS):
        """Raw logits per decision (numpy float64), many decisions per forward."""
        items = self.encode(decisions, spec)
        out = [None] * len(decisions)
        self.net.eval()
        with self.torch.inference_mode():
            for group, batch, pool, _, _ in self.batches(items, max_tokens):
                logits = self.forward(batch, pool).cpu().numpy().astype(np.float64)
                start = 0
                for i in group:
                    n = len(decisions[i].candidates)
                    out[i] = logits[start:start + n]
                    start += n
        return out

    def token_counts(self, decision):
        return [len(self.tok(c.text, add_special_tokens=False)['input_ids']) for c in decision.candidates]

    def tokenizer_digest(self):
        backend = getattr(self.tok, 'backend_tokenizer', None)
        return digest(backend.to_str() if backend is not None else repr(self.tok.get_vocab()))[:16]

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
        config = dict(base=self.base, layout=self.layout, head=self.head_type,
                      pooling='candidate-span-mean' if self.layout == 'packed' else 'masked-mean',
                      max_length=self.max_length, max_chars=self.max_chars, **(extra or {}))
        (out/CONFIG).write_text(json.dumps(config, indent=2) + '\n')


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


def fit_subset_temperature(logits, labels_, bounds=(0.05, 20.0)):
    """The temperature T minimising the Brier score of sigmoid(logit / T) over every candidate:
    a log-spaced grid, then golden-section refinement around the best grid point."""
    z = np.concatenate([np.asarray(l, np.float64) for l in logits])
    y = np.concatenate([np.asarray(l, np.float64) for l in labels_])

    def brier(log_t):
        return float(np.mean((1 / (1 + np.exp(-z / math.exp(log_t))) - y) ** 2))

    grid = np.linspace(math.log(bounds[0]), math.log(bounds[1]), 121)
    best = int(np.argmin([brier(g) for g in grid]))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if brier(a) < brier(b):
            hi = b
        else:
            lo = a
    candidates = [grid[best], (lo + hi) / 2, 0.0]
    return float(math.exp(min(candidates, key=brier)))


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


def token_lengths(model, rows, spec, dataset, world=1, rank=0, log=None):
    """Tokens per training decision in the model's layout, cached in DATASET/cache/ by
    dataset, tokenizer, layout and limits (rank 0 computes it under torchrun)."""
    key = digest(dict(rows=file_digest(Path(dataset)/'train.jsonl'), tokenizer=model.tokenizer_digest(),
                      layout=model.layout, max_length=model.max_length, max_chars=model.max_chars))[:16]
    path = Path(dataset)/'cache'/f'encoder-lengths-{key}.npy'
    if rank == 0 and not path.exists():
        (log or print)(f'counting tokens in {len(rows)} training rows (cached in {path.parent})')
        lengths = np.zeros(len(rows), np.int64)
        for start in range(0, len(rows), 1024):
            for i, item in enumerate(model.encode(rows[start:start + 1024], spec)):
                lengths[start + i] = packing.useful(model.layout, item)
        path.parent.mkdir(exist_ok=True)
        np.save(path.with_suffix('.tmp.npy'), lengths)
        os.replace(path.with_suffix('.tmp.npy'), path)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    return np.load(path)


def plan_steps(order, world, *, batch_size=None, batch_tokens=None, lengths=None):
    """Each step as one list of decision indices per rank. Token budgets fill a step with
    `batch_tokens * world` tokens and split it across ranks by tokens; decision budgets take
    `batch_size * world` decisions. A trailing partial step is dropped (unless it is the only one)."""
    steps, current, total = [], [], 0
    for i in order:
        current.append(i)
        total += lengths[i] if batch_tokens else 1
        if total >= (batch_tokens or batch_size) * world and len(current) >= world:
            steps.append(current)
            current, total = [], 0
    if current and not steps and len(current) >= world:
        steps.append(current)
    out = []
    for step in steps:
        if not batch_tokens:
            out.append([step[r::world] for r in range(world)])
            continue
        bins, loads = [[] for _ in range(world)], [0] * world
        for i in sorted(step, key=lambda i: -lengths[i]):
            r = loads.index(min(loads))
            bins[r].append(i)
            loads[r] += lengths[i]
        out.append(bins)
    return out


def _prefetch(produce, items, depth):
    """Yield produce(x) for x in items, computed `depth` ahead on a background thread."""
    results, stop = queue.Queue(maxsize=max(1, depth)), object()

    def work():
        try:
            for x in items:
                results.put(produce(x))
        except BaseException as exc:  # surfaced in the training loop
            results.put(exc)
        results.put(stop)

    threading.Thread(target=work, daemon=True).start()
    while (value := results.get()) is not stop:
        if isinstance(value, BaseException):
            raise value
        yield value


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


def finetune(dataset, out, *, source=DEFAULT_MODEL, epochs=1, learning_rate=3e-5, head_learning_rate=None,
             batch_size=None, batch_tokens=None, max_steps=None, loss='bce', focal_gamma=2.0, focal_alpha=0.5,
             brier=0.0, warmup=0.05, weight_decay=0.01, max_tokens=MAX_TOKENS, max_length=None, max_chars=MAX_CHARS,
             layout=None, head=None, attention='auto', gradient_checkpointing=False, prefetch=4, eval_every=None,
             eval_split='validation', eval_rows=2000, checkpoint_every=None, keep_checkpoints=2,
             targets=operating.TARGETS, survival='any', default_point=operating.DEFAULT_POINT, seed=7, log=None):
    """Fine-tune `source` end to end on a dataset directory written by `Dataset.write`, then
    calibrate on its calibration split and report its test split. Under torchrun every rank
    trains on its share of each step; rank 0 probes, checkpoints, calibrates and writes.
    `head_learning_rate` defaults to 3x `learning_rate`."""
    import torch
    if loss not in ('bce', 'focal'):
        raise ValueError("loss must be 'bce' or 'focal'")
    if batch_size and batch_tokens:
        raise ValueError('pass batch_size or batch_tokens, not both')
    if not batch_size:
        batch_tokens = batch_tokens or BATCH_TOKENS
    head_learning_rate = head_learning_rate or 3 * learning_rate
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
    torch.manual_seed(seed)
    started = time.perf_counter()
    model = EncoderModel(source, device=rank_device, trainable=True, max_length=max_length, max_chars=max_chars,
                         seed=seed, gradient_checkpointing=gradient_checkpointing, layout=layout, head=head,
                         attention=attention, log=say)
    resumed = model.config is not None
    net = model.net
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model.net, device_ids=[model.device.index] if model.device.type == 'cuda' else None)
    trunk_params = [p for p in model.net.trunk.parameters() if p.requires_grad]
    head_params = [p for p in model.net.head.parameters() if p.requires_grad]
    params = trunk_params + head_params
    say(f'{model.base} {"(continuing from " + str(source) + ") " if resumed else ""}loaded on {model.device} '
        f'x{world} in {time.perf_counter() - started:.0f}s: {sum(p.numel() for p in params) / 1e6:.1f}M parameters, '
        f'layout {model.layout}, head {model.head_type}, attention {model.attention}; '
        f'{len(train)} training rows, kinds {kind_targets}')
    lengths = token_lengths(model, train, spec, dataset, world, rank, say) if batch_tokens else None
    plan = []
    for epoch in range(epochs):
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        plan += [(epoch, step) for step in plan_steps(order, world, batch_size=batch_size, batch_tokens=batch_tokens,
                                                     lengths=lengths)]
    if not plan:
        raise ValueError(f'too few training rows for one step of {batch_tokens or batch_size} per GPU')
    total = min(max_steps, len(plan)) if max_steps else len(plan)
    plan = plan[:total]
    warm = max(1, int(warmup * total)) if warmup else 0
    optimizer = torch.optim.AdamW([dict(params=trunk_params, lr=learning_rate),
                                   dict(params=head_params, lr=head_learning_rate)], weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    settings = dict(epochs=epochs, learning_rate=learning_rate, head_learning_rate=head_learning_rate,
                    batch_size=batch_size, batch_tokens=batch_tokens, world_size=world, steps=total, loss=loss,
                    focal_gamma=focal_gamma if loss == 'focal' else None,
                    focal_alpha=focal_alpha if loss == 'focal' else None, brier=brier, warmup=warmup,
                    weight_decay=weight_decay, layout=model.layout, head=model.head_type, attention=model.attention,
                    gradient_checkpointing=gradient_checkpointing, max_length=model.max_length,
                    max_chars=model.max_chars, train_rows=len(train), seed=seed)
    say(f'{total} steps of {f"{batch_tokens} tokens" if batch_tokens else f"{batch_size} decisions"} per GPU')
    probe_rows = _rows(dataset, eval_split, set(kind_targets))
    if len(probe_rows) > eval_rows:
        probe_rows = random.Random(seed).sample(probe_rows, eval_rows)

    def write_model(path):
        model.save(path, extra=dict(kinds=kind_targets))
        bind_spec(path, spec_fp, method='encoder', source=str(source), fine_tuned_from=str(source) if resumed else None,
                  dataset=str(dataset.resolve()), kinds=sorted(kind_targets), kind_targets=kind_targets,
                  skipped={}, training=settings)

    worker_tok = copy.deepcopy(model.tok)  # fast tokenizers must not be shared across threads

    def prepare(entry):  # background thread: tokenize and collate this rank's share of a step
        epoch, ranks = entry
        mine = [train[i] for i in ranks[rank]]
        items = model.encode(mine, spec, worker_tok)
        return epoch, mine, model.batches(items, max_tokens, worker_tok)

    history, step, window = [], 0, dict(useful=0, padded=0, tokens=0)
    clock = time.perf_counter()
    for epoch, mine, batches in _prefetch(prepare, plan, prefetch):
        net.train()
        optimizer.zero_grad()
        step_loss = 0.
        for g, (group, batch, pool, used, padded) in enumerate(batches):
            sync = g == len(batches) - 1 or world == 1
            with (net.no_sync() if not sync else _null()):
                logits = model.forward(batch, pool, net)
                value, start = 0., 0
                for i in group:
                    d = mine[i]
                    n = len(d.candidates)
                    l = logits[start:start + n]
                    start += n
                    if target_of(d) == 'subset':
                        y = torch.tensor(labels(d), device=model.device)
                        value = value + subset_loss(l, y, loss=loss, gamma=focal_gamma, alpha=focal_alpha, brier=brier)
                    else:
                        value = value + torch.nn.functional.cross_entropy(
                            l[None], torch.tensor([int(labels(d))], device=model.device))
                (value / len(mine)).backward()
                step_loss += float(value.detach()) / len(mine)
            window['useful'] += used
            window['padded'] += padded
            window['tokens'] += used
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        if step % 50 == 0 or step == total:
            elapsed = time.perf_counter() - clock
            say(f'  step {step}/{total}: loss {step_loss:.4f}, lr {schedule.get_last_lr()[0]:.2e}, '
                f'{elapsed / step:.2f}s/step, {window["tokens"] / min(step, 50):.0f} tokens/step/GPU, '
                f'useful/padded {window["useful"] / max(1, window["padded"]):.2f}')
            window = dict(useful=0, padded=0, tokens=0)
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


def temperatures_of(model):
    path = Path(model)/'calibration.json'
    return json.loads(path.read_text())['temperatures'] if path.exists() else {}


def _questions(encoder, rows, spec, kind_targets, temperatures, max_tokens=MAX_TOKENS, tokens=False):
    logits = encoder.score(rows, spec, max_tokens)
    out = []
    for d, l in zip(rows, logits):
        target = kind_targets[d.kind]
        q = dict(decision=d, logits=l, p=probabilities(target, l, temperatures.get(d.kind, 1.0)),
                 relevant=labels(d).astype(bool) if target == 'subset' else None)
        if tokens:
            q['tokens'] = encoder.token_counts(d)
        out.append(q)
    return out


def calibrate(model, dataset, *, split='calibration', targets=operating.TARGETS, survival='any',
              default_point=operating.DEFAULT_POINT, encoder=None, max_tokens=MAX_TOKENS):
    """Fit a temperature per kind and choose each Subset kind's operating points (on the
    calibrated probabilities) on `split` of `dataset`, writing them into the model. Rerun on
    your own calibration split to re-choose them without retraining."""
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
    questions = _questions(encoder, rows, spec, kind_targets, {}, max_tokens)
    temperatures, points, counts = {}, {}, {}
    for kind, target in sorted(kind_targets.items()):
        mine = [q for q in questions if q['decision'].kind == kind]
        if not mine:
            continue
        if target == 'choice':
            temperatures[kind] = fit_temperature([q['logits'] for q in mine],
                                                 [int(labels(q['decision'])) for q in mine])['temperature']
            continue
        temperatures[kind] = fit_subset_temperature([q['logits'] for q in mine], [labels(q['decision']) for q in mine])
        calibrated = [(probabilities('subset', q['logits'], temperatures[kind]), q['relevant']) for q in mine]
        points[kind] = operating.choose_points(calibrated, targets, survival)
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


def calibration_error(questions, bins=15):
    """Expected calibration error and Brier score over every candidate."""
    p = np.concatenate([q['p'] for q in questions])
    y = np.concatenate([q['relevant'] for q in questions]).astype(np.float64)
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[edges == b].mean() - y[edges == b].mean()) * (edges == b).sum()
              for b in range(bins) if (edges == b).any()) / len(p)
    return dict(calibration_error=float(ece), brier=float(np.mean((p - y) ** 2)))


def evaluate(model, dataset, *, split='test', out=None, probabilities_path=None, refit=False, source_sep=':',
             match_survival=None, encoder=None, max_tokens=MAX_TOKENS):
    """Report `split` of `dataset`: per Choice kind accuracy and confidence; per Subset kind the
    calibration error, and set metrics at every stored operating point (answer and
    all-relevant survival, share of candidate tokens kept, candidates kept, the fixed top-k
    prefix needed for the same survival), overall and per source (the decision id up to
    `source_sep`, e.g. `nq:123`). With `match_survival` X, also at the tau and rho that reach
    survival X on this split itself (`tau_matched`, `rho_matched`: for comparing methods at
    equal survival).

    `dataset` may be evaluation-only (e.g. an external set with only a test split). The report
    goes to `out` (default: the model's evaluation.json); `probabilities_path` saves
    per-candidate probabilities as JSON lines."""
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
    temperatures = temperatures_of(model)
    questions = _questions(encoder, rows, spec, kind_targets, temperatures, max_tokens, tokens=True)
    stored = operating.load(model) or {}
    report = {}
    for kind, target in sorted(kind_targets.items()):
        mine = [q for q in questions if q['decision'].kind == kind]
        if not mine:
            continue
        t = temperatures.get(kind, 1.0)
        if target == 'choice':
            report[kind] = dict(accuracy=float(np.mean([int(q['p'].argmax()) == int(labels(q['decision']))
                                                        for q in mine])),
                                mean_confidence=float(np.mean([q['p'].max() for q in mine])), n=len(mine),
                                temperature=t)
            continue
        survival = stored.get('survival', 'any')
        points = {name: (operating.parse_point(name)[0], value)
                  for name, value in stored.get('kinds', {}).get(kind, {}).items()}
        if match_survival is not None:
            for rule in operating.RULES:
                points[f'{rule}_matched'] = (rule, operating.choose([(q['p'], q['relevant']) for q in mine], rule,
                                                                    match_survival, survival))
        entry = dict(default=stored.get('default'), survival=survival, temperature=t, **calibration_error(mine),
                     **operating.metrics(mine, points, survival))
        if match_survival is not None:
            entry['match_survival'] = match_survival
        sources = sorted({s for s in (_source(q['decision'], source_sep) for q in mine) if s})
        if len(sources) > 1:
            entry['by_source'] = {s: operating.metrics([q for q in mine if _source(q['decision'], source_sep) == s],
                                                       points, survival) for s in sources}
        report[kind] = entry
    if probabilities_path:
        with Path(probabilities_path).open('w') as stream:
            for q in questions:
                d = q['decision']
                stream.write(json.dumps(dict(id=d.id, kind=d.kind, probabilities={c.id: float(x) for c, x in
                                                                                  zip(d.candidates, q['p'])},
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


# --------------------------------------------------------------------------- export
def export(model, out, *, check=64, tolerance=0.02, device=None, hf=False, log=None):
    """Write a shippable copy: bfloat16 trunk, float32 head, with tod.json, spec.json,
    calibration, operating points and evaluation, and no checkpoints. The copy is reloaded
    and must give the same probabilities (within `tolerance`) and decisions on up to `check`
    rows of the model's dataset.

    With `hf`, the copy is instead one self-contained Hugging Face folder (config.json with
    auto_map, model.safetensors, the tokenizer and modeling_openjev.py) that loads with
    `AutoModel.from_pretrained(out, trust_remote_code=True)` and scores with
    `model.score(tokenizer, query, candidates, kind)` / `model.keep(...)`, no SDK needed."""
    import torch
    log = log or (lambda message: print(message, flush=True))
    model, out = Path(model), Path(out)
    if not is_encoder_model(model):
        raise ValueError(f'{model}: export applies to models trained with --method encoder')
    if out.exists():
        raise ValueError(f'{out} already exists')
    binding = read_binding(model)
    targets = binding['kind_targets']
    sample = []
    if check and binding.get('dataset') and Path(binding['dataset']).exists():
        for split in ('test', 'validation', 'calibration'):
            sample = _rows(binding['dataset'], split, set(targets))
            if sample:
                sample = random.Random(0).sample(sample, min(check, len(sample)))
                break
    if check and not sample:
        log('no rows to check against (the training dataset is not here); exporting without a check')
    source = EncoderModel(model, device=device)
    spec = _spec_for(model)
    temperatures = temperatures_of(model)
    before = [probabilities(targets[d.kind], l, temperatures.get(d.kind, 1.0))
              for d, l in zip(sample, source.score(sample, spec))] if sample else []
    if hf:
        score_copy = _export_hf(source, model, out, temperatures)
    else:
        source.save(out, dtype=torch.bfloat16, extra=dict(exported_from=str(model), kinds=targets))
        for name in ('spec.json', 'calibration.json', operating.FILE, 'evaluation.json'):
            if (model/name).exists():
                shutil.copy(model/name, out/name)
        (out/'tod.json').write_text(json.dumps(dict(binding, exported_from=str(model)), indent=2) + '\n')

        def score_copy(rows):
            copy = EncoderModel(out, device=device)
            return [probabilities(targets[d.kind], l, temperatures.get(d.kind, 1.0))
                    for d, l in zip(rows, copy.score(rows, spec))]
    del source
    report = dict(model=str(out), bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    if sample:
        after = score_copy(sample)
        stored = operating.load(model)
        diff, same = 0.0, 0
        for d, pa, pb in zip(sample, before, after):
            pa, pb = np.asarray(pa), np.asarray(pb)
            if targets[d.kind] == 'subset' and stored and d.kind in stored['kinds']:
                rule = operating.parse_point(stored['default'])[0]
                value = stored['kinds'][d.kind][stored['default']]
                same += bool((operating.kept(pa, rule, value) == operating.kept(pb, rule, value)).all())
            elif targets[d.kind] == 'subset':
                same += bool(((pa >= .5) == (pb >= .5)).all())
            else:
                same += int(pa.argmax()) == int(pb.argmax())
            diff = max(diff, float(np.abs(pa - pb).max()))
        report.update(checked=len(sample), max_probability_difference=diff, same_decisions=same)
        if diff > tolerance:
            raise ValueError(f'exported model differs from the source by {diff:.4f} (> {tolerance}); see {out}')
    release()
    log(json.dumps(report))
    return report


HF_MODELING = '''

# ---------------------------------------------------------------- Hugging Face wrapper
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class OpenjevScorerConfig(PretrainedConfig):
    model_type = 'openjev_scorer'

    def __init__(self, base_config=None, head='mlp', layout='packed', sequence_tokens=8192, max_chars=4000,
                 temperatures=None, operating_points=None, kind_targets=None, **kwargs):
        super().__init__(**kwargs)
        self.base_config, self.head, self.layout = base_config or {}, head, layout
        # not max_length: transformers reserves that name for generation settings
        self.sequence_tokens, self.max_chars = sequence_tokens, max_chars
        self.temperatures, self.operating_points = temperatures or {}, operating_points or {}
        self.kind_targets = kind_targets or {}


class OpenjevScorer(PreTrainedModel):
    """A per-candidate scorer exported by `openjev tod export --hf`. `query` is the decision's
    question, a blank line, then its state (JSON with indent 1 unless it is a string)."""
    config_class = OpenjevScorerConfig
    base_model_prefix = 'scorer'
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        base = dict(config.base_config)
        trunk_config = AutoConfig.for_model(base.pop('model_type'), **base)
        trunk = AutoModel.from_config(trunk_config, attn_implementation='sdpa')
        self.scorer = Scorer(trunk, make_head(config.head, trunk_config.get_text_config().hidden_size))
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, **inputs):
        return self.scorer(**inputs)

    def score(self, tokenizer, query, candidates, kind=None):
        """Calibrated probability per candidate (sigmoid; for a Choice kind, a softmax)."""
        temperature = self.config.temperatures.get(kind, 1.0) if kind else 1.0
        if self.config.kind_targets.get(kind) == 'choice':
            logits = torch.logit(torch.tensor(score(self.scorer, tokenizer, self.config.layout, query, candidates,
                                                    max_length=self.config.sequence_tokens,
                                                    max_chars=self.config.max_chars)))
            return torch.softmax(logits / temperature, 0).tolist()
        return score(self.scorer, tokenizer, self.config.layout, query, candidates, max_length=self.config.sequence_tokens,
                     max_chars=self.config.max_chars, temperature=temperature)

    def keep(self, tokenizer, query, candidates, kind, point=None):
        """Indices of the candidates to keep at an operating point (default: the model's)."""
        points = self.config.operating_points
        name = point or points['default']
        value = points['kinds'][kind][name]
        p = torch.tensor(self.score(tokenizer, query, candidates, kind))
        keep = p >= (value if name.startswith('tau') else value * p.max())
        keep[int(p.argmax())] |= name.startswith('rho')
        return [i for i, k in enumerate(keep.tolist()) if k]


OpenjevScorerConfig.register_for_auto_class()
OpenjevScorer.register_for_auto_class('AutoModel')
'''


def _export_hf(source, model, out, temperatures):
    """Write `source` as a self-contained Hugging Face folder; returns a scorer for the check."""
    import importlib.util as util
    import tempfile
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp)/'modeling_openjev.py'
        code.write_text(Path(packing.__file__).read_text() + HF_MODELING)
        spec_ = util.spec_from_file_location('modeling_openjev', code)
        module = util.module_from_spec(spec_)
        sys.modules['modeling_openjev'] = module  # save_pretrained copies the class's source file
        spec_.loader.exec_module(module)
        base = source.net.trunk.config.to_dict()
        base.pop('_attn_implementation_autoset', None)
        config = module.OpenjevScorerConfig(
            base_config=base, head=source.head_type, layout=source.layout, sequence_tokens=source.max_length,
            max_chars=source.max_chars, temperatures=temperatures, operating_points=operating.load(model) or {},
            kind_targets=read_binding(model)['kind_targets'])
        wrapper = module.OpenjevScorer(config)
        wrapper.scorer.load_state_dict({k: v.float() for k, v in source.net.state_dict().items()})
        wrapper.scorer.trunk.to(torch.bfloat16)
        try:
            wrapper.save_pretrained(str(out))
        finally:
            sys.modules.pop('modeling_openjev', None)
    source.tok.save_pretrained(str(out))
    spec = _spec_for(model)
    device = source.device

    def score_copy(rows):
        from transformers import AutoModel, AutoTokenizer
        loaded = AutoModel.from_pretrained(str(out), trust_remote_code=True).to(device)
        loaded.scorer.float()
        tok = AutoTokenizer.from_pretrained(str(out))
        return [loaded.score(tok, query_text(d, spec), [c.text for c in d.candidates], d.kind) for d in rows]

    return score_copy


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
            temperatures = temperatures_of(model) if calibrated else {}
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
            probs = probabilities(self.targets[kind], logits, self.heads[kind]['temperature'])
            if self.targets[kind] == 'subset':
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
           'subset_loss', 'fit_subset_temperature', 'plan_steps', 'is_encoder_model', 'query_text', 'probabilities']
