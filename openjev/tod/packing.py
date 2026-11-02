"""Input layouts, heads and pooling for per-candidate scorers. Pure PyTorch, no SDK imports:
`openjev tod export --hf` copies this file into the exported model, so the SDK and the
exported model share one implementation.

Two layouts read a decision (a query, and the candidates to score against it):

    pairs    one sequence per candidate, tokenizer(query, candidate); the pooled state is the
             mean over the whole pair. Works with any model and attention implementation.
    packed   one sequence per decision: [cls] query [sep] | cand_1 [sep] | cand_2 [sep] ...
             The query attends to itself; each candidate span attends to the query and to
             itself, never to another candidate (a block mask), and its positions restart
             where the query ends, so a candidate's score does not change when other
             candidates are added, removed or reordered. The query is encoded once per
             decision, and each candidate is pooled over its own span. Needs a model that
             takes `position_ids` and a 4D attention mask (checked by `packing_supported`).

Heads map a pooled state to one logit: `linear`, or `mlp` (Linear-GELU-LayerNorm-Linear).
"""
from __future__ import annotations

import torch

LAYOUTS = ('packed', 'pairs')
HEADS = ('mlp', 'linear')


def make_head(kind, hidden):
    if kind == 'linear':
        return torch.nn.Linear(hidden, 1)
    if kind == 'mlp':
        return torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.LayerNorm(hidden),
                                   torch.nn.Linear(hidden, 1))
    raise ValueError(f'head must be one of {HEADS}')


def special_ids(tokenizer):
    """(cls, sep): the tokens that open the query and close each span (BOS/EOS for decoders)."""
    cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    return cls, sep


def pack(query_ids, candidate_ids, cls, sep, max_length):
    """One packed decision: dict(ids, positions, spans). The query keeps at most half of
    `max_length`; each candidate is cut so that query + candidate fits in `max_length`."""
    prefix = ([cls] if cls is not None else []) + list(query_ids)[:max_length // 2] + ([sep] if sep is not None else [])
    ids, positions, spans = list(prefix), list(range(len(prefix))), []
    room = max(1, max_length - len(prefix) - (sep is not None))
    for c in candidate_ids:
        span = list(c)[:room] + ([sep] if sep is not None else [])
        start = len(ids)
        ids += span
        positions += range(len(prefix), len(prefix) + len(span))
        spans.append((start, len(ids)))
    return dict(ids=ids, positions=positions, spans=spans, prefix=len(prefix))


def collate_packed(items, pad_id):
    """Tensors for a forward over packed decisions, plus `pool_index` (each token's
    candidate number, or `count` for tokens that are not in a candidate span)."""
    width = max(len(x['ids']) for x in items)
    count = sum(len(x['spans']) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    positions = torch.zeros((len(items), width), dtype=torch.long)
    mask = torch.zeros((len(items), 1, width, width), dtype=torch.bool)
    pool = torch.full((len(items), width), count, dtype=torch.long)
    n = 0
    for b, x in enumerate(items):
        length, p = len(x['ids']), x['prefix']
        ids[b, :length] = torch.tensor(x['ids'])
        positions[b, :length] = torch.tensor(x['positions'])
        mask[b, 0, :p, :p] = True
        for s, e in x['spans']:
            mask[b, 0, s:e, :p] = True
            mask[b, 0, s:e, s:e] = True
            pool[b, s:e] = n
            n += 1
        idx = torch.arange(length, width)
        mask[b, 0, idx, idx] = True  # padding attends to itself only (no empty rows)
    return dict(input_ids=ids, position_ids=positions, attention_mask=mask), dict(pool_index=pool, count=count)


def pooled_spans(hidden, pool_index, count):
    """Mean of each candidate's span: (count, hidden) in float32."""
    flat = hidden.float().reshape(-1, hidden.shape[-1])
    index = pool_index.reshape(-1)
    sums = torch.zeros((count + 1, flat.shape[1]), dtype=flat.dtype, device=flat.device).index_add_(0, index, flat)
    counts = torch.zeros(count + 1, dtype=flat.dtype, device=flat.device).index_add_(
        0, index, torch.ones_like(index, dtype=flat.dtype))
    return sums[:count] / counts[:count].clamp(min=1).unsqueeze(1)


def pooled_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)


class Scorer(torch.nn.Module):
    """trunk (a Hugging Face base model) + head: one logit per candidate."""

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head

    def forward(self, pool_index=None, count=None, **batch):
        hidden = self.trunk(**batch).last_hidden_state
        pooled = pooled_mean(hidden, batch['attention_mask']) if pool_index is None else \
            pooled_spans(hidden, pool_index, count)
        return self.head(pooled.to(next(self.head.parameters()).dtype)).squeeze(-1).float()


def encode(tokenizer, layout, query, candidates, max_length, max_chars):
    """Tokenize one decision for `layout`: a list of pair encodings, or one packed dict."""
    candidates = [c[:max_chars] for c in candidates]
    if layout == 'pairs':
        encoded = tokenizer([query] * len(candidates), candidates, truncation='longest_first', max_length=max_length)
        return [{k: encoded[k][i] for k in encoded.keys()} for i in range(len(candidates))]
    cls, sep = special_ids(tokenizer)
    q = tokenizer(query, add_special_tokens=False)['input_ids']
    cs = tokenizer(candidates, add_special_tokens=False)['input_ids'] if candidates else []
    return pack(q, cs, cls, sep, max_length)


def size(layout, item):
    """(sequences, longest) of one encoded decision, for budgeting padded tokens."""
    if layout == 'pairs':
        return len(item), max(len(r['input_ids']) for r in item)
    return 1, len(item['ids'])


def useful(layout, item):
    return sum(len(r['input_ids']) for r in item) if layout == 'pairs' else len(item['ids'])


def collate(tokenizer, layout, items):
    """(model inputs, pooling inputs, padded token count) for a forward over encoded decisions."""
    if layout == 'pairs':
        rows = [r for item in items for r in item]
        batch = tokenizer.pad(rows, return_tensors='pt')
        return dict(batch), {}, batch['input_ids'].numel()
    batch, pool = collate_packed(items, tokenizer.pad_token_id)
    return batch, pool, batch['input_ids'].numel()


@torch.inference_mode()
def score(model, tokenizer, layout, query, candidates, *, max_length=8192, max_chars=4000, temperature=1.0):
    """Probabilities (sigmoid of logit / temperature) for one decision's candidates."""
    item = encode(tokenizer, layout, query, candidates, max_length, max_chars)
    batch, pool, _ = collate(tokenizer, layout, [item])
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    pool = {k: v.to(device) if torch.is_tensor(v) else v for k, v in pool.items()}
    logits = model(**pool, **batch)
    return torch.sigmoid(logits / temperature).tolist()


def packing_supported(trunk, tolerance=1e-2):
    """Whether `trunk` honours the packed layout: a candidate's pooled state must not change
    when another candidate is added or the order changes."""
    try:
        vocab = trunk.config.get_text_config().vocab_size
        q, a, b = [5, 6, 7], [8, 9, 10, 11], [12, 13]
        device = next(trunk.parameters()).device
        states = []
        for cands in ([a, b], [b, a], [a]):
            batch, pool = collate_packed([pack(q, cands, 1 % vocab, 2 % vocab, 64)], 0)
            with torch.inference_mode():
                hidden = trunk(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state
            states.append(pooled_spans(hidden, pool['pool_index'].to(device), pool['count']))
        ref = states[0][0]
        scale = ref.abs().max().clamp(min=1e-6)
        return all(float((s - ref).abs().max() / scale) < tolerance for s in (states[1][1], states[2][0]))
    except (TypeError, ValueError, RuntimeError, IndexError):
        return False
