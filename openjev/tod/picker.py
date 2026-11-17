"""Fine-tune the TOD picker on your decisions (PyTorch: CUDA, ROCm, MPS).

TOD (`parsecai/tod`, code in daseinlabs/tod) is Gemma 4 12B with a stage-2 LoRA. It reads a
decision as a lettered multiple-choice prompt and scores the answer letters. With up to 16
options that letter read *is* TOD's answer (its stage-1 shortlist keeps every option), so
fine-tuning continues training the stage-2 LoRA with cross-entropy over the letters, the
way it was trained. Decisions may have up to 26 candidates.

The prompt below is copied from `tod.eval.letter_logit` (prompt "cygnet.v1", option style
"label_desc") and must stay byte-identical to it: the adapter was trained on exactly this text.

Model layout:

    model/tod.json             spec fingerprint, dataset, kinds, method "tod"
    model/picker.json          TOD's manifest; adapters.stage2 is the fine-tuned adapter
    model/stage2_adapter/      the LoRA (PEFT)
    model/calibration.json     temperature per kind, fitted on the calibration split
    model/evaluation.json      per-kind test report (and zero-shot TOD, when measured)
"""
from __future__ import annotations

import json
import random
import shutil
import string
import time
from pathlib import Path

import numpy as np

from .schema import Choice
from .spec import load_decisions
from .train import SPLITS, bind_spec, dataset_spec, read_binding

TOD_REPO = 'parsecai/tod'
LETTERS = list(string.ascii_uppercase)
SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a "
    "question about it, and a fixed list of lettered options. You reply with that option's "
    "LETTER and nothing else - a single character, no words, no punctuation, no explanation."
)


# ------------------------------------------------------------------ prompt (from tod)
def record(decision) -> dict:
    """A Decision as a TOD record: candidate ids are the labels, their texts the criteria."""
    return dict(id=decision.id, state=decision.state, labels=[c.id for c in decision.candidates],
                question=dict(type='choice', instructions=decision.question,
                              criteria={c.id: c.text for c in decision.candidates}))


def _state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def render_options(rec) -> list[str]:
    criteria = rec['question'].get('criteria') or {}
    lines = []
    for i, label in enumerate(str(x) for x in rec['labels']):
        desc = criteria.get(label)
        d = str(desc).strip() if desc else ''
        if d and d != label and not d.lower().startswith(label.lower()):
            lines.append(f'{LETTERS[i]}. {label}: {d}')
        elif d:
            lines.append(f'{LETTERS[i]}. {d}')
        else:
            lines.append(f'{LETTERS[i]}. {label}')
    return lines


def build_user_message(rec) -> str:
    return (SYSTEM_PROMPT + "\n\n" + _state_text(rec['state']).strip()
            + "\n\n" + str(rec['question']['instructions']).strip() + "\n\n"
            + "Options:\n" + "\n".join(render_options(rec)) + "\n\n"
            + "Answer with the letter of exactly one option, and nothing else:")


# ------------------------------------------------------------------------- model
def _fetch(source) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(source), allow_patterns=['picker.json', 'stage2_adapter/*', 'LICENSE', 'NOTICE']))


class TodModel:
    """TOD's base model plus a stage-2 adapter, scoring decisions by their answer letters.

    `source` is a TOD bundle (default `parsecai/tod`) or a model directory written by
    `finetune`. The base (google/gemma-4-12B-it, Apache-2.0) downloads on first use."""

    def __init__(self, source=TOD_REPO, *, device=None, dtype=None, trainable=False, max_tokens=8192):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .. import compute
        self.torch, self.source = torch, _fetch(source)
        self.manifest = json.loads((self.source/'picker.json').read_text())
        base = self.manifest['base']['id']
        self.device = torch.device(device or compute.device())
        if dtype is None:
            # Gemma overflows float16; bfloat16 where the device has it, else float32.
            bf16 = self.device.type == 'mps' or (self.device.type == 'cuda' and torch.cuda.is_bf16_supported())
            dtype = torch.bfloat16 if bf16 else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation='sdpa').to(self.device)
        adapter = self.source/(self.manifest['adapters'].get('stage2') or 'stage2_adapter')
        self.model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
        self.model.eval()
        self.trainable = False
        if trainable:
            self.make_trainable()
        self.max_tokens, self._keep_last = max_tokens, None
        self.letter_ids = []
        for letter in LETTERS:  # "A" and " A": logsumexp over both, as tod does
            ids = {t[0] for t in (self.tok.encode(v, add_special_tokens=False) for v in (letter, ' ' + letter))
                   if len(t) == 1}
            self.letter_ids.append(sorted(ids))
        self.T2 = float(self.manifest['temperatures']['T2'])

    def make_trainable(self):
        """LoRA weights in float32 with gradients (base stays frozen bfloat16); gradient checkpointing on."""
        if self.trainable:
            return
        for name, p in self.model.named_parameters():
            if 'lora_' in name:
                p.requires_grad_(True)
                p.data = p.data.float()
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
        self.trainable = True

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def prompt_ids(self, decision) -> list[int]:
        if len(decision.candidates) > len(LETTERS):
            raise ValueError(f'{decision.id}: TOD fine-tuning supports up to {len(LETTERS)} candidates')
        text = self.tok.apply_chat_template([{'role': 'user', 'content': build_user_message(record(decision))}],
                                            add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f'{decision.id}: prompt has {len(ids)} tokens (> {self.max_tokens}); truncation is disabled')
        return ids

    def letter_logits(self, decision):
        """Letter logits (n,) for one decision, as a torch tensor (with grad when training)."""
        torch = self.torch
        ids = torch.tensor([self.prompt_ids(decision)], device=self.device)
        if self._keep_last is not False:  # only the last position's logits, when the model supports it
            try:
                out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
                self._keep_last = True
            except TypeError:
                self._keep_last = False
        if self._keep_last is False:
            out = self.model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -1].float()
        return torch.stack([torch.logsumexp(logits[self.letter_ids[i]], 0) for i in range(len(decision.candidates))])

    def logits(self, decision) -> np.ndarray:
        with self.torch.inference_mode():
            return self.letter_logits(decision).cpu().numpy().astype(np.float64)


def release():
    """Free accelerator memory held by models nobody references any more (TOD is ~24 GB)."""
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------- training
def _rows(dataset, split):
    return [d for d in load_decisions(Path(dataset)/f'{split}.jsonl') if isinstance(d.target, Choice)]


def _index(decision):
    return [c.id for c in decision.candidates].index(decision.target.choice_id)


def _softmax(x, t=1.0):
    z = np.exp((x - x.max()) / t)
    return z / z.sum()


def _report(model, rows, temperatures, default_t):
    report = {}
    for kind in sorted({d.kind for d in rows}):
        hits = confidence = 0.
        subset = [d for d in rows if d.kind == kind]
        for d in subset:
            p = _softmax(model.logits(d), temperatures.get(kind, default_t))
            hits += int(p.argmax()) == _index(d)
            confidence += float(p.max())
        report[kind] = dict(accuracy=hits / len(subset), mean_confidence=confidence / len(subset), n=len(subset))
    return report


def finetune(dataset, out, *, source=TOD_REPO, epochs=2, learning_rate=2e-5, accumulate=8, max_train=None,
             baseline=True, seed=7, log=None):
    """Continue training TOD's stage-2 LoRA on a dataset directory written by `Dataset.write`.

    One adapter serves every kind (TOD is a general picker; the question tells kinds apart).
    The checkpoint with the best validation accuracy is kept, including the starting adapter
    if no epoch beats it. Temperatures are fitted per kind on the calibration split, and the
    test split is reported per kind (and for zero-shot TOD with `baseline`)."""
    import torch
    from ..decision_probability import fit_temperature
    log = log or (lambda message: print(message, flush=True))
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        raise ValueError(f'{out} already exists')
    spec = dataset_spec(dataset)
    rows = {split: _rows(dataset, split) for split in SPLITS}
    kinds = sorted({d.kind for d in rows['train']})
    missing = {k: [s for s in SPLITS if not any(d.kind == k for d in rows[s])] for k in kinds}
    if any(missing.values()):
        raise ValueError(f'every kind needs labelled rows in every split: {missing}')
    rng = random.Random(seed)
    torch.manual_seed(seed)
    train = rows['train'] if not max_train else rng.sample(rows['train'], min(max_train, len(rows['train'])))
    started = time.perf_counter()
    model = TodModel(source, trainable=True)
    log(f'TOD loaded on {model.device} in {time.perf_counter() - started:.0f}s; '
        f'{len(train)} training rows, kinds {kinds}')
    report = {}
    if baseline:
        model.model.eval()
        report['zero_shot_test'] = _report(model, rows['test'], {}, model.T2)
        log(f'zero-shot TOD test: {json.dumps(report["zero_shot_test"])}')

    def accuracy(split):
        model.model.eval()
        return float(np.mean([int(model.logits(d).argmax()) == _index(d) for d in rows[split]]))

    out.mkdir(parents=True)
    out_bundle(out, source)
    adapter = out/'stage2_adapter'
    best = accuracy('validation')
    model.model.save_pretrained(str(out/'_best'))
    log(f'validation accuracy before fine-tuning: {best:.3f}')
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started, total = time.perf_counter(), 0.
        rng.shuffle(train)
        model.model.train()
        optimizer.zero_grad()
        for i, d in enumerate(train, 1):
            loss = torch.nn.functional.cross_entropy(model.letter_logits(d)[None],
                                                     torch.tensor([_index(d)], device=model.device))
            (loss / accumulate).backward()
            total += loss.item()
            if i % accumulate == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if i % 100 == 0:
                log(f'  epoch {epoch}: {i}/{len(train)} rows, loss {total / i:.4f}, '
                    f'{(time.perf_counter() - epoch_started) / i:.2f}s/row')
        val = accuracy('validation')
        history.append(dict(epoch=epoch, train_loss=total / len(train), val_accuracy=val,
                            seconds=round(time.perf_counter() - epoch_started, 1)))
        log(json.dumps(history[-1]))
        if val > best:
            best = val
            shutil.rmtree(out/'_best')
            model.model.save_pretrained(str(out/'_best'))
    shutil.move(str(out/'_best'), str(adapter))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model.model, load_file(str(adapter/'adapter_model.safetensors')))
    model.model.eval()
    temperatures = {}
    for kind in kinds:
        subset = [d for d in rows['calibration'] if d.kind == kind]
        temperatures[kind] = fit_temperature([model.logits(d) for d in subset], [_index(d) for d in subset])['temperature']
    (out/'calibration.json').write_text(json.dumps(dict(temperatures=temperatures), indent=2) + '\n')
    report['test'] = _report(model, rows['test'], temperatures, model.T2)
    for kind, r in report['test'].items():
        r['temperature'] = temperatures[kind]
    (out/'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    bind_spec(out, spec, method='tod', source=str(source), dataset=str(dataset.resolve()), kinds=kinds,
              skipped={}, training=dict(epochs=epochs, learning_rate=learning_rate, accumulate=accumulate,
                                        train_rows=len(train), seed=seed, best_val_accuracy=best, history=history))
    log('test: ' + json.dumps(report['test']))
    del model, optimizer
    release()
    return dict(model=str(out), **report)


def out_bundle(out, source):
    """Write `out/picker.json`: the source manifest, pointing at the fine-tuned adapter."""
    manifest = json.loads((_fetch(source)/'picker.json').read_text())
    manifest['adapters'] = dict(manifest['adapters'], stage2='stage2_adapter', fine_tuned_from=str(source))
    (Path(out)/'picker.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return out


def is_tod_model(model) -> bool:
    try:
        return read_binding(model).get('method') == 'tod'
    except ValueError:
        return False


def calibrated_temperatures(model) -> dict:
    path = Path(model)/'calibration.json'
    return json.loads(path.read_text())['temperatures'] if path.exists() else {}


def evaluate(model, dataset, refit=False):
    """Refit (or reuse) the per-kind temperatures on calibration, then report test per kind."""
    from ..decision_probability import fit_temperature
    from .train import artifact_spec
    model, dataset = Path(model), Path(dataset)
    if artifact_spec(model) != dataset_spec(dataset):
        raise ValueError('model and dataset were built with different DecisionSpecs')
    tod = TodModel(model)
    tod.model.eval()
    kinds = read_binding(model)['kinds']
    temperatures = calibrated_temperatures(model)
    if refit or set(kinds) - set(temperatures):
        calibration = _rows(dataset, 'calibration')
        for kind in kinds:
            subset = [d for d in calibration if d.kind == kind]
            temperatures[kind] = fit_temperature([tod.logits(d) for d in subset], [_index(d) for d in subset])['temperature']
        (model/'calibration.json').write_text(json.dumps(dict(temperatures=temperatures), indent=2) + '\n')
    report = _report(tod, [d for d in _rows(dataset, 'test') if d.kind in kinds], temperatures, tod.T2)
    for kind, r in report.items():
        r['temperature'] = temperatures[kind]
    previous = json.loads((model/'evaluation.json').read_text()) if (model/'evaluation.json').exists() else {}
    (model/'evaluation.json').write_text(json.dumps(dict(previous, test=report), indent=2) + '\n')
    del tod
    release()
    return dict(test=report)


def policy(model, spec, *, calibrated=True, tod=None):
    """A fine-tuned TOD model acting through a spec (same interface as HeadPolicy)."""
    from .rl import HeadPolicy

    class TodPolicy(HeadPolicy):
        def __init__(self):
            self.model, self.spec = Path(model), spec
            self.binding = read_binding(model)
            if spec is not None:
                spec.require(self.binding['spec'])
            self.tod = tod or TodModel(model)
            self.tod.model.eval()
            temperatures = calibrated_temperatures(model) if calibrated else {}
            self.heads = {kind: dict(temperature=temperatures.get(kind, 1.0)) for kind in self.binding['kinds']}

        def logits(self, decision):
            # (decision, None, logits): RL keeps the decision itself as the training sample
            return decision, None, self.tod.logits(decision)

        # RL hooks: one adapter shared by every kind
        shared_weights = True
        rl_defaults = dict(steps=16, learning_rate=1e-5, minibatch=8, anchor_size=64)

        def trainer(self, anchor, anchor_weight, learning_rate, rng):
            return _Trainer(self, anchor, anchor_weight, learning_rate, rng)

        def _split(self, split):
            if not hasattr(self, '_rows'):
                self._rows = {}
            if split not in self._rows:
                self._rows[split] = _rows(self.binding['dataset'], split)
            return self._rows[split]

        def validation_accuracy(self, kinds):
            rows = self._split('validation')
            return {k: float(np.mean([int(self.tod.logits(d).argmax()) == _index(d) for d in rows if d.kind == k]))
                    for k in kinds if any(d.kind == k for d in rows)}

        def snapshot(self, kind):
            from peft import get_peft_model_state_dict
            return {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.tod.model).items()}

        def restore(self, kind, state):
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.tod.model, state)

        def save(self, out, info):
            """Write the updated adapter as a new model; calibration must be refit for it."""
            out = Path(out)
            out.mkdir(parents=True)
            self.tod.model.save_pretrained(str(out/'_adapter'))
            shutil.move(str(out/'_adapter'), str(out/'stage2_adapter'))
            shutil.copy(self.model/'picker.json', out/'picker.json')
            binding = dict(self.binding)
            binding.pop('rl_from', None)
            bind_spec(out, binding.pop('spec'), **dict(binding, rl=info), rl_from=str(self.model))

    return TodPolicy()


class _Trainer:
    """Updates TOD's shared adapter from RL samples [(decision, _, index, weight)], mixed with
    SFT training rows of the same kind (the anchor), one forward/backward pass per row."""

    def __init__(self, policy, anchor, anchor_weight, learning_rate, rng):
        import torch
        self.torch, self.policy, self.rng, self.anchor_weight = torch, policy, rng, anchor_weight
        policy.tod.make_trainable()
        self.params = policy.tod.parameters()
        self.optimizer = torch.optim.AdamW(self.params, lr=learning_rate, weight_decay=0.0)
        self.anchors = {}
        if anchor:
            for d in policy._split('train'):
                self.anchors.setdefault(d.kind, []).append(d)

    def update(self, kind, batch, anchor_size, pooled=False):
        torch, tod = self.torch, self.policy.tod
        rows = [(d, index, weight) for d, _, index, weight in batch]
        n = len(rows)
        anchor = self.anchors.get(kind, [])
        m = min(anchor_size, len(anchor))
        scale = (1 / (n + m), self.anchor_weight / (n + m)) if pooled else (1 / n, self.anchor_weight / max(1, m))
        rows = [(d, i, w * scale[0]) for d, i, w in rows]
        rows += [(anchor[j], _index(anchor[j]), scale[1]) for j in self.rng.choice(len(anchor), m, replace=False)]
        tod.model.train()
        self.optimizer.zero_grad()
        for d, index, weight in rows:
            loss = torch.nn.functional.cross_entropy(tod.letter_logits(d)[None], torch.tensor([index], device=tod.device))
            (weight * loss).backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        tod.model.eval()


__all__ = ['TOD_REPO', 'TodModel', 'finetune', 'evaluate', 'policy', 'record', 'build_user_message', 'is_tod_model']
