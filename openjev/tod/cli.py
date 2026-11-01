"""`openjev tod`: train, calibrate, evaluate and export decision models from Dataset.write directories."""
import json
from pathlib import Path

from ..backbone import DEFAULT_REPO

# Flags only `--method encoder` reads; other methods refuse them instead of ignoring them.
ENCODER_FLAGS = ('loss', 'focal_gamma', 'focal_alpha', 'brier', 'batch_size', 'max_steps', 'eval_every',
                 'eval_split', 'checkpoint_every', 'keep_checkpoints', 'max_length', 'targets', 'survival',
                 'default_point')


def _targets(text):
    return tuple(float(x) for x in text.split(',')) if text else None


def run(args):
    from .. import compute
    compute.select(args.backend, args.device)
    if args.action == 'train':
        result = _train(args)
    elif args.action == 'calibrate':
        from .encoder import calibrate, is_encoder_model
        if not is_encoder_model(args.model):
            raise SystemExit('calibrate applies to --method encoder models; use `openjev tod evaluate --refit` otherwise')
        options = {k: v for k, v in dict(targets=_targets(args.targets), survival=args.survival,
                                          default_point=args.default_point).items() if v is not None}
        result = calibrate(args.model, args.dataset, split=args.split, **options)
    elif args.action == 'export':
        from .encoder import export
        result = export(args.model, args.out, check=args.check, device=args.device)
    else:
        from .encoder import evaluate, is_encoder_model
        if is_encoder_model(args.model):
            result = evaluate(args.model, args.dataset, split=args.split, out=args.out,
                              probabilities=args.probabilities, refit=args.refit, source_sep=args.source_sep)
        else:
            if (args.split, args.out, args.probabilities) != ('test', None, None):
                raise SystemExit('--split, --out and --probabilities apply to --method encoder models')
            from .train import evaluate_dataset
            result = evaluate_dataset(args.model, args.dataset, refit=args.refit)
    print(json.dumps(result, indent=2))


def _train(args):
    if args.method != 'encoder':
        given = [f'--{k.replace("_", "-")}' for k in ENCODER_FLAGS if getattr(args, k) is not None]
        if given:
            raise SystemExit(f'{", ".join(given)} only apply to --method encoder')
    if args.method == 'tod':
        from .picker import finetune
        return finetune(args.dataset, args.out, source=args.model or 'parsecai/tod', epochs=args.epochs or 2,
                        learning_rate=args.learning_rate or 2e-5, max_train=args.max_train, seed=args.seed)
    if args.method == 'head':
        from .train import train_dataset
        return train_dataset(args.dataset, args.model, args.out, epochs=args.epochs or 20,
                             learning_rate=args.learning_rate or 5e-5, rank=args.rank, seed=args.seed)
    from .encoder import DEFAULT_MODEL, finetune
    if args.focal_gamma is not None or args.focal_alpha is not None:
        if args.loss != 'focal':
            raise SystemExit('--focal-gamma and --focal-alpha need --loss focal')
    options = dict(loss=args.loss, focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha, brier=args.brier,
                   batch_size=args.batch_size, max_steps=args.max_steps, eval_every=args.eval_every,
                   eval_split=args.eval_split, checkpoint_every=args.checkpoint_every,
                   keep_checkpoints=args.keep_checkpoints, max_length=args.max_length,
                   targets=_targets(args.targets), survival=args.survival, default_point=args.default_point)
    return finetune(args.dataset, args.out, source=args.model or DEFAULT_MODEL, epochs=args.epochs or 1,
                    learning_rate=args.learning_rate or 2e-5, seed=args.seed,
                    **{k: v for k, v in options.items() if v is not None})


def register(subparsers):
    p = subparsers.add_parser('tod', help='train and evaluate task-oriented decision models (MLX or PyTorch)')
    commands = p.add_subparsers(dest='action', required=True)
    t = commands.add_parser('train', help='fine-tune TOD or any Hugging Face model, or train heads on a frozen '
                                          'backbone; calibrate; test per kind')
    t.add_argument('dataset', type=Path, help='directory written by openjev.tod.Dataset.write')
    t.add_argument('--method', choices=('tod', 'head', 'encoder'), default='head',
                   help='tod: fine-tune the TOD picker (Gemma 4 12B + LoRA, PyTorch); '
                        'head: small heads on a frozen Gemma 3 4B (MLX or PyTorch, ~9 GB); '
                        'encoder: fine-tune all of any Hugging Face model with a per-candidate head '
                        '(PyTorch; the only method for Subset targets)')
    t.add_argument('--model', default=None,
                   help=f'tod: a TOD bundle (default parsecai/tod); head: the backbone '
                        f'(default: $OPENJEV_MODEL, else {DEFAULT_REPO}); encoder: the model to fine-tune '
                        f'(default answerdotai/ModernBERT-large), or a model trained here to continue from. '
                        f'Directories or Hugging Face repo ids')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=None, help='default: 2 (tod), 20 (head), 1 (encoder)')
    t.add_argument('--learning-rate', type=float, default=None,
                   help='default: 2e-5 (tod, encoder), 5e-5 (head)')
    t.add_argument('--max-train', type=int, default=None, help='tod: train on a random subset of this many rows')
    t.add_argument('--rank', type=int, default=256)
    t.add_argument('--seed', type=int, default=7)
    g = t.add_argument_group('encoder', 'options for --method encoder (other methods refuse them)')
    g.add_argument('--loss', choices=('bce', 'focal'), default=None, help='Subset kinds (default bce)')
    g.add_argument('--focal-gamma', type=float, default=None, help='default 2')
    g.add_argument('--focal-alpha', type=float, default=None, help='weight of positives, default 0.5')
    g.add_argument('--brier', type=float, default=None, help='add this times the Brier score (default 0)')
    g.add_argument('--batch-size', type=int, default=None, help='decisions per step per GPU (default 8)')
    g.add_argument('--max-steps', type=int, default=None, help='stop after this many steps')
    g.add_argument('--max-length', type=int, default=None, help='tokens per (question, candidate) pair')
    g.add_argument('--eval-every', type=int, default=None, help='probe --eval-split every N steps')
    g.add_argument('--eval-split', default=None, help='default validation')
    g.add_argument('--checkpoint-every', type=int, default=None, help='save model/checkpoints/step-N every N steps')
    g.add_argument('--keep-checkpoints', type=int, default=None, help='newest checkpoints to keep (default 2)')
    e = commands.add_parser('evaluate', help='calibrate (if needed) and report the test split per kind')
    e.add_argument('model', type=Path)
    e.add_argument('dataset', type=Path, help='a dataset built with the same spec; evaluation-only sets work')
    e.add_argument('--refit', action='store_true', help='refit calibration (e.g. after RL or on new data)')
    e.add_argument('--split', default='test', help='encoder: the split to report (default test)')
    e.add_argument('--out', type=Path, default=None, help="encoder: write the report here, not the model's evaluation.json")
    e.add_argument('--probabilities', type=Path, default=None, help='encoder: save per-candidate probabilities (JSON lines)')
    e.add_argument('--source-sep', default=':', help='encoder: per-source metrics by decision id prefix (default ":")')
    c = commands.add_parser('calibrate', help='encoder: re-fit temperatures and re-choose operating points without '
                                              'retraining')
    c.add_argument('model', type=Path)
    c.add_argument('dataset', type=Path)
    c.add_argument('--split', default='calibration')
    x = commands.add_parser('export', help='encoder: write a shippable copy (bf16 trunk, fp32 head, no checkpoints) '
                                           'and check it matches')
    x.add_argument('model', type=Path, help='a model or checkpoint directory')
    x.add_argument('out', type=Path)
    x.add_argument('--check', type=int, default=64, help='rows to compare against the source (0: skip)')
    for command in (t, c):
        command.add_argument('--targets', default=None, help='encoder: survival targets, default 0.95,0.98,0.99')
        command.add_argument('--survival', choices=('any', 'all'), default=None,
                             help='encoder: a question survives when any (default) or all of its relevant '
                                  'candidates are kept')
        command.add_argument('--default-point', default=None, help='encoder: default operating point (rho_99)')
    for command in (t, e, c, x):
        command.add_argument('--backend', choices=('auto', 'mlx', 'torch'), default=None,
                             help='default: $OPENJEV_BACKEND, else MLX on Apple silicon and PyTorch elsewhere')
        command.add_argument('--device', default=None,
                             help='PyTorch device: cuda, cuda:N, mps, cpu (default: $OPENJEV_DEVICE, else the first available)')
    p.set_defaults(fn=run)
