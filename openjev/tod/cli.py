"""`openjev tod`: train and evaluate decision models from Dataset.write directories."""
import json
from pathlib import Path

from ..backbone import DEFAULT_REPO


def run(args):
    from .. import compute
    from .train import evaluate_dataset, train_dataset
    compute.select(args.backend, args.device)
    if args.action == 'train' and args.method == 'tod':
        from .picker import finetune
        result = finetune(args.dataset, args.out, source=args.model or 'parsecai/tod', epochs=args.epochs or 2,
                          learning_rate=args.learning_rate or 2e-5, max_train=args.max_train, seed=args.seed)
    elif args.action == 'train':
        result = train_dataset(args.dataset, args.model, args.out, epochs=args.epochs or 20,
                               learning_rate=args.learning_rate or 5e-5, rank=args.rank, seed=args.seed)
    else:
        result = evaluate_dataset(args.model, args.dataset, refit=args.refit)
    print(json.dumps(result, indent=2))


def register(subparsers):
    p = subparsers.add_parser('tod', help='train and evaluate task-oriented decision models (MLX or PyTorch)')
    commands = p.add_subparsers(dest='action', required=True)
    t = commands.add_parser('train', help='fine-tune TOD, or train heads on a frozen backbone; calibrate; test per kind')
    t.add_argument('dataset', type=Path, help='directory written by openjev.tod.Dataset.write')
    t.add_argument('--method', choices=('tod', 'head'), default='head',
                   help='tod: fine-tune the TOD picker (Gemma 4 12B + LoRA, PyTorch); '
                        'head: small heads on a frozen Gemma 3 4B (MLX or PyTorch, ~9 GB)')
    t.add_argument('--model', default=None,
                   help=f'tod: a TOD bundle (default parsecai/tod); head: the backbone '
                        f'(default: $OPENJEV_MODEL, else {DEFAULT_REPO}). Directories or Hugging Face repo ids')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=None, help='default: 2 (tod), 20 (head)')
    t.add_argument('--learning-rate', type=float, default=None, help='default: 2e-5 (tod), 5e-5 (head)')
    t.add_argument('--max-train', type=int, default=None, help='tod: train on a random subset of this many rows')
    t.add_argument('--rank', type=int, default=256)
    t.add_argument('--seed', type=int, default=7)
    e = commands.add_parser('evaluate', help='calibrate (if needed) and report per-kind test accuracy')
    e.add_argument('model', type=Path)
    e.add_argument('dataset', type=Path)
    e.add_argument('--refit', action='store_true', help='refit the calibrator (e.g. after RL)')
    for command in (t, e):
        command.add_argument('--backend', choices=('auto', 'mlx', 'torch'), default=None,
                             help='default: $OPENJEV_BACKEND, else MLX on Apple silicon and PyTorch elsewhere')
        command.add_argument('--device', default=None,
                             help='PyTorch device: cuda, cuda:N, mps, cpu (default: $OPENJEV_DEVICE, else the first available)')
    p.set_defaults(fn=run)
