"""`openjev tod`: train and evaluate decision models from Dataset.write directories."""
import json
from pathlib import Path

from ..backbone import DEFAULT_REPO


def run(args):
    from .. import compute
    from .train import evaluate_dataset, train_dataset
    compute.select(args.backend, args.device)
    if args.action == 'train':
        result = train_dataset(args.dataset, args.model, args.out, epochs=args.epochs,
                               learning_rate=args.learning_rate, rank=args.rank, seed=args.seed)
    else:
        result = evaluate_dataset(args.model, args.dataset, refit=args.refit)
    print(json.dumps(result, indent=2))


def register(subparsers):
    p = subparsers.add_parser('tod', help='train and evaluate task-oriented decision models (MLX or PyTorch)')
    commands = p.add_subparsers(dest='action', required=True)
    t = commands.add_parser('train', help='features -> head -> calibration -> per-kind test metrics')
    t.add_argument('dataset', type=Path, help='directory written by openjev.tod.Dataset.write')
    t.add_argument('--model', default=None,
                   help=f'backbone: a local directory or a Hugging Face repo id (default: $OPENJEV_MODEL, else {DEFAULT_REPO}, downloaded once)')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=20)
    t.add_argument('--learning-rate', type=float, default=5e-5)
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
