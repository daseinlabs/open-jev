"""Train Gemma 3 4B to play 2048 with an all-moves (expected) policy gradient.

One command does the whole run:
  1. evaluate a random player and the untouched base model,
  2. train LoRA weights for ``--hours``,
  3. evaluate the trained model and write report.md + charts.svg.

    caffeinate -i .venv/bin/python -m rl_2048.train --hours 4.5

The training step, for every board the model visits:
  * Policy: openjev scores the 4 moves -> probabilities p(a).
  * Critic: random rollouts estimate how many points each move leads to -> R(a).
  * Advantage: R normalised across the 4 moves (z-score) -> A(a).
  * Loss: -sum_a p(a) * A(a)  +  kl_coef * KL(p || p_base).
    Because every move is scored, this is the exact expected policy gradient for
    the board; no sampling noise from "which move happened to be played".
  * The game itself continues with a move sampled from p (exploration).
Gradients from ``--batch`` boards are averaged into one Adam step on LoRA weights.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from openjev.scorer import DEFAULT_MODEL

from . import evaluate, report
from .game import ACTIONS, Game, normalised_values, rollout_values
from .policy import GemmaPolicy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="run directory (default runs/rl2048-<timestamp>)")
    ap.add_argument("--hours", type=float, default=4.5, help="training time budget (evaluation is extra, ~20-30 min)")
    ap.add_argument("--max-updates", type=int, default=0, help="stop after this many updates (0 = time budget only)")
    ap.add_argument("--batch", type=int, default=16, help="boards per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--temperature", type=float, default=5.0, help="policy softmax temperature during training")
    ap.add_argument("--kl-coef", type=float, default=0.1, help="pull toward the base model (0 disables, saves ~1/3 time)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--rollouts", type=int, default=16, help="random playouts per move for the critic")
    ap.add_argument("--depth", type=int, default=10, help="moves per playout")
    ap.add_argument("--death-penalty", type=float, default=256.0)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=20.0)
    ap.add_argument("--eval-games", type=int, default=10, help="games for base and trained model evaluation")
    ap.add_argument("--random-games", type=int, default=50)
    ap.add_argument("--max-eval-moves", type=int, default=1500)
    ap.add_argument("--skip-base-eval", action="store_true")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint the adapter every N updates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test run (~3 minutes)")
    args = ap.parse_args()
    if args.smoke:
        args.hours, args.batch, args.eval_games, args.random_games = 0.02, 4, 1, 5
        args.max_eval_moves, args.save_every = 20, 2
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.out or f"runs/rl2048-{datetime.now():%Y%m%d-%H%M}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log_file = (out / "console.log").open("a")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"run directory: {out}")
    mx.random.seed(args.seed)
    policy = GemmaPolicy(args.model, lora_layers=args.lora_layers,
                         lora_rank=args.lora_rank, lora_scale=args.lora_scale,
                         temperature=args.temperature)
    n_params = sum(v.size for _, v in tree_flatten(policy.model.trainable_parameters()))
    log(f"model loaded; trainable LoRA parameters: {n_params:,}")

    # ---- 1. baselines -----------------------------------------------------
    log(f"phase 1/3: baseline evaluation ({args.random_games} random games, {args.eval_games} base-model games)")
    res = evaluate.play(evaluate.random_chooser(args.seed), args.random_games, args.max_eval_moves, "random",
                        log=lambda m: None)
    (out / "eval_random.json").write_text(json.dumps(res, indent=2))
    log(f"random: mean score {res['mean_score']:.0f}")
    if not args.skip_base_eval:
        res = evaluate.play(evaluate.greedy_chooser(policy), args.eval_games, args.max_eval_moves, "base", log=log)
        (out / "eval_base.json").write_text(json.dumps(res, indent=2))
        log(f"base model: mean score {res['mean_score']:.0f}, top choice invalid {res['invalid_top_rate']:.1%}")

    # ---- 2. training ------------------------------------------------------
    rng = random.Random(args.seed)
    opt = optim.Adam(learning_rate=args.lr)

    def state_loss(model, ids, adv, ref_logp):
        logp = policy.log_probs(model, ids)
        p = mx.exp(logp)
        pg = -(p * adv).sum()
        kl = (p * (logp - ref_logp)).sum()
        return pg + args.kl_coef * kl, (p, kl)

    loss_and_grad = nn.value_and_grad(policy.model, state_loss)
    game, n_games, update, t_start = Game(seed=rng.randrange(1 << 30)), 0, 0, time.time()
    deadline = t_start + args.hours * 3600
    steps_f, eps_f = (out / "train_log.jsonl").open("a"), (out / "episodes.jsonl").open("a")
    log(f"phase 2/3: training for {args.hours:.2f} h, {args.batch} boards per update; Ctrl+C stops early and still evaluates")

    try:
        while time.time() < deadline and not (args.max_updates and update >= args.max_updates):
            t0, grads, stats = time.time(), None, []
            for _ in range(args.batch):
                valid = game.valid_actions()
                ids = policy.encode(game.board, valid)
                adv = normalised_values(rollout_values(game.board, rng, args.rollouts, args.depth,
                                                       args.death_penalty))
                ref_logp = (mx.log(policy.reference_probs(ids) + 1e-12) if args.kl_coef > 0
                            else mx.zeros(4))
                (loss, (p, kl)), g = loss_and_grad(policy.model, ids, mx.array(adv), ref_logp)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                mx.eval(grads, p, kl)
                p = p.tolist()
                best = max(adv)
                stats.append({
                    "best_move_prob": sum(pi for pi, a in zip(p, adv) if a >= best - 1e-9),
                    "advantage_z": sum(pi * a for pi, a in zip(p, adv)),
                    "invalid_prob": sum(pi for pi, a in zip(p, ACTIONS) if a not in valid),
                    "entropy": -sum(pi * math.log(pi) for pi in p if pi > 0),
                    "kl": kl.item(),
                })
                # Keep playing: sample among valid moves in proportion to the policy.
                weights = [p[ACTIONS.index(a)] for a in valid]
                game.step(rng.choices(valid, weights=weights)[0] if sum(weights) > 0 else rng.choice(valid))
                if game.over():
                    n_games += 1
                    eps_f.write(json.dumps({"game": n_games, "update": update, "score": game.score,
                                            "max_tile": game.max_tile, "moves": game.moves}) + "\n")
                    eps_f.flush()
                    log(f"training game {n_games} finished: score {game.score}, max tile {game.max_tile}, {game.moves} moves")
                    game = Game(seed=rng.randrange(1 << 30))

            grads = tree_map(lambda x: x / args.batch, grads)
            grads, gnorm = optim.clip_grad_norm(grads, args.max_grad_norm)
            opt.update(policy.model, grads)
            mx.eval(policy.model.trainable_parameters(), opt.state)
            update += 1
            row = {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}
            row.update(update=update, states=len(stats), grad_norm=gnorm.item(), sec=time.time() - t0,
                       elapsed_h=(time.time() - t_start) / 3600)
            steps_f.write(json.dumps(row) + "\n")
            steps_f.flush()
            left = max(0.0, deadline - time.time()) / 60
            log(f"update {update}: best-move prob {row['best_move_prob']:.3f}, advantage {row['advantage_z']:+.3f}, "
                f"invalid {row['invalid_prob']:.3f}, entropy {row['entropy']:.3f}, KL {row['kl']:.4f}, "
                f"{row['sec']:.1f}s, {left:.0f} min left")
            if update % args.save_every == 0:
                policy.save(out / "adapter")
                report.build(out)
    except KeyboardInterrupt:
        log("training stopped by Ctrl+C; saving and evaluating (press Ctrl+C again to abort)")

    policy.save(out / "adapter")
    log(f"adapter saved to {out / 'adapter'} after {update} updates")

    # ---- 3. final evaluation ----------------------------------------------
    log(f"phase 3/3: evaluating the trained model on {args.eval_games} games")
    res = evaluate.play(evaluate.greedy_chooser(policy), args.eval_games, args.max_eval_moves, "trained", log=log)
    (out / "eval_trained.json").write_text(json.dumps(res, indent=2))
    log(f"trained model: mean score {res['mean_score']:.0f}, top choice invalid {res['invalid_top_rate']:.1%}")
    report.build(out)
    log(f"done. report: {out / 'report.md'}, charts: {out / 'charts.svg'}")


if __name__ == "__main__":
    main()
