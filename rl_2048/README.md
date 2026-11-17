# RL on 2048 with openjev

Gemma 3 4B learns to play 2048 by reinforcement learning. openjev's option
scoring is the policy, random playouts are the critic, and LoRA weights are
trained with an all-moves (expected) policy gradient. Runs on Apple silicon (MLX).

## Run

```sh
caffeinate -i .venv/bin/python -m rl_2048.train --hours 4.75
```

`caffeinate -i` keeps the Mac awake. One command does three phases and writes
everything to `runs/rl2048-<timestamp>/`:

1. Baseline: 50 games by a random player, 10 games by the untouched base model.
2. Training for `--hours`. Progress lines every ~15 s; adapter checkpoint and
   report refresh every 25 updates.
3. 10 games by the trained model (same starting seeds as the base model), then
   `report.md` and `charts.svg`.

Ctrl+C during training stops early and still saves and evaluates.

Watch it play in the browser (board plus live move probabilities and a move log;
switch between the trained model and base Gemma):

```sh
.venv/bin/python -m rl_2048.web          # newest adapter in runs/; then open http://127.0.0.1:8001
```

Or in the terminal:

```sh
.venv/bin/python -m rl_2048.play --adapter runs/rl2048-<timestamp>/adapter   # trained
.venv/bin/python -m rl_2048.play                                             # base model
```

Smoke test (about 3 minutes): `.venv/bin/python -m rl_2048.train --smoke`.
Game-logic tests: `.venv/bin/python -m unittest tests.test_rl_2048_game`.

## How it works

| RL term | Here |
|---|---|
| Environment | `game.py`: 4x4 2048, plain Python |
| State | The board rendered as text in a prompt (`policy.py`) |
| Actions | `up`, `down`, `left`, `right`: each a single Gemma token |
| Policy p(a) | openjev scoring: log-probability of each move given the prompt, softmax over the 4 (`norm="sum"`, chat template), at temperature 5 during training |
| Reward | Points from merges |
| Critic Q(s, a) | Monte Carlo: play the move, then 10 random moves, 16 times, average the points (`rollout_values`) |
| Advantage A(s, a) | Q normalised to a z-score over the 4 moves; invalid moves rank last |
| Update | Adam on LoRA weights (8 layers, rank 8) |

For each board the model visits:

```
loss = -sum_a p(a) * A(a)  +  0.1 * KL(p || p_base)
```

* **Why all four moves at once?** Standard REINFORCE samples one move and only
  learns about that move, so the gradient is noisy. openjev already scores every
  option in one pass and the game is cheap to simulate, so we take the exact
  expectation over moves. This is the "expected policy gradient" / "all-action"
  estimator. With the exact expectation, a baseline does not change the gradient,
  because sum_a grad p(a) = 0; it only matters when you sample.
* **Why z-scores?** Points vary from 4 early on to thousands late in the game.
  Normalising per board, as GRPO does per group, keeps every board's update a similar size.
* **Why the KL term?** It stops the LoRA weights drifting far from the base
  model (forgetting language, collapsing to one move). The reference
  probabilities come from the same model with LoRA switched off.
* **Why temperature 5?** The base model is almost certain on every board (it
  answers "right" about 39 times in 40). A near-0/1 softmax has near-zero
  gradient, so nothing learns. Temperature softens it; greedy play is unchanged.
* **Exploration:** the game continues with a move *sampled* from p, so the
  model visits boards it would not reach greedily.

## Reading the log

| Field | Meaning | Good direction |
|---|---|---|
| `best-move prob` | Probability on the move the critic rates highest (uniform = 0.25) | Up |
| `advantage` | Expected z-score of the policy's move mix (uniform = 0) | Up, above 0 |
| `invalid` | Probability on moves that do nothing | Down |
| `entropy` | Spread of the policy, max 1.386 (uniform) | Moderate; ~0 means collapse |
| `KL` | Distance from the base model | Stays bounded |

## Limitations

* The critic sees 10 random moves ahead, so the policy learns good short-term
  play (merges, survival), not the long-horizon corner strategy strong players use.
* Expect only about 1,000 updates (~17,000 boards) in 4-5 hours, and only 10
  evaluation games per model, so differences have wide error bars.
* Classic agents (expectimax search, n-tuple TD learning) play far better. The
  point here is the RL loop on an LLM policy, not state-of-the-art 2048.

## Ideas for next steps

GRPO over multi-move segments for comparison; longer or learned critic (value
head on openjev features); expert-iteration warm start from expectimax; the same
loop on the Doom demo's action menu.
