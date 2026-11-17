# API reference

## Cross-platform inference

`OptionScorer` accepts `backend="auto"` and `device="auto"`. Automatic backend
selection uses MLX on Apple silicon and PyTorch elsewhere. For PyTorch, automatic
device selection prefers CUDA, then MPS, then CPU.

```python
from openjev import OptionScorer

scorer = OptionScorer("google/gemma-3-4b-it", backend="torch", device="cuda", batch_size=2)
results = scorer.score("The capital of France is", [" Paris", " Berlin"])
```

Use `device="cpu"` for CPU inference. The server uses the same options:
`uv run openjev serve --backend torch --device auto`. Both `/score` and
`/v1/systemone` retain the same request and response contracts across backends.
See [Getting started](getting-started.md) for GPU installation instructions.


The `openjev` package. `OptionScorer` is re-exported from the top level:

```python
from openjev import OptionScorer
```

## `openjev`

::: openjev

## `openjev.scorer`

::: openjev.scorer

## `openjev.systemone`

::: openjev.systemone

## `openjev.video`

::: openjev.video

## `openjev.server`

::: openjev.server

## `openjev.features`

::: openjev.features

## `openjev.head`

::: openjev.head

## `openjev.train`

::: openjev.train

## `openjev.cli`

::: openjev.cli
