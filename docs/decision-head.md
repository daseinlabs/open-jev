# Native candidate-decision heads

The `openjev decision` commands train and serve a shared attention head on frozen Gemma features. Unlike continuation scoring, this head learns a candidate-level decision objective rather than the probability of spelling each candidate's text. The existing `score`, `features`, `train`, chess LoRA, and server commands retain their existing behavior.

This path currently requires MLX on Apple silicon, an accessible Metal device, and local model weights. Continuation inference remains available through the existing MLX/PyTorch backends. HTTP integration, preference/ranking objectives, Brier training, and calibration are separate follow-up work.

## Choice contract

A native example has the same representation for training and inference:

```json
{
  "schema_version": 1,
  "id": "ticket-001",
  "group_id": "customer-17",
  "state": {"message": "My item arrived broken. Please replace it."},
  "question": "Which department should handle this request?",
  "candidates": [
    {"id": "returns", "text": "Returns and replacements"},
    {"id": "billing", "text": "Billing and payments"}
  ],
  "target": {"choice_id": "returns"}
}
```

`target` is required for training/evaluation and optional for inference. `group_id` is optional but should identify customers, source games, documents, or other units that must not cross splits. IDs must be unique within a dataset and disjoint across its splits. There must be 2–255 candidates with unique nonempty IDs and texts. State accepts a JSON string, object, or array. Unsupported keys and supervision forms are rejected.

Candidate IDs, example IDs, group IDs, and targets never enter the encoder. Only state, question, and candidate texts supply model features. Objects are serialized with sorted keys and compact JSON. The rendering recipe is versioned and saved with the artifact. Exact utility ties use lexicographic candidate ID ordering for deterministic selection; IDs otherwise have no effect on scores.

[Example inference request](https://github.com/daseinlabs/open-jev/blob/main/examples/decision-choice.json) illustrates the schema, not an assertion that a head trained on another task can route support requests correctly.

## Train from native JSONL

```sh
.venv/bin/openjev decision features train.jsonl \
  --model models/gemma-3-4b-it --split train --out runs/native/train.npz
.venv/bin/openjev decision features validation.jsonl \
  --model models/gemma-3-4b-it --split validation --out runs/native/validation.npz

.venv/bin/openjev decision train runs/native/train.npz \
  --validation runs/native/validation.npz --out runs/native/artifact \
  --rank 256 --epochs 8 --batch-size 64 --learning-rate 0.0001 --seed 7
```

Outputs must be new paths; commands do not overwrite caches, artifacts, or evaluation directories. Contexts longer than 4,096 tokens or candidates longer than 256 tokens fail explicitly rather than being silently truncated. Set `--context-tokens` and `--candidate-tokens` during feature extraction to change these limits; training and validation feature contracts must agree.

Feature extraction retains every final-layer context vector and mean-pools each independently encoded candidate after its BOS prefix. Mean pooling uses float32 and excludes the BOS token; cached vectors use float16. The backbone is frozen. The existing attention head normalizes features, uses candidates as queries over context keys/values, and produces one utility per candidate.

Training uses mean categorical cross-entropy, AdamW (native default learning rate 1e-4), and the existing gradient-clipping path. The native learning rate is lower than the legacy head-training default; the latter was unstable on the rendered state/question features in validation. The best validation top-1 checkpoint is exported; ties keep the first checkpoint. Test data cannot be used as the validation feature split. Features must agree on model/tokenizer contents, extraction recipe, dimensions, token limits, and extraction library versions. Legacy feature caches lack this contract and must be regenerated for this path.

## Reload and score

```sh
.venv/bin/openjev decision score runs/native/artifact \
  --request examples/decision-choice.json
```

The output includes the selected candidate ID, utilities, probabilities in request order, artifact identity, and elapsed time. The probabilities are a softmax over learned candidate utilities. They are not automatically calibrated confidence estimates or probabilities of real-world success.

The artifact records the backbone's original absolute path. For relocated model files, supply `--model /new/path`; their content fingerprints must match. Backbone weights are referenced, not copied into the head artifact. Model and tokenizer files are hashed to detect incompatible substitutions; this adds startup I/O. Inference requires the feature-extraction library versions recorded by the artifact; restore that runtime or regenerate features and retrain after an incompatible upgrade.

## Evaluate and check invariance

```sh
.venv/bin/openjev decision eval runs/native/artifact \
  --data test.jsonl --out runs/native/evaluation --baseline

.venv/bin/openjev decision check runs/native/artifact \
  --data test.jsonl --limit 5
```

Evaluation reports accuracy, top-3, stable log loss, summed multiclass Brier, 10-bin top-label ECE, and uniform-choice chance accuracy. It saves per-example predictions and an artifact/data manifest. `--baseline` evaluates zero-shot summed continuation likelihoods using the same state/question and leading-space candidate text. This baseline is explicitly a different scoring strategy and has no task-specific head training.

Evaluation rejects overlap with the artifact's training or validation examples by ID, semantic input fingerprint, and supplied group ID. Fingerprints ignore candidate order/IDs and include state, question, and candidate texts. These checks do not detect paraphrases, pretraining contamination, or group leakage when group IDs are omitted. `--limit` marks a run as a smoke test; it does not bypass overlap checks on the full dataset.

The check command compares utilities and probabilities after candidate permutation, candidate-ID replacement, and an interleaved unrelated request followed by repetition. It exits unsuccessfully when the selected ID changes under permutation/repetition or numerical differences exceed the configured tolerance. It does not measure task accuracy.

## Controlled exact-match benchmark

The existing synthetic files define an exact badge-matching task with 2,000 training, 400 validation, and 400 test examples. Their labels can be independently checked against the badge explicitly stated in each context. This controlled benchmark verifies the training and artifact path; it does not establish general-purpose judgment quality.

Convert the legacy JSONL format explicitly, using distinct split prefixes:

```sh
.venv/bin/openjev decision convert data/synthetic/train.jsonl \
  --out runs/badges/train.jsonl --prefix train \
  --question 'Select the candidate matching the badge stated in the context.'
.venv/bin/openjev decision convert data/synthetic/validation.jsonl \
  --out runs/badges/validation.jsonl --prefix validation \
  --question 'Select the candidate matching the badge stated in the context.'
.venv/bin/openjev decision convert data/synthetic/test.jsonl \
  --out runs/badges/test.jsonl --prefix test \
  --question 'Select the candidate matching the badge stated in the context.'
```

Use those paths in the extraction, training, and evaluation commands above. The converter maps legacy `{context, options, label}` records into the native contract without treating candidate IDs as features. Keep the original files and labels unchanged, and do not tune on the test set.

## Artifact layout

```text
artifact/
  head.safetensors   learned attention-head parameters
  head.json          dimensions, training history, and feature metadata
  manifest.json      strategy, recipe, backbone/tokenizer hashes, runtime,
                     head checksums, objective, and train/validation identities
```

The manifest is written last. A failed training run without it is not a loadable artifact. Cached data retains its source hash, split role, provenance, and feature signature. Keep caches and artifacts in ignored `runs/` directories.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_decision*.py' -v
# Additional actual MLX head masking, invariance, and reload checks:
OPENJEV_TEST_MLX=1 .venv/bin/python -m unittest discover -s tests -p test_decision_mlx.py -v
```
