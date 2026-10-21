# KV-cache prefix composition experiment

This repository trains one virtual prefix for each of three domains and evaluates whether prefixes can be composed without retraining the base model.

## Public entry point

All experiment lifecycle commands go through one launcher:

```bash
./experiment.sh help
```

Common operations:

```bash
# Start or inspect the AWM, EnvScaler, and TACO tool environments.
./experiment.sh tools start
./experiment.sh tools status

# Train AWM, EnvScaler, and TACO serially. Each job owns all eight GPUs
# (six student GPUs and two teacher GPUs); export and compose prefixes.
./experiment.sh train all

# Run one-domain training or a three-update partial-rollout smoke run.
./experiment.sh train awm
./experiment.sh train-smoke taco

# Deploy Base, AWM, EnvScaler, and TACO as four two-GPU vLLM services.
./experiment.sh serve start
./experiment.sh serve status

# Evaluate all four models or one selected model.
nohup ./experiment.sh eval all > outputs/formal_eval.log 2>&1 &
./experiment.sh eval taco

# Show training and evaluation status.
./experiment.sh status
```

The launcher uses `outputs/prefix_composition_3domain_20260917` by default. Override it with `EXPERIMENT_ROOT` when starting a new experiment.

Training defaults to shared-GPU partial rollout (`TRAINER_MODE=colocate_async`).
The training batch and optimizer mini-batch remain 24. The rollout pool contains
up to 32 prompt groups, including unfinished episodes and completed trajectories
waiting to train. Each update consumes 24 completed, teacher-scored groups;
the remaining 8 stay in the pool and only the missing 24 are added for the next
update. This keeps the pool bounded instead of adding 32 fresh prompts every
step. Near the end, new admission decreases so the final update drains the pool
without changing the requested number of optimizer updates or training samples.

`PARTIAL_ROLLOUT_TARGET` defaults to ceil(4/3 × training batch), so it is 32 for
batch 24. Set `PARTIAL_ROLLOUT_TARGET=36` for 1.5× oversubscription. Pool size must
be at least the training batch. Dataset fetch chunks are automatically set to
one prompt so arbitrary refill sizes can be coalesced; training batch, epoch
length, and optimizer step counts are unchanged. A null
`trainer.v1.colocate_async.rollout_target_size` retains the legacy whole-batch
warmup controlled by `PARTIAL_WARMUP_BATCHES=1`.

Unfinished generations retain tokens and resume after the prefix update,
rebuilding KV/GDN state under the new weights. The actor and rollout continue
to share their GPUs. Retries wait until weights and KV memory are fully restored;
each request keeps its original token budget across continuation.

`PARTIAL_MAX_OFF_POLICY=8` bounds age in model versions. The default
`PARTIAL_OFF_POLICY_STRATEGY=wait` retains old trajectories and waits for them
at the limit rather than dropping them. An individual trajectory may span
multiple prefix versions. Tool sessions remain live across in-process pauses;
process restart restores actor/optimizer/dataloader state. Queue restoration
requires TransferQueue checkpoint APIs (version >= 0.1.9); supported versions
reissue in-flight episodes from their original prompts. The current H200
environment has TransferQueue 0.1.7 and does not persist queued trajectories: a
mid-run restart loses the outstanding episodes already fetched by the
dataloader. In-process partial continuation is unaffected, and the final drain
avoids this overhang at a normally completed run.

```bash
# Three-update smoke, to exercise cross-step continuation.
./experiment.sh train-smoke taco

# Original synchronous trainer for a controlled comparison.
TRAINER_MODE=sync ./experiment.sh train taco
```

Default single-domain and `train all` run names end in `_partial_pool32`
(or the chosen pool size), separating checkpoints from synchronous and older
whole-batch warmup runs. Start a fresh run when switching dataset fetch batch
size; the older dataloader checkpoint used a different fetch size.
When specifying `RUN_NAME`, choose
a fresh experiment/checkpoint directory for the first partial run. To reuse
existing data and initialization in a new experiment, set `DATA_ROOT` and
`MODEL` explicitly. Existing synchronous checkpoints do not contain the V1
replay-buffer and generation state.

## Current protocol

- Domains: AWM, EnvScaler, TACO.
- Base model: Qwen3.5-4B.
- Trainable module: hybrid delta/residual attention virtual prefix.
- Code evaluation: HumanEval+, MBPP+, LiveCodeBench v6; thinking enabled; 16K completion limit.
- Tool evaluation: BFCL v3 and tau2 Airline/Retail.
- tau2 holds the external user simulator and NL judge fixed via `ARK_API_KEY` from `.env`.
- Code execution uses the official evaluators inside NsJail.

Implementation helpers live under `prefix_tuning/virtual_prefix/internal/`. They are private to `experiment.sh`; do not invoke them directly.

## Layout

- `experiment.sh`: the only training/inference/evaluation launcher.
- `prefix_tuning/virtual_prefix/`: current hybrid prefix implementation, export, composition, and runtime integration.
- `rollout/`: agent loop and environment adapters used by training.
- `benchmark/`: benchmark adapters and isolated code scoring.
- `traj_data/`: source trajectories.
- `outputs/`: checkpoints, exported prefixes, logs, and evaluation results.
- `verl/`, `OpenEnv/`: pinned upstream/runtime dependencies.

Activate the `kvcache` conda environment before direct development commands. The launcher activates it automatically.
