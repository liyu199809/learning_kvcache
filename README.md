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

# Run one-domain training or a one-step smoke run.
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
