# BFCL v3

Official Berkeley Function Calling Leaderboard v3, wired into the project's
unified evaluation entry (`benchmark/eval/run_eval.py`). Generation, execution,
possible answers, checkers, and score aggregation all run **unchanged** from the
pinned official source, so results match the official leaderboard.

The official source is pinned to:

```text
cd9429ccf3d4d04156affe883c495b3b047e6b64
```

This is the direct parent of the upstream `BFCL V4 Release` commit
(`VERSION_PREFIX == BFCL_v3`).

## Why it is not a `TaskSuite`

The interactive benchmarks (`lifelong_db` / `lifelong_os`) run through the
generic `EvalRunner` episode loop. BFCL v3 is a batch pipeline
(`generate → evaluate → aggregate`) whose scoring — AST checker, stateful
multi-turn backends, relevance detection, category aggregation, scoreboard — is
owned by upstream. Re-implementing that as a project `Scorer` would drift from
the official leaderboard, so BFCL keeps the official engine and only shares the
project's entry point, endpoint/model config, and output tree.

## Setup (one-time environment bootstrap)

From the repository root:

```bash
source .venv/bin/activate
benchmark/bfcl_v3/setup.sh
```

`setup.sh` exports the pinned upstream subtree into `benchmark/bfcl_v3/official`
and installs only the missing BFCL imports with `--no-deps` (BFCL pins
NumPy 1.x / older vendor SDKs; the project keeps its own versions and does not
register BFCL's distribution metadata).

## Run (unified entry)

Start the model server as usual:

```bash
MODEL_PATH=/path/to/model ./start_services.sh start vllm
```

Then evaluate through the same entry as every other benchmark:

```bash
# Full official run
python -m benchmark.eval.run_eval --benchmark bfcl_v3 \
    --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
    --bfcl-model-path /path/to/model --bfcl-categories all --concurrency 100

# Only the non-live AST categories
python -m benchmark.eval.run_eval --benchmark bfcl_v3 \
    --model qwen3.5-4b --bfcl-model-path /path/to/model \
    --bfcl-categories non_live --concurrency 100

# Multi-turn only
python -m benchmark.eval.run_eval --benchmark bfcl_v3 \
    --model qwen3.5-4b --bfcl-model-path /path/to/model \
    --bfcl-categories multi_turn --concurrency 100

# Targeted generation from an official ID-selection file
python -m benchmark.eval.run_eval --benchmark bfcl_v3 \
    --model qwen3.5-4b --bfcl-model-path /path/to/model \
    --bfcl-run-ids-file /path/to/test_case_ids_to_generate.json
```

`--model` is the vLLM `--served-model-name` (also the API model id).
`--bfcl-model-path` is the local weights dir; it is only used as the official
OSS handler's tokenizer/config source. The API model id is passed separately
via `BFCL_API_MODEL_ID` (local patch `api_model_id.patch`, applied by
`setup.sh`), so no symlink is created. `--bfcl-model-key` selects the official
handler (`Qwen/Qwen3-4B-FC` by default) and stays fixed across checkpoints
(Base / Simulator / EnvScaler / AWM); only `--bfcl-model-path` and the vLLM
service change per experiment.

## Output

Official `result/` and `score/` land under `--output-dir` (default
`workspace/eval_logs/bfcl_v3/<categories>/<timestamp>`), the same tree as the
other benchmarks. The summary table is printed from `score/data_overall.csv`.

> Note: the pinned upstream `bfcl scores` subcommand hard-codes a
> `Non-Live Exec Acc` column that v3 removed, so it crashes. The runner prints
> the aggregate table directly from `data_overall.csv` instead, without
> patching upstream. When you evaluate a subset (e.g. `non_live`), the
> `Overall Acc` column counts the un-run groups as 0 — read the per-group
> columns (`data_non_live.csv` etc.) for subset runs.

## Key files

| File | Role |
| --- | --- |
| `benchmark/eval/run_eval.py` | Unified entry; `--benchmark bfcl_v3` dispatches to the official pipeline |
| `benchmark/eval/bfcl_v3.py` | Driver: translates CLI to official `generate/evaluate`, prints scores |
| `benchmark/bfcl_v3/setup.sh` | Exports the pinned official source, installs missing deps |
| `benchmark/bfcl_v3/official/` | Pinned upstream `bfcl_eval` source (unchanged) |
