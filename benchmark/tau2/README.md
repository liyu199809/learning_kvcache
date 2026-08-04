# tau2-bench

Official [tau2-bench](https://github.com/sierra-research/tau2-bench) (τ²-bench),
wired into the project's unified evaluation entry
(`benchmark/eval/run_eval.py`). Simulation, the user-simulator, and scoring
(DB end-state hash + communicate checks, `pass^k`) all run **unchanged** from
the pinned official source.

Pinned to:

```text
363133ada1936491fb5bcec33cd62c3518a99f65   # v1.0.1
```

## Why a dedicated venv (unlike BFCL)

BFCL v3 runs in the project's main `.venv` through `PYTHONPATH`. tau2 cannot:

- it requires **Python >=3.12,<3.14**, while the main env is Python 3.11;
- every LLM call goes through **litellm** (`>=1.80.15,<1.82.7`), which drags in
  `openai`/`httpx`/`tokenizers` pins that would fight vLLM / transformers.

So `setup.sh` creates an isolated **uv venv (Python 3.12)** under
`benchmark/tau2/.venv` and editable-installs the pinned official source there.
This does not touch the main env, vLLM, MS-Swift, or the training stack. The
existing vLLM from `start_services.sh` is reused over HTTP.

## Why it is not a `TaskSuite`

Like BFCL, tau2 is a batch pipeline whose scoring is owned by upstream (stateful
domain DBs, user-simulator dialogue, DB/communicate reward). Re-implementing it
as a project `Scorer` would drift from the official metric, so tau2 keeps the
official engine and only shares the project's entry point, endpoint/model
config, and output tree.

## Setup (one-time)

From the repository root:

```bash
benchmark/tau2/setup.sh
```

## Run (unified entry)

Start the model server as usual (main env):

```bash
MODEL_PATH=/path/to/model ./start_services.sh start vllm
```

Then evaluate through the same entry as every other benchmark (uses the main
`.venv` only to parse args; the actual tau2 run happens in the tau2 venv):

```bash
# Airline domain, full base split
python -m benchmark.eval.run_eval --benchmark tau2 \
    --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
    --tau2-domain airline --tau2-split base --concurrency 8

# Quick smoke: mock domain, 3 tasks
python -m benchmark.eval.run_eval --benchmark tau2 \
    --model qwen3.5-4b --tau2-domain mock --tau2-num-tasks 3 --concurrency 4

# pass^k with multiple trials
python -m benchmark.eval.run_eval --benchmark tau2 \
    --model qwen3.5-4b --tau2-domain retail --tau2-num-trials 4 --concurrency 8
```

Both the **agent** and the **user-simulator** are routed to the local vLLM via
litellm's `openai/<model>` provider with `api_base` pointing at
`--openai-base-url` (also set as `OPENAI_API_BASE` for the subprocess).
`--model` must match the vLLM `--served-model-name`.

## Output

Official results land at `<output-dir>/results.json`
(default `workspace/eval_logs/tau2/<domain>/<timestamp>`), the same tree as the
other benchmarks. `pass^k` metrics are printed by the official run at the end.
Browse a run with:

```bash
benchmark/tau2/.venv/bin/tau2 view --dir <output-dir>
```

> Note: `--auto-review`, NL-assertion-gated tasks, and banking_knowledge
> `ACTION` tasks call hardcoded judge models (`gpt-4.1`, `claude-opus-4-5`).
> The default airline/retail/telecom scoring (`reward_basis=[DB, COMMUNICATE]`)
> only calls the agent + user models, so it stays fully on the local vLLM.

## Key files

| File | Role |
| --- | --- |
| `benchmark/eval/run_eval.py` | Unified entry; `--benchmark tau2` dispatches here |
| `benchmark/eval/tau2.py` | Driver: routes agent+user LLM to vLLM, invokes official `tau2 run` |
| `benchmark/tau2/setup.sh` | Creates the Python 3.12 uv venv, pins & installs official source |
| `benchmark/tau2/official/` | Pinned upstream tau2 source (unchanged) |
| `benchmark/tau2/.venv/` | Isolated Python 3.12 environment (git-ignored) |
