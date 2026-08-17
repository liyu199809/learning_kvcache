"""tau2-bench 驱动：把统一评测入口对接到官方 tau2 批处理管线。

与 BFCL v3 同样的整合哲学（官方源码固定 commit + 统一入口 + vLLM 复用 + 判分留
官方），差异在于 tau2 必须跑在自己的 Python 3.12 venv 里（见 setup.sh）：

  - tau2 是带 user-simulator 的多轮对话 benchmark，判分（DB 末态 hash + communicate
    checks，pass^k 指标）全在官方代码，原样运行；
  - 所有 LLM 调用走 litellm，用 openai/ provider + api_base 把 agent 与
    user-simulator 两个模型都指向 start_services.sh 拉起的现有 vLLM；
  - 官方产物落到统一 output-dir：--save-to 传绝对路径时，官方的
    ``DATA_DIR / "simulations" / <save_to>`` 会被 pathlib 折叠成该绝对路径本身，
    从而与其它 benchmark 同处一棵产物树，且不干扰 TAU2_DATA_DIR 的任务加载。

tau2 只提供 ``tau2`` 控制台命令（无 __main__），故以其专用 venv 的 tau2 可执行
文件子进程调用。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


DEFAULT_TAU2_MAX_COMPLETION_TOKENS = 4096
DEFAULT_TAU2_USER_MAX_COMPLETION_TOKENS = 4096
DEFAULT_TAU2_MAX_STEPS = 50


def _tau2_root() -> Path:
    """benchmark/tau2 目录（官方源码、专用 venv、setup.sh 所在处）。"""
    return Path(__file__).resolve().parents[1] / "tau2"


def _ensure_ready(tau2_root: Path, venv_dir: Path, official_root: Path) -> None:
    """venv 或官方源码缺失时运行 setup.sh。"""
    tau2_bin = venv_dir / "bin" / "tau2"
    if tau2_bin.exists() and (official_root / "src" / "tau2" / "cli.py").exists():
        return
    setup = tau2_root / "setup.sh"
    if not setup.exists():
        raise SystemExit(f"tau2 环境未就绪且未找到 setup.sh: {setup}")
    subprocess.run([str(setup)], check=True)


def _check_server(base_url: str, model_alias: str) -> None:
    import urllib.request

    url = f"{base_url.rstrip('/')}/models"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    model_ids = [item["id"] for item in data.get("data", [])]
    if model_alias not in model_ids:
        raise SystemExit(
            f"vLLM model 不匹配：期望 {model_alias!r}，实际 {model_ids!r}。"
            " 请先用 ./start_services.sh start vllm 启动对应模型。"
        )
    print(f"[tau2] Using vLLM model: {model_alias}")


def run_tau2(args) -> int:
    """统一入口的 tau2 分发目标。args 为 run_eval 的 argparse.Namespace。"""
    tau2_root = _tau2_root()
    venv_dir = Path(os.getenv("TAU2_VENV", str(tau2_root / ".venv")))
    official_root = Path(os.getenv("TAU2_OFFICIAL_ROOT", str(tau2_root / "official")))

    _ensure_ready(tau2_root, venv_dir, official_root)

    tau2_bin = venv_dir / "bin" / "tau2"

    # litellm 路由：openai/ provider + api_base 指向本地 vLLM，agent 与 user 同源。
    base_url = args.openai_base_url
    model_alias = args.model
    _check_server(base_url, model_alias)

    llm_model = f"openai/{model_alias}"
    max_completion_tokens = (
        args.llm_max_completion_tokens
        if args.llm_max_completion_tokens is not None
        else DEFAULT_TAU2_MAX_COMPLETION_TOKENS
    )
    common_llm_args = {
        "api_base": base_url,
        "api_key": args.api_key or "EMPTY",
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    agent_thinking = getattr(args, "tau2_agent_thinking", False)
    user_thinking = getattr(args, "tau2_user_thinking", False)
    agent_llm_args = {
        **common_llm_args,
        "max_tokens": max_completion_tokens,
    }
    if not agent_thinking:
        agent_llm_args["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    # The user simulator only needs to produce short conversational turns. Its
    # cap remains separate so explicitly enabling agent thinking does not make
    # user turns unexpectedly verbose.
    user_max_completion_tokens = min(
        max_completion_tokens, DEFAULT_TAU2_USER_MAX_COMPLETION_TOKENS
    )
    user_llm_args = {
        **common_llm_args,
        "max_tokens": user_max_completion_tokens,
    }
    if not user_thinking:
        user_llm_args["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    agent_llm_args_json = json.dumps(agent_llm_args)
    user_llm_args_json = json.dumps(user_llm_args)
    max_steps = (
        args.tau2_max_steps
        if args.tau2_max_steps is not None
        else DEFAULT_TAU2_MAX_STEPS
    )

    # 官方产物落统一 output-dir：--save-to 传绝对路径即被 pathlib 折叠为该路径。
    save_to = str(Path(args.output_dir).resolve())
    Path(save_to).mkdir(parents=True, exist_ok=True)

    cmd = [
        str(tau2_bin), "run",
        "--domain", args.tau2_domain,
        "--agent-llm", llm_model,
        "--agent-llm-args", agent_llm_args_json,
        "--user-llm", llm_model,
        "--user-llm-args", user_llm_args_json,
        "--task-split-name", args.tau2_split,
        "--num-trials", str(args.tau2_num_trials),
        "--max-concurrency", str(args.concurrency),
        "--max-steps", str(max_steps),
        "--save-to", save_to,
    ]
    if args.tau2_num_tasks is not None:
        cmd += ["--num-tasks", str(args.tau2_num_tasks)]
    if args.task_ids:
        cmd += ["--task-ids", *str(args.task_ids).replace(",", " ").split()]
    # litellm 仍要求存在 API key（本地 vLLM 用占位）；同时通过环境变量兜底，
    # 覆盖 openai/ provider 的 base_url/key（与 --agent-llm-args 双保险）。
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", args.api_key or "EMPTY")
    env["OPENAI_API_BASE"] = base_url
    # Retail includes natural-language assertions.  Upstream defaults their
    # judge to GPT-4.1, which is unavailable when the whole benchmark is run
    # against a local OpenAI-compatible endpoint.  Route this judge through the
    # same model and keep thinking disabled so every tau2 LLM role follows the
    # requested protocol.
    nl_judge_args = {
        **common_llm_args,
        "max_tokens": max_completion_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    env["TAU2_LLM_NL_ASSERTIONS"] = llm_model
    env["TAU2_LLM_NL_ASSERTIONS_ARGS"] = json.dumps(nl_judge_args)
    # tau2 runs one synchronous LiteLLM call per simulation worker.  Size the
    # shared HTTPX pool above the worker count so --concurrency is not silently
    # capped by the upstream default of ten connections.
    http_max_connections = max(32, args.concurrency * 2)
    http_max_keepalive_connections = max(16, args.concurrency)
    env["TAU2_LLM_HTTP_MAX_CONNECTIONS"] = str(http_max_connections)
    env["TAU2_LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS"] = str(
        http_max_keepalive_connections
    )

    print("=" * 72)
    print("tau2-bench (official pipeline)")
    print("=" * 72)
    print(f"Domain:       {args.tau2_domain}  split={args.tau2_split}")
    print(f"Agent/User:   {llm_model} @ {base_url}")
    print(f"Trials:       {args.tau2_num_trials}   concurrency={args.concurrency}")
    print(f"Max steps:    {max_steps}")
    print(
        f"Max tokens:   agent={max_completion_tokens} "
        f"user={user_max_completion_tokens}"
    )
    print(
        "Thinking:     "
        f"agent={'on' if agent_thinking else 'off'} "
        f"user={'on' if user_thinking else 'off'}"
    )
    print(
        "HTTP pool:    "
        f"connections={http_max_connections} "
        f"keepalive={http_max_keepalive_connections}"
    )
    print(f"Output:       {save_to}")
    print("=" * 72)

    proc = subprocess.run(cmd, cwd=str(official_root), env=env)
    results_json = Path(save_to) / "results.json"
    if results_json.exists():
        print(f"\n[tau2] 结果与 pass^k 指标见: {results_json}")
        print(f"[tau2] 浏览: {tau2_bin} view --dir {save_to}")
    return proc.returncode
