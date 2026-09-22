"""tau2-bench 驱动：把统一评测入口对接到官方 tau2 批处理管线。

与 BFCL v3 同样的整合哲学（官方源码固定 commit + 统一入口 + vLLM 复用 + 判分留
官方），差异在于 tau2 必须跑在自己的 Python 3.12 venv 里（见 setup.sh）：

  - tau2 是带 user-simulator 的多轮对话 benchmark，判分（DB 末态 hash + communicate
    checks，pass^k 指标）全在官方代码，原样运行；
  - 所有 LLM 调用走 litellm。被测 agent 指向 start_services.sh 拉起的现有
    vLLM；user-simulator 与自然语言断言 judge 默认固定到外部模型，避免它们随
    被测模型变化而污染对比结果；
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

# User simulator 与 Retail 自然语言断言 judge 默认走火山方舟（OpenAI 兼容）。
# API key 只从环境变量读取，绝不写入代码/命令行/产物。
DEFAULT_TAU2_USER_MODEL = "openai/deepseek-v4-pro-ga-260813"
DEFAULT_TAU2_USER_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TAU2_USER_API_KEY_ENV = "ARK_API_KEY"
DEFAULT_TAU2_JUDGE_MODEL = "openai/deepseek-v4-pro-ga-260813"
DEFAULT_TAU2_JUDGE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TAU2_JUDGE_API_KEY_ENV = "ARK_API_KEY"


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

    # 被测 agent 固定走本地 vLLM，并把路由信息显式写进参数，避免被进程级
    # OPENAI_API_BASE/OPENAI_API_KEY（供外部 user/judge 使用）覆盖。
    base_url = args.openai_base_url
    model_alias = args.model
    _check_server(base_url, model_alias)

    llm_model = f"openai/{model_alias}"
    max_completion_tokens = (
        args.llm_max_completion_tokens
        if args.llm_max_completion_tokens is not None
        else DEFAULT_TAU2_MAX_COMPLETION_TOKENS
    )
    agent_llm_args = {
        "api_base": base_url,
        "api_key": args.api_key or "EMPTY",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_completion_tokens,
    }
    agent_thinking = getattr(args, "tau2_agent_thinking", False)
    user_thinking = getattr(args, "tau2_user_thinking", False)
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
    user_api_key_env = (
        getattr(args, "tau2_user_api_key_env", None)
        or DEFAULT_TAU2_USER_API_KEY_ENV
    )
    user_api_key = os.getenv(user_api_key_env)
    if user_api_key:
        user_llm_model = (
            getattr(args, "tau2_user_model", None) or DEFAULT_TAU2_USER_MODEL
        )
        user_base_url = (
            getattr(args, "tau2_user_base_url", None)
            or DEFAULT_TAU2_USER_BASE_URL
        )
        user_llm_args = {
            "api_base": user_base_url,
            "temperature": getattr(args, "tau2_user_temperature", 0.0),
            "top_p": getattr(args, "tau2_user_top_p", 1.0),
            "max_tokens": user_max_completion_tokens,
        }
        if not user_thinking:
            user_llm_args["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        if getattr(args, "tau2_require_external_user", False):
            raise SystemExit(
                f"tau2 外部 user simulator 必需，但环境变量 {user_api_key_env} 未设置"
            )
        print(
            f"[tau2] 未设置 {user_api_key_env}，user simulator 回退本地 vLLM "
            f"({llm_model})。"
        )
        user_llm_model = llm_model
        user_base_url = base_url
        user_llm_args = {
            "api_base": base_url,
            "api_key": args.api_key or "EMPTY",
            "temperature": args.temperature,
            "top_p": args.top_p,
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
        str(venv_dir / "bin" / "python"), str(tau2_root / "launch.py"), "run",
        "--domain", args.tau2_domain,
        "--agent-llm", llm_model,
        "--agent-llm-args", agent_llm_args_json,
        "--user-llm", user_llm_model,
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
    # 外部 key 只通过子进程环境传递。命令行与产物只包含 api_base，不包含 key。
    env = dict(os.environ)
    # Retail includes natural-language assertions.  Upstream defaults their
    # judge to GPT-4.1; route this judge to an external OpenAI-compatible
    # model (Volcano Ark deepseek by default) whenever its API key is present
    # in the environment, falling back to the local vLLM model otherwise.
    # The key is only read from the environment so it never appears in the
    # repo, process args, or artifacts.
    judge_api_key_env = getattr(args, "tau2_judge_api_key_env", None) or DEFAULT_TAU2_JUDGE_API_KEY_ENV
    judge_api_key = os.getenv(judge_api_key_env)
    judge_thinking = getattr(args, "tau2_judge_thinking", False)
    if judge_api_key:
        judge_llm_model = getattr(args, "tau2_judge_model", None) or DEFAULT_TAU2_JUDGE_MODEL
        judge_base_url = getattr(args, "tau2_judge_base_url", None) or DEFAULT_TAU2_JUDGE_BASE_URL
        nl_judge_args = {
            "api_base": judge_base_url,
            "temperature": getattr(args, "tau2_judge_temperature", 0.0),
            "top_p": getattr(args, "tau2_judge_top_p", 1.0),
            "max_tokens": max_completion_tokens,
        }
        # 思考长度默认压到最低：Ark 的 budget_tokens 不生效（实测 budget=1 反而
        # 多于默认），disabled 是唯一可靠的最低档（0 思考 token）。
        if not judge_thinking:
            nl_judge_args["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        if getattr(args, "tau2_require_external_judge", False):
            raise SystemExit(
                f"tau2 外部 NL judge 必需，但环境变量 {judge_api_key_env} 未设置"
            )
        print(
            f"[tau2] 未设置 {judge_api_key_env}，NL 断言 judge 回退本地 vLLM 模型 "
            f"({llm_model})。"
        )
        judge_llm_model = llm_model
        nl_judge_args = {
            **agent_llm_args,
            "max_tokens": max_completion_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    external_credentials = []
    if user_api_key:
        external_credentials.append((user_api_key, user_base_url, "user simulator"))
    if judge_api_key:
        external_credentials.append((judge_api_key, judge_base_url, "NL judge"))
    if external_credentials:
        first_key, first_base, first_role = external_credentials[0]
        for key, endpoint, role in external_credentials[1:]:
            if key != first_key or endpoint != first_base:
                raise SystemExit(
                    "tau2 当前子进程只支持 user simulator 与 NL judge 共用一组外部 "
                    f"OpenAI 凭据；{first_role} 与 {role} 的 key/base 不一致"
                )
        env["OPENAI_API_KEY"] = first_key
        env["OPENAI_API_BASE"] = first_base
    else:
        env["OPENAI_API_KEY"] = args.api_key or "EMPTY"
        env["OPENAI_API_BASE"] = base_url
    env["TAU2_LLM_NL_ASSERTIONS"] = judge_llm_model
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
    print(f"Agent:        {llm_model} @ {base_url}")
    print(
        f"User:         {user_llm_model} @ {user_base_url}"
        + (" (external)" if user_api_key else " (fallback)")
    )
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
    print(
        f"NL judge:     {judge_llm_model}"
        + (f" @ {nl_judge_args['api_base']}" if judge_api_key else " (fallback)")
        + f" thinking={'on' if judge_thinking else 'off'}"
    )
    print(f"Output:       {save_to}")
    print("=" * 72)

    proc = subprocess.run(cmd, cwd=str(official_root), env=env)
    results_json = Path(save_to) / "results.json"
    if results_json.exists():
        print(f"\n[tau2] 结果与 pass^k 指标见: {results_json}")
        print(f"[tau2] 浏览: {tau2_bin} view --dir {save_to}")
    if proc.returncode:
        return proc.returncode
    if not results_json.exists():
        print("[tau2] Missing results.json")
        return 1
    saved = json.loads(results_json.read_text())
    simulations = saved.get("simulations", [])
    expected = len(saved.get("tasks", [])) * args.tau2_num_trials
    infrastructure_errors = [s for s in simulations
                             if s.get("termination_reason") == "infrastructure_error"]
    if not expected or len(simulations) != expected or infrastructure_errors:
        print(f"[tau2] Incomplete run: expected={expected}, saved={len(simulations)}, "
              f"infrastructure_errors={len(infrastructure_errors)}")
        return 1
    return 0
