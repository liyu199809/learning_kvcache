"""
Agentic harness runner.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_FILE_POINTER = (
    "The trajectory is provided as one JSON object per line at `trajectory.jsonl` "
    "in the current directory. Each line has the schema "
    '`{{"turn_idx": int, "action": str, "observation": str}}`, in chronological '
    "order. There are {num_turns} turns. The file is large; use file tools "
    "(head/tail/grep/jq/python) to read and search it rather than loading it all "
    "at once."
)
_FOOTER = (
    "\n\nWrite the entire formatted response — every `Answer[i]:` line, in order, "
    "from 1 to {num_questions} — to `answers.txt` in the current directory. Do not "
    "include any other text in that file. Do not invent facts that are not in the "
    "trajectory. Do not ask for human help."
)


_OPENEND_INTRO = (
    "Please answer the following questions based on the task description "
    "and agent trajectory above. For each question, provide a direct and "
    "concise answer."
)
_OPENEND_INSTR = "Please provide answers in the following format:"
_MCQ_INTRO = (
    "Please answer the following multiple-choice questions based on "
    "the task description and agent trajectory above."
)
_MCQ_INSTR = (
    "For each question, select all correct options and respond using "
    "the format (A), (B), (C), or (D). "
    "If multiple options are correct, combine them like (A)(B)."
)


def _build_instruction(task: str, num_turns: int, questions: list[str], mcq_mode: bool) -> str:
    """The upstream longcontext prompt with the trajectory delivered as a file.

    The task/trajectory header and the ## Questions / ## Instructions / Answer[i]
    section are byte-for-byte the upstream longcontext wording; only the
    trajectory pointer (codex greps the file) and the write-to-answers.txt footer
    are agent-harness specific.
    """
    questions_block = "\n".join(
        f"Question {i}: {q}\n" for i, q in enumerate(questions, 1)
    )
    n = len(questions)
    if mcq_mode:
        section_intro, instructions = _MCQ_INTRO, _MCQ_INSTR
        answer_slots = "\n".join(
            f"Answer[{i}]: [(A)/(B)/(C)/(D) or combination such as (A)(B)]"
            for i in range(1, n + 1)
        )
    else:
        section_intro, instructions = _OPENEND_INTRO, _OPENEND_INSTR
        answer_slots = "\n".join(
            f"Answer[{i}]: [your answer here]" for i in range(1, n + 1)
        )
    return (
        f"## Task Description\n{task}\n\n"
        f"## Agent Trajectory\n"
        f"The following is a step-by-step trajectory of the agent's actions and observations:\n\n"
        f"{_FILE_POINTER.format(num_turns=num_turns)}"
        f"\n\n## Questions\n{section_intro}\n\n"
        f"{questions_block}\n"
        f"## Instructions\n{instructions}\n\n"
        f"{answer_slots}"
        f"{_FOOTER.format(num_questions=n)}"
    )


def _write_codex_home(codex_home: Path, base_url: str | None, api_key: str) -> None:
    """
    Isolated CODEX_HOME with API-key auth and optional base URL.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": api_key}))
    if base_url:
        (codex_home / "config.toml").write_text(f'openai_base_url = "{base_url}"\n')


def run_codex_episode(
    *,
    trajectory: list[dict],
    task: str,
    questions: list[str],
    mcq_mode: bool,
    model: str,
    base_url: str | None,
    api_key: str | None,
    reasoning_effort: str = "high",
    codex_cmd: str = "codex",
    timeout: float = 1800.0,
    max_retries: int = 3,
) -> str:
    """Run one agentic codex session over an episode's raw trajectory.

    Returns the text written to answers.txt (the `Answer[i]:` block), which the
    caller parses exactly as it parses a normal model completion.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the codex provider")

    instruction = _build_instruction(task, len(trajectory), questions, mcq_mode)
    inner_model = model.split("/")[-1]

    last_err: Exception | None = None
    for attempt in range(max_retries):
        workdir = Path(tempfile.mkdtemp(prefix="ama_codex_"))
        codex_home = workdir / ".codex-home"
        try:
            with (workdir / "trajectory.jsonl").open("w", encoding="utf-8") as f:
                for turn in trajectory:
                    f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            _write_codex_home(codex_home, base_url, api_key)

            env = {
                **os.environ,
                "CODEX_HOME": str(codex_home),
                "OPENAI_API_KEY": api_key,
            }
            if base_url:
                env["OPENAI_BASE_URL"] = base_url

            cmd = [
                *shlex.split(codex_cmd),
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--model",
                inner_model,
                "--enable",
                "unified_exec",
                "-c",
                f"model_reasoning_effort={reasoning_effort}",
                "--",
                instruction,
            ]

            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            answers_path = workdir / "answers.txt"
            if answers_path.exists():
                text = answers_path.read_text(encoding="utf-8").strip()
                if text:
                    return text

            raise RuntimeError(
                f"codex did not write answers.txt (rc={proc.returncode}). "
                f"stderr tail: {proc.stderr[-2000:]!r}"
            )
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 30 * (2**attempt)
                print(f"codex attempt {attempt + 1}/{max_retries} failed: {e}; "
                      f"retrying in {wait}s...")
                time.sleep(wait)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    raise RuntimeError(f"codex failed after {max_retries} attempts: {last_err}")


def _claude_auth_model_env(model: str | None) -> dict[str, str]:
    """Auth + model env for the claude CLI.

    Claude Code has no per-call API key like OpenAI; it authenticates via the
    local subscription login, or one of several env credentials. We pass through
    whatever is present and drop empty values so the CLI picks its own method (a
    logged-in machine needs none). The model is selected via ANTHROPIC_MODEL, and
    when a custom gateway (ANTHROPIC_BASE_URL) is set we also pin the alias models.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    env = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "ANTHROPIC_AUTH_TOKEN": os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
        "CLAUDE_CODE_OAUTH_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
        "ANTHROPIC_BASE_URL": base_url,
    }
    if model:
        m = model if base_url else model.split("/")[-1]
        env["ANTHROPIC_MODEL"] = m
        if base_url:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = m
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = m
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = m
    return {k: v for k, v in env.items() if v}


def run_claude_episode(
    *,
    trajectory: list[dict],
    task: str,
    questions: list[str],
    mcq_mode: bool,
    model: str | None,
    claude_cmd: str = "claude",
    timeout: float = 1800.0,
    max_retries: int = 3,
) -> str:
    """Run one agentic Claude Code session over an episode's raw trajectory.

    Same contract as run_codex_episode: writes the full raw trajectory.jsonl,
    hands claude the same instruction, and returns the `Answer[i]:` block. Auth
    and model are handled the harbor way (see _claude_auth_model_env): the local
    Claude Code login is used unless an env credential is present.
    """
    instruction = _build_instruction(task, len(trajectory), questions, mcq_mode)
    extra_env = _claude_auth_model_env(model)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        workdir = Path(tempfile.mkdtemp(prefix="ama_claude_"))
        try:
            # Full raw trajectory — no truncation; the agent greps what it needs.
            with (workdir / "trajectory.jsonl").open("w", encoding="utf-8") as f:
                for turn in trajectory:
                    f.write(json.dumps(turn, ensure_ascii=False) + "\n")

            cmd = [*shlex.split(claude_cmd), "-p", "--dangerously-skip-permissions"]

            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env={**os.environ, **extra_env},
                input=instruction,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            answers_path = workdir / "answers.txt"
            if answers_path.exists():
                text = answers_path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            # Fallback: claude -p prints the final message to stdout.
            out = (proc.stdout or "").strip()
            if "Answer[" in out:
                return out

            raise RuntimeError(
                f"claude produced no answers (rc={proc.returncode}). "
                f"stderr tail: {proc.stderr[-2000:]!r}"
            )
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 30 * (2**attempt)
                print(f"claude attempt {attempt + 1}/{max_retries} failed: {e}; "
                      f"retrying in {wait}s...")
                time.sleep(wait)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    raise RuntimeError(f"claude failed after {max_retries} attempts: {last_err}")
