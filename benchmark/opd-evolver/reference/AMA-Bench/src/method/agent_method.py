"""
Agent-harness methods (codex, claude-code, mem0, etc.)
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any

from src.method.base_method import BaseMethod


class AgentHarnessMethod(BaseMethod):
    """Base class for external tool-using agent harnesses."""

    def __init__(
        self,
        config_path: str = None,
        client: Any = None,
        embedding_engine: Any = None,
        **kwargs,
    ):
        self.config = self._load_config(config_path) if config_path else {}
        self.client = client
        self.requires_embedding = False
        self.requires_network = True

    @abstractmethod
    def run_episode(
        self,
        *,
        trajectory: list[dict],
        task: str,
        questions: list[str],
        mcq_mode: bool,
    ) -> str:
        """Run the agent over one episode; return the raw `Answer[i]:` block.

        The agent reads the raw trajectory itself and answers all questions in a
        single session. Parsing of the returned block is the interface's job, so
        every harness shares the same answer-extraction path.
        """

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        raise RuntimeError(
            f"{type(self).__name__} is an agent harness, driven per-episode by "
            "MemoryQAInterface.run_episode, not memory_construction."
        )

    def memory_retrieve(self, memory: Any, question: str) -> str:
        raise RuntimeError(
            f"{type(self).__name__} is an agent harness, driven per-episode by "
            "MemoryQAInterface.run_episode, not memory_retrieve."
        )


class CodexAgentMethod(AgentHarnessMethod):
    """Codex CLI as a tool-using agent."""

    def run_episode(
        self,
        *,
        trajectory: list[dict],
        task: str,
        questions: list[str],
        mcq_mode: bool,
    ) -> str:
        from src.agent_harness import run_codex_episode

        cfg = self.client.config
        return run_codex_episode(
            trajectory=trajectory,
            task=task,
            questions=questions,
            mcq_mode=mcq_mode,
            model=self.client.model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort=cfg.get("reasoning_effort", "high"),
            codex_cmd=cfg.get("codex_cmd", "codex"),
            timeout=self.client._timeout,
        )


class ClaudeCodeAgentMethod(AgentHarnessMethod):
    """Claude Code CLI as a tool-using agent."""

    def run_episode(
        self,
        *,
        trajectory: list[dict],
        task: str,
        questions: list[str],
        mcq_mode: bool,
    ) -> str:
        from src.agent_harness import run_claude_episode

        cfg = self.client.config
        return run_claude_episode(
            trajectory=trajectory,
            task=task,
            questions=questions,
            mcq_mode=mcq_mode,
            model=self.client.model,
            claude_cmd=cfg.get("claude_cmd", "claude"),
            timeout=self.client._timeout,
        )
