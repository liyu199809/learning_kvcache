"""Opt-in integration test for a live AWM OpenEnv service.

Run with::

    AWM_INTEGRATION=1 pytest -q \
      verl/tests/experimental/agent_loop/test_awm_agent_loop_openenv_integration.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request

import pytest

from rollout.verl_awm_agent_loop import AWMAgentLoop


pytestmark = pytest.mark.skipif(os.getenv("AWM_INTEGRATION") != "1", reason="requires live AWM service")


def get_stats(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/stats", timeout=5) as response:
        return json.load(response)


def test_live_openenv_session_is_persistent_and_closed():
    base_url = os.getenv("AWM_BASE_URL", "http://localhost:8899").rstrip("/")
    before_sessions = get_stats(base_url)["total_sessions"]

    async def exercise_session():
        loop = object.__new__(AWMAgentLoop)
        loop._env_client = None
        loop._env_lock = asyncio.Lock()
        loop._reset_succeeded = False
        loop._transport_failed = False
        loop._transport_error = ""
        loop._last_assistant_text = "Environment inspected successfully."
        loop._verify_reward_type = "not_verified"
        loop._close_error = ""
        try:
            await loop._reset_openenv(
                loop._normalize_env_config(
                    {
                        "scenario": "marketplace_1",
                        "task_idx": 3,
                        "awm_base_url": base_url,
                    }
                )
            )
            assert loop._reset_succeeded
            assert get_stats(base_url)["total_sessions"] == before_sessions + 1
            response = await loop._call_openenv_tool("get_current_user_profile", {})
            assert "jdoe" in (response.text or "")
            reward = await loop._verify()
            assert reward in (0.0, 1.0)
            assert not loop._transport_error
        finally:
            await loop._close_openenv()

    asyncio.run(exercise_session())
    time.sleep(0.25)
    assert get_stats(base_url)["total_sessions"] == before_sessions
