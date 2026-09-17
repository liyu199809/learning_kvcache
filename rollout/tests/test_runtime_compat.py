from __future__ import annotations

import pytest

from rollout.json_compat import loads
from rollout.verl_awm_agent_loop import AWMAgentLoop


@pytest.mark.parametrize(
    ("raw", "service", "base_url"),
    [
        (
            {"scenario": "marketplace_1", "task_idx": 0, "service": "awm"},
            "awm",
            "http://127.0.0.1:8899",
        ),
        (
            {"scenario": "env_148_rl", "task_idx": 1, "service": "envscaler"},
            "envscaler",
            "http://127.0.0.1:8900",
        ),
        (
            {"scenario": "taco", "task_idx": 2, "service": "codejudge"},
            "codejudge",
            "http://127.0.0.1:8901",
        ),
        (
            {
                "scenario": "taco",
                "task_idx": 3,
                "awm_base_url": "http://localhost:8901/",
            },
            "codejudge",
            "http://localhost:8901",
        ),
    ],
)
def test_three_service_config_and_legacy_routing(raw, service, base_url):
    normalized = AWMAgentLoop._normalize_env_config(raw)
    assert normalized["service"] == service
    assert normalized["base_url"] == base_url


def test_unknown_service_is_rejected():
    with pytest.raises(ValueError, match="Unsupported OpenEnv service"):
        AWMAgentLoop._normalize_env_config(
            {"scenario": "x", "task_idx": 0, "service": "unknown"}
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"name": "tool",}', {"name": "tool"}),
        ('{name: "tool"}', {"name": "tool"}),
        ("```json\n{\"value\": 1}\n```", {"value": 1}),
    ],
)
def test_stdlib_json_repair_fallback(text, expected):
    assert loads(text) == expected
