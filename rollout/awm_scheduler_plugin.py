"""AWM multi-turn scheduler plugin for ms-swift GRPO online rollout.

Adapts swift's OpenEnvScheduler to the Agent World Model (AWM) OpenEnv
server, which uses OpenAI-native function calling (parallel tool calls).

Env lifecycle uses swift's OpenEnvWrapper (GenericEnvClient.sync()), NOT a
raw async AWMEnv. The wrapper's SyncEnvClient owns a dedicated background
event loop, so the WebSocket - and the server-side AWMEnvironment that holds
the running sub-env subprocess - stays alive across turns. A raw async
AWMEnv binds its WebSocket to whichever loop first calls connect(); when
swift re-enters on_turn_end on a different loop (after vLLM generation),
EnvClient._connect_async treats the live socket as stale and reconnects ->
server /ws spawns a fresh AWMEnvironment that was never reset -> step()
returns "Sub-environment is not running. Call reset() first.". The sync
wrapper's fixed background loop is what prevents that reconnect.

Per rollout trajectory:
  on_trajectory_start: create one OpenEnvWrapper per request, reset(scenario,
                       task_idx) to a clean state. `messages` and `tools`
                       already arrive complete on each request (carried from
                       the dataset row by samples2requests ->
                       to_infer_request), so this hook only manages env
                       lifecycle.
  on_turn_end:         parse the model's tool_calls, execute them in parallel
                       against the AWM env via wrapper.step, format tool
                       responses, and decide whether the episode is done
                       (model stopped calling tools => final answer).
  step:                append the tool responses for the next turn.

Env config comes from each dataset row's `env_config` column:
    {"scenario": "...", "task_idx": 0, "awm_base_url": "http://localhost:8899"}

Register via:
    swift rlhf --rlhf_type grpo \
        --external_plugins rollout/awm_scheduler_plugin.py \
        --multi_turn_scheduler awm_scheduler ...
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from openenv.core.env_server.mcp_types import CallToolAction
from swift.rollout.multi_turn import OpenEnvScheduler, multi_turns


class AWMScheduler(OpenEnvScheduler):
    """Multi-turn scheduler for AWM OpenEnv function-calling rollout.

    Inherits ``_create_env`` (returns an ``OpenEnvWrapper``) from
    ``OpenEnvScheduler`` and overrides the turn hooks for AWM's
    function-calling protocol (parallel tool_calls, verify-on-done).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._envs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------
    async def _close_env(self, uuid: str) -> None:
        wrapper = self._envs.pop(uuid, None)
        if wrapper is not None:
            try:
                await asyncio.to_thread(wrapper.close)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    async def on_trajectory_start(self, requests) -> None:
        """Reset each trajectory's AWM env to a clean state via the sync wrapper.

        `messages` and `tools` already arrive complete on every request (the
        dataset row carries both, forwarded by samples2requests ->
        to_infer_request), so this hook only spins up and resets the env that
        on_turn_end executes tool calls against.
        """

        async def _init(req) -> None:
            cfg = (req.data_dict or {}).get('env_config', {}) if hasattr(req, 'data_dict') else {}
            scenario = cfg.get('scenario')
            task_idx = cfg.get('task_idx')
            base_url = cfg.get('awm_base_url', 'http://localhost:8899')
            if scenario is None or task_idx is None:
                raise ValueError(f"env_config missing scenario/task_idx: {cfg}")

            wrapper = self._create_env({
                'base_url': base_url,
                'reset_kwargs': {'scenario': scenario, 'task_idx': task_idx},
            })
            await asyncio.to_thread(wrapper.reset)
            self._envs[req.uuid] = wrapper

        await asyncio.gather(*[_init(r) for r in requests])

    async def on_turn_end(self, infer_request, response_choice, current_turn: int):
        uuid = infer_request.uuid
        wrapper = self._envs.get(uuid)
        if wrapper is None:
            return {'done': True, 'rollout_infos': {}}

        msg = response_choice.message
        tool_calls = getattr(msg, 'tool_calls', None) or []

        # No tool calls => model produced a final answer; episode ends.
        if not tool_calls:
            final_answer = (msg.content or '').strip()
            reward = await self._verify(wrapper, final_answer)
            await self._close_env(uuid)
            return {'done': True,
                    'rollout_infos': {'total_reward': reward,
                                      'final_answer': final_answer}}

        # Execute up to 3 tool calls in parallel; collect tool messages.
        # wrapper.step is synchronous (blocking WebSocket I/O on the wrapper's
        # background loop), so route each call through asyncio.to_thread to
        # keep swift's event loop free. The action is a raw dict - the server
        # deserializes it via its "type" discriminator (call_tool -> CallToolAction).
        tool_calls = tool_calls[:3]

        async def _exec(tc):
            fn = tc.get('function', {}) if isinstance(tc, dict) else tc.function
            name = fn.get('name') if isinstance(fn, dict) else fn.name
            args = fn.get('arguments') if isinstance(fn, dict) else fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tcid = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', '')
            action = {'tool_name': name, 'arguments': args}
            obs, _r, _d, _m = await asyncio.to_thread(wrapper.step, CallToolAction(**action))
            context = self._tool_text(obs) + f"\n\nYou have {self.max_turns-current_turn-1} remaining opportunities for parallel tool calls. Once the count hits 0, you must respond to the user directly whether the task is completed or not."
            return {'role': 'tool', 'tool_call_id': tcid,
                    'name': name, 'content': context}

        tool_msgs = await asyncio.gather(*[_exec(tc) for tc in tool_calls])

        # Turn budget exhausted: swift force-stops the episode after this turn
        # (multi_turn.py: should_stop |= current_turn >= max_turns). The model
        # never emitted a plain-text final answer, but the task may STILL be
        # complete in the DB - verify (code mode) grades DB side-effects, not
        # the answer text. Verify the current env state now (after the tool
        # calls above have mutated the DB) and close, so the reward reflects
        # the student's real rollout instead of defaulting to 0, and the env
        # doesn't leak.
        if self.max_turns and current_turn >= self.max_turns:
            final_answer = (msg.content or '').strip()
            reward = await self._verify(wrapper, final_answer)
            await self._close_env(uuid)
            return {'done': True,
                    'rollout_infos': {'total_reward': reward,
                                      'final_answer': final_answer}}

        # Stash for step() to append.
        self._pending = getattr(self, '_pending', {})
        self._pending[uuid] = tool_msgs
        return {'done': False, 'rollout_infos': {}}

    def step(self, infer_request, response_choice, current_turn: int):
        uuid = infer_request.uuid
        pending = getattr(self, '_pending', {}).pop(uuid, [])
        if pending:
            infer_request.messages.extend(pending)
        return {'infer_request': infer_request}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_text(obs: Any) -> str:
        """Render a tool-call observation (dict from GenericEnvClient) as text.

        Mirrors rollout.common.execute_tool_call's observation-reading logic,
        but operates on the raw dict returned by the sync GenericEnvClient
        (AWMObservation fields serialized via serialize_observation).
        """
        if not isinstance(obs, dict):
            return json.dumps(obs, ensure_ascii=False)
        tool_result = obs.get('tool_result')
        if tool_result is not None:
            if isinstance(tool_result, str):
                return tool_result
            return json.dumps(tool_result, ensure_ascii=False)
        if obs.get('error'):
            return f"Error: {obs['error']}"
        return json.dumps(obs, ensure_ascii=False)

    async def _verify(self, wrapper, final_answer: str) -> float:
        action = {
            'type': 'call_tool',
            'tool_name': 'verify',
            'arguments': {'verifier_mode': 'code', 'final_answer': final_answer},
        }
        try:
            obs, _r, _d, _m = await asyncio.to_thread(wrapper.step, action)
            return 1.0 if (obs or {}).get('reward_type') == 'complete' else 0.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
multi_turns['awm_scheduler'] = AWMScheduler