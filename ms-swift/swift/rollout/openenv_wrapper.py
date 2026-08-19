# Copyright (c) ModelScope Contributors. All rights reserved.
"""Thin wrapper around OpenEnv's GenericEnvClient with reconnection support.

This wrapper only manages the WebSocket connection and forwards raw actions
to the server.  Action parsing (LLM text → dict) and observation formatting
(dict → LLM string) are handled by the :class:`OpenEnvScheduler` subclass,
not here.

env_config keys consumed:
    base_url (str): OpenEnv server URL (e.g. 'http://localhost:8000').
    reset_kwargs (dict, optional): Extra kwargs passed to client.reset().
"""
import os
import threading
import time
from typing import Any, Dict, Tuple

from swift.utils import get_logger

logger = get_logger()

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0

# WebSocket keepalive / message timeouts. Raised from openenv's 20s/60s defaults
# to survive long multi-turn rollouts: between turns, vLLM generation can leave
# the WebSocket idle long enough that a 20s pong timeout fires and one side
# closes the socket. The wrapper then reconnects to a fresh AWMEnvironment that
# was never reset, so every subsequent step returns
# "Sub-environment is not running. Call reset() first.".
_WS_PING_INTERVAL = float(os.environ.get('OPENENV_WS_PING_INTERVAL', '20.0'))
_WS_PING_TIMEOUT = float(os.environ.get('OPENENV_WS_PING_TIMEOUT', '300.0'))
_MESSAGE_TIMEOUT = float(os.environ.get('OPENENV_MESSAGE_TIMEOUT', '300.0'))


class OpenEnvWrapper:
    """Thin wrapper around ``GenericEnvClient`` with reconnection support.

    Unlike the previous version, this class does **not** inherit from
    ``Env`` and does **not** parse LLM text or format observations.
    All such logic lives in :class:`~swift.rollout.multi_turn.OpenEnvScheduler`
    and its subclasses.
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.base_url = env_config.get('base_url', 'http://localhost:8000')
        self.reset_kwargs = env_config.get('reset_kwargs', {})
        self._client = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_client(self):
        """Lazily create the OpenEnv client (sync wrapper)."""
        if self._client is None:
            from openenv.core.generic_client import GenericEnvClient

            client = GenericEnvClient(
                base_url=self.base_url,
                websocket_ping_interval_s=_WS_PING_INTERVAL,
                websocket_ping_timeout_s=_WS_PING_TIMEOUT,
                message_timeout_s=_MESSAGE_TIMEOUT,
            )
            sync_client = client.sync()
            sync_client.__enter__()
            self._client = sync_client
        return self._client

    def _reconnect_client(self):
        """Close old client (recover from connection loss).

        Only cleans up; the next _ensure_client() call will create a new connection.
        """
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                pass
            self._client = None

    def _reset_after_reconnect(self) -> None:
        """Re-reset the env after a reconnect (fallback).

        A reconnect opens a fresh WebSocket, which the server maps to a
        brand-new AWMEnvironment that was never reset -- without this, every
        retried step returns 'Sub-environment is not running. Call reset()
        first.' and the whole trajectory is lost. We re-reset here to revive
        the sub-env. Calls the sync client directly (NOT self.reset()) so it
        bypasses _call_with_retry and cannot recurse. Best-effort: a failure
        is logged but does not abort the retried call.

        Caveat: reset rewinds the env to its initial state, so the model's
        prior tool history no longer matches env state. This is a lossy
        fallback to avoid an all-'not running' (reward=0) trajectory, not a
        seamless recovery.
        """
        try:
            self._ensure_client().reset(**self.reset_kwargs)
        except Exception as e:
            logger.warning(f'OpenEnv post-reconnect reset failed: {e}')

    def _call_with_retry(self, fn_name: str, *args, **kwargs):
        """Call a client method with retry on connection failure."""
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                client = self._ensure_client()
                return getattr(client, fn_name)(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f'OpenEnv {fn_name} failed (attempt {attempt + 1}/{_MAX_RETRIES}): {e}')
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
                    self._reconnect_client()
                    # Reconnect -> fresh server-side AWMEnvironment that was
                    # never reset. Revive it so the retried step doesn't hit
                    # 'Sub-environment is not running'. Only for step: a reset
                    # retry's next attempt is itself a reset, so an extra
                    # reset here would just needlessly kill+restart the sub-env.
                    if fn_name == 'step':
                        self._reset_after_reconnect()
        raise last_exc

    # ------------------------------------------------------------------
    # Public API (synchronous — called from async scheduler via run_in_executor
    # or directly since GenericEnvClient.sync() is already blocking)
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Any, Any]:
        """Reset the OpenEnv environment.

        Returns:
            (raw_observation, metadata) — unformatted, as returned by the server.
        """
        result = self._call_with_retry('reset', **self.reset_kwargs)
        return result.observation, getattr(result, 'metadata', None)

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Any, float, bool, Any]:
        """Execute one step in the OpenEnv environment.

        Args:
            action_dict: Pre-parsed action dict (e.g. ``{'answer': '7'}``).

        Returns:
            (raw_observation, reward, done, metadata) — unformatted.
        """
        result = self._call_with_retry('step', action_dict)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        return result.observation, reward, done, getattr(result, 'metadata', None)

    def close(self):
        """Close the OpenEnv client connection."""
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                logger.debug('OpenEnv client close failed', exc_info=True)
            self._client = None
