# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free helpers for inspecting MCP-compatible tool servers."""

from typing import Any, Dict

from ..utils import run_async_safely


def get_server_tools(mcp_server: Any) -> Dict[str, Any]:
    """Return tools from a FastMCP-compatible server without importing FastMCP.

    FastMCP is optional for OpenEnv's Gym/WebSocket transport.  Keeping this
    duck-typed helper outside ``mcp_environment`` lets environments that
    implement ``Environment`` directly use the HTTP/WebSocket server without
    installing the MCP runtime.
    """
    if hasattr(mcp_server, "get_tools"):
        result = run_async_safely(mcp_server.get_tools())
        if isinstance(result, dict):
            return result
    if hasattr(mcp_server, "list_tools"):
        tools_list = run_async_safely(mcp_server.list_tools())
        return {tool.name: tool for tool in tools_list}
    return {}
