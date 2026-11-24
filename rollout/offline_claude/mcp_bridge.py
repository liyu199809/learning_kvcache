"""Expose only tools advertised by this episode's environment, over MCP stdio."""
import asyncio
import json
import socket
import sys

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


def request(payload):
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(sys.argv[1])
        conn.sendall((json.dumps(payload) + "\n").encode())
        reply = json.loads(conn.makefile("rb").readline())
    if "error" in reply:
        raise ValueError(reply["error"])
    return reply["value"]


async def main():
    async def list_tools():
        return [types.Tool(**tool) for tool in request({"op": "list"})]

    async def call_tool(name, arguments):
        value = request({"op": "call", "name": name, "arguments": arguments})
        return [types.TextContent(type="text", text=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))]

    if hasattr(Server, "list_tools"):
        server = Server("episode")
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
    else:
        async def on_list_tools(ctx, params):
            return types.ListToolsResult(tools=await list_tools())

        async def on_call_tool(ctx, params):
            return types.CallToolResult(content=await call_tool(params.name, params.arguments))

        server = Server("episode", on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
