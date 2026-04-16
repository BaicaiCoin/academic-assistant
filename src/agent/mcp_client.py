"""
Persistent MCP client singleton.

Initialised once at FastAPI startup (via server.py lifespan) and kept alive
for the lifetime of the process. On init, all tools exposed by each MCP server
are loaded and injected into ALL_TOOLS / TOOL_MAP so the Brain Agent's Planner
can see and choose them.
"""

from __future__ import annotations

import os
from langchain_mcp_adapters.client import MultiServerMCPClient

_client: MultiServerMCPClient | None = None


def _build_server_config() -> dict:
    return {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_TOKEN"]},
            "transport": "stdio",
        },
    }


async def init_mcp() -> None:
    """
    Start all MCP server subprocesses and load their tools.
    Injects loaded tools into ALL_TOOLS and TOOL_MAP so the Planner sees them.
    Call once at application startup, before the first request.
    """
    global _client

    _client = MultiServerMCPClient(_build_server_config())
    mcp_tools = await _client.get_tools()

    # Inject into the Brain Agent's static tool registry
    from agent.tools_and_schemas import ALL_TOOLS, TOOL_MAP
    for t in mcp_tools:
        if t.name not in TOOL_MAP:
            ALL_TOOLS.append(t)
            TOOL_MAP[t.name] = t

    print(f"MCP tools loaded: {[t.name for t in mcp_tools]}")


async def close_mcp() -> None:
    """No-op: langchain-mcp-adapters 0.1.0 manages subprocess lifecycle internally."""
    global _client
    _client = None
