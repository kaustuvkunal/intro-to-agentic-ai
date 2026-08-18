"""
MCP (Model Context Protocol) server exposing ``execute_code``.

Any MCP-compatible client (Claude Desktop, VS Code Copilot Chat,
LangGraph agent, custom Python client) can connect and call the
tool without any LangChain dependency.

Run
---
    # stdio (default - for local MCP clients)
    uv run python -m mls_agents.tools.mcp_server

    # SSE (for remote / browser-based MCP clients)
    uv run python -m mls_agents.tools.mcp_server --transport sse --port 8000

Claude Desktop config (claude_desktop_config.json)
--------------------------------------------------
    {
        "mcpServers": {
            "mls-code-executor": {
                "command": "uv",
                "args": ["run", "python", "-m",
                         "mls_agents.tools.mcp_server"]
            }
        }
    }
"""

from __future__ import annotations

import argparse
from typing import Any

from fastmcp import FastMCP
from mls_agents.config.settings import settings
from mls_agents.tools.code_execution import execute_code_core

# -- Create the MCP server --
mcp = FastMCP("mls-code-executor")


@mcp.tool()
def execute_code(user_code: str, test_input: str) -> dict[str, Any]:
    """Execute a learner's Python function against a single test input.

    Parameters
    ----------
    user_code : str
        Full Python source containing at least one function definition.
    test_input : str
        A string evaluable by ``ast.literal_eval``.

    Returns
    -------
    dict
        ``{"success": True,  "output": "..."}``
    or
        ``{"success": False, "error":  "..."}``
    """
    print("--- EXECUTING CODE VIA MCP SERVER ---")
    return execute_code_core(user_code, test_input)


@mcp.tool()
def list_available_tools() -> list[str]:
    """Return the list of available tool names in this MCP server."""
    return ["execute_code", "list_available_tools"]


# -- CLI entry-point --
def main() -> None:
    """Parse CLI args and run the MCP server."""
    parser = argparse.ArgumentParser(
        description="MLS code-executor MCP server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=settings.mcp_transport,
        help="MCP transport (default: from settings / .env)",
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_host,
        help="Bind host for sse/http transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_port,
        help="Bind port for sse/http transport",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        print("[MCP] Starting 'mls-code-executor' on stdio")
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        print(
            f"[MCP] Starting 'mls-code-executor' "
            f"on sse {args.host}:{args.port}"
        )
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print(
            f"[MCP] Starting 'mls-code-executor' "
            f"on http {args.host}:{args.port}"
        )
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
