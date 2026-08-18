"""
Tool layer.

Two exposure paths share one core:

    code_execution.py   ->  pure Python  execute_code_core()
    langchain_tool.py   ->  @tool wrapper  (LangGraph nodes)
    mcp_server.py       ->  fastmcp server (any MCP client)
"""

from src.tools.code_execution import execute_code_core
from src.tools.langchain_tool import execute_code

__all__ = ["execute_code", "execute_code_core"]
