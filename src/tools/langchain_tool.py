"""
LangChain ``@tool`` wrapper around ``execute_code_core``.

Used inside LangGraph nodes so the tool participates in the
LangChain tool-calling / tracing ecosystem.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.tools.code_execution import execute_code_core


@tool
def execute_code(user_code: str, test_input: str) -> dict[str, Any]:
    """Execute the learner's Python code with the given test input.

    Parameters
    ----------
    user_code : str
        Full Python source containing at least one function definition.
    test_input : str
        A string that can be evaluated by ``ast.literal_eval``.

    Returns
    -------
    dict
        ``{"success": True,  "output": <str>}``
    or
        ``{"success": False, "error":  <str>}``
    """
    return execute_code_core(user_code, test_input)
