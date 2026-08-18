"""
Core code-execution logic - framework-free.

Uses subprocess for isolation instead of the deprecated
langchain_experimental.PythonREPLTool.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import textwrap
from typing import Any


def execute_code_core(user_code: str, test_input: str) -> dict[str, Any]:
    """Execute the learner's Python code against a single test input.

    Parameters
    ----------
    user_code : str
        Full Python source containing at least one function definition.
    test_input : str
        A string evaluable by ``ast.literal_eval``.

    Returns
    -------
    dict
         ``{"success": True,  "output": <str>}``
    or
         ``{"success": False, "error":   <str>}``
     """
    try:
        tree = ast.parse(user_code)
        function_name = next(
             (
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
             ),
            None,
         )
    except SyntaxError as exc:
        return {
             "success": False,
             "error": f"Syntax error in user code: {exc}",
         }

    if function_name is None:
        return {
             "success": False,
             "error": "No function definition found",
         }

     # -- 2. Safely evaluate the test input --
    try:
        input_value = ast.literal_eval(test_input)
    except (ValueError, SyntaxError):
        return {
             "success": False,
             "error": f"Invalid test input: {test_input}",
         }

     # -- 3. Build a self-contained script --
    harness = textwrap.dedent(
        f"""\
        import json, sys, traceback

        {user_code}

        try:
            result = {function_name}({input_value!r})
            if isinstance(result, (dict, list)):
                print(json.dumps(result))
            else:
                print(repr(result))
        except Exception as exc:
            print(f"Error: {{str(exc)}}", file=sys.stderr)
            sys.exit(1)
        """
     )

     # -- 4. Run in a subprocess (isolated, no REPL needed) --
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
     ) as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
             [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
         )
    except subprocess.TimeoutExpired:
        return {
             "success": False,
             "error": "Code execution timed out (15 s limit)",
         }
    except Exception as exc:
        return {
             "success": False,
             "error": f"Subprocess failed: {exc}",
         }
    finally:
        import os

        os.unlink(tmp_path)

     # -- 5. Interpret the result --
    if proc.returncode != 0:
        error_msg = proc.stderr.strip()
        return {"success": False, "error": error_msg or "Unknown error"}

    raw_output = proc.stdout.strip()

     # -- 6. Try to parse as JSON (dict/list), else use raw string --
    try:
        parsed = json.loads(raw_output)
        result_str = json.dumps(parsed)
    except json.JSONDecodeError:
        result_str = raw_output

    return {"success": True, "output": result_str}