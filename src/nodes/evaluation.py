"""
Evaluation & Feedback nodes (Conditional pattern).

    START -> evaluate_user_response
                     |
                     +-- [pass] -> success_feedback          -> END
                     +-- [fail] -> improvement_feedback -> END

IMPORTANT: this module calls ``execute_code_core`` directly (not the
``@tool`` wrapper) so the return value is a real ``dict``, not a
JSON string.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.config.llm_client import get_langchain_llm
from src.prompts.evaluation import (
    EVALUATION_SYSTEM,
    IMPROVEMENT_FEEDBACK_SYSTEM,
    SUCCESS_FEEDBACK_SYSTEM,
)
from src.states.evaluation import EvaluationState
from src.tools.code_execution import execute_code_core

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _normalise(value: str) -> str:
    """Strip surrounding quotes / whitespace for comparison."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1]
    return v.strip()


def _outputs_match(actual: str, expected: str) -> bool:
    """Fast direct comparison before falling back to LLM."""
    a = _normalise(actual)
    e = _normalise(expected)

    # 1. Exact match after normalisation
    if a == e:
        return True

    # 2. JSON comparison (handles dict / list key-order differences)
    try:
        if json.loads(a) == json.loads(e):
            return True
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Case-insensitive match
    if a.lower() == e.lower():
        return True

    return False


# ------------------------------------------------------------------
# Node: Evaluate user response
# ------------------------------------------------------------------


def evaluate_user_response(
    state: EvaluationState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Evaluate user code against all test cases.

    For each test case the learner's code is executed via
    ``execute_code_core`` (subprocess, not ``@tool``).  A fast
    direct comparison is attempted first; the LLM is used only
    as a semantic-comparison fallback.
    """
    llm = get_langchain_llm()

    print("--- EVALUATING USER RESPONSE ---")

    prompt = ChatPromptTemplate.from_template(EVALUATION_SYSTEM)

    passed: list[dict] = []
    failed: list[dict] = []

    for test_case in state.test_cases:
        test_input = test_case["input"]
        expected_output = test_case["expected_output"]
        testcase_type = test_case.get("testcase_type", "")
        difficulty = test_case.get("difficulty", "")

        # -- Execute learner code (direct call, returns a real dict) --
        execution_result = execute_code_core(
            user_code=state.user_response,
            test_input=test_input,
        )

        # Debug: print what we got
        print(
            "   [TC] input="
            + repr(test_input)
            + "  success="
            + str(execution_result.get("success"))
            + "  output="
            + repr(
                execution_result.get(
                    "output",
                    execution_result.get("error", "?"),
                )
            )
        )

        detail = {"test_case": test_case, "error": ""}

        # -- Handle execution errors --
        if not execution_result.get("success"):
            expected_has_error = (
                "error" in expected_output.lower()
                or "valueerror" in expected_output.lower()
            )
            actual_error = execution_result.get("error", "")

            if expected_has_error and actual_error:
                passed.append(test_case)
                continue

            detail["error"] = actual_error or "Execution failed"
            failed.append(detail)
            continue

        # -- Compare outputs --
        actual_output = execution_result.get("output", "")

        # Fast path: direct comparison
        if _outputs_match(actual_output, expected_output):
            passed.append(test_case)
            continue

        # Slow path: LLM semantic comparison
        prompt_input = {
            "problem": state.question.get("problem", ""),
            "user_code": state.user_response,
            "test_input": test_input,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "testcase_type": testcase_type,
            "difficulty": difficulty,
        }

        try:
            evaluation_chain = prompt | llm
            raw = evaluation_chain.invoke(prompt_input).content
            eval_result = json.loads(raw)
            if eval_result.get("is_correct"):
                passed.append(test_case)
            else:
                detail["error"] = eval_result.get(
                    "error_message", "Output mismatch"
                )
                failed.append(detail)
        except json.JSONDecodeError:
            detail["error"] = "Failed to parse LLM evaluation response"
            failed.append(detail)
        except Exception as exc:
            detail["error"] = f"LLM evaluation error: {exc}"
            failed.append(detail)

    result_str: str = "fail" if failed else "pass"

    print(
        "--- RESULT: "
        + result_str
        + "   passed="
        + str(len(passed))
        + "  failed="
        + str(len(failed))
        + " ---"
    )

    return {
        "result": result_str,
        "passed_test_cases": passed,
        "failed_test_cases": failed,
    }


# ------------------------------------------------------------------
# Node: Success feedback
# ------------------------------------------------------------------


def success_feedback(
    state: EvaluationState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Generate positive feedback when all test cases pass."""
    llm = get_langchain_llm()

    print("--- SUCCESS FEEDBACK ---")

    prompt = ChatPromptTemplate.from_template(SUCCESS_FEEDBACK_SYSTEM)

    prompt_input = {
        "problem": state.question.get("problem", ""),
        "sample_input": state.question.get("sample_input", ""),
        "sample_output": state.question.get("sample_output", ""),
        "user_code": state.user_response,
    }

    feedback_chain = prompt | llm
    feedback = feedback_chain.invoke(prompt_input).content

    return {"feedback": feedback}


# ------------------------------------------------------------------
# Node: Improvement feedback
# ------------------------------------------------------------------


def improvement_feedback(
    state: EvaluationState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Generate improvement feedback when test cases fail."""
    llm = get_langchain_llm()

    print("--- IMPROVE FEEDBACK ---")

    prompt = ChatPromptTemplate.from_template(
        IMPROVEMENT_FEEDBACK_SYSTEM,
    )

    failed_str = "\n".join(
        "Input: "
        + d["test_case"]["input"]
        + ", Type: "
        + d["test_case"].get("testcase_type", "")
        + ", Difficulty: "
        + d["test_case"].get("difficulty", "")
        + ", Error: "
        + d["error"]
        for d in state.failed_test_cases
    )

    prompt_input = {
        "problem": state.question.get("problem", ""),
        "sample_input": state.question.get("sample_input", ""),
        "sample_output": state.question.get("sample_output", ""),
        "user_code": state.user_response,
        "failed_test_cases": failed_str,
    }

    feedback_chain = prompt | llm
    feedback = feedback_chain.invoke(prompt_input).content

    return {"feedback": feedback}


# ------------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------------


def classify_result(
    state: EvaluationState,
) -> Literal["pass", "fail"]:
    """Route to the appropriate feedback node."""
    print("--- CLASSIFYING RESULT ---")
    return state.result
