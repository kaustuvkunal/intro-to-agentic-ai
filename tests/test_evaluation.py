"""Unit tests for the Evaluation & Feedback workflow."""

from __future__ import annotations

import pytest

from src.graphs.evaluation_graph import (
    build_evaluation_graph,
)
from src.states.evaluation import EvaluationState

SAMPLE_QUESTION = {
    "question_no": "Q1",
    "problem": "Write a Python function to reverse a string.",
    "sample_input": '"hello"',
    "sample_output": '"olleh"',
}

PASSING_CODE = "def reverse_string(s): return s[::-1]\n"
FAILING_CODE = "def reverse_string(s): return s\n"

SAMPLE_TEST_CASES = [
    {
        "input": '"hello"',
        "expected_output": '"olleh"',
        "testcase_type": "Standard Valid",
        "difficulty": "easy",
    },
]


@pytest.fixture()
def eval_graph():
    """Return a compiled evaluation graph."""
    return build_evaluation_graph()


def test_passing_case(eval_graph):
    """A correct implementation should pass."""
    result = eval_graph.invoke(
        EvaluationState(
            question=SAMPLE_QUESTION,
            test_cases=SAMPLE_TEST_CASES,
            user_response=PASSING_CODE,
        )
    )
    assert result["result"] == "pass"
    assert result["feedback"]  # non-empty


def test_failing_case(eval_graph):
    """An incorrect implementation should fail."""
    result = eval_graph.invoke(
        EvaluationState(
            question=SAMPLE_QUESTION,
            test_cases=SAMPLE_TEST_CASES,
            user_response=FAILING_CODE,
        )
    )
    assert result["result"] == "fail"
    assert result["feedback"]
