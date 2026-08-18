"""Unit tests for the Test Case Generation workflow."""

from __future__ import annotations

import pytest

from src.graphs.test_generation_graph import (
    build_test_generation_graph,
)
from src.states.test_generation import (
    TestCase,
    TestCaseList,
    TestGenerationState,
)

SAMPLE_QUESTION = {
    "question_no": "Q1",
    "problem": "Write a Python function to reverse a string.",
    "sample_input": '"hello"',
    "sample_output": '"olleh"',
}


@pytest.fixture()
def tg_graph():
    """Return a compiled test-generation graph."""
    return build_test_generation_graph()


def test_testcase_schema():
    """TestCase model should enforce required fields."""
    tc = TestCase(
        input="hello",
        expected_output="olleh",
        testcase_type="Standard Valid",
        difficulty="easy",
    )
    assert tc.input == "hello"
    assert tc.difficulty == "easy"


def test_testcase_list_schema():
    """TestCaseList should hold a list of TestCase."""
    tl = TestCaseList(test_cases=[])
    assert tl.test_cases == []


def test_graph_runs(tg_graph):
    """End-to-end smoke test (requires a running LLM provider)."""
    result = tg_graph.invoke(
        TestGenerationState(
            question=SAMPLE_QUESTION,
            no_of_test_cases=2,
        )
    )
    assert "test_cases" in result
    assert len(result["test_cases"]) > 0
