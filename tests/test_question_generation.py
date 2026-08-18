"""Unit tests for the Question Generation workflow."""

from __future__ import annotations

import pytest

from src.graphs.question_generation_graph import (
    build_question_generation_graph,
)
from src.states.question_generation import QPGenerationState


@pytest.fixture()
def qg_graph():
    """Return a compiled question-generation graph."""
    return build_question_generation_graph()


def test_state_schema_defaults(qg_graph):
    """QPGenerationState should have sensible defaults."""
    state = QPGenerationState(
        topic="strings",
        num_questions=1,
    )
    assert state.questions == []
    assert state.question_validation == ""
    assert state.error_message == ""


def test_graph_runs(qg_graph):
    """End-to-end smoke test (requires a running LLM provider)."""
    result = qg_graph.invoke(
        QPGenerationState(
            topic="Dictionaries",
            num_questions=1,
        )
    )
    assert result["questions"]  # non-empty
    assert result["question_validation"] in ("VALID", "INVALID")
