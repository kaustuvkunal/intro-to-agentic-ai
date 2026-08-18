"""
Question Generation graph (Sequential pattern).

    generate_questions -- validate_questions -- END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.nodes.question_generation import (
    generate_questions,
    validate_questions,
)
from src.states.question_generation import QPGenerationState


def build_question_generation_graph():
    """Build and compile the question-generation LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke()``.
    """
    workflow = StateGraph(QPGenerationState)

    # -- Nodes --
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("validate_questions", validate_questions)

    # -- Edges --
    workflow.set_entry_point("generate_questions")
    workflow.add_edge("generate_questions", "validate_questions")
    workflow.add_edge("validate_questions", END)

    return workflow.compile()
