"""
Evaluation & Feedback graph (Conditional pattern).

    START -> evaluate_user_response
                  |
                  +-- [pass] -> success_feedback      -> END
                  +-- [fail] -> improvement_feedback -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.nodes.evaluation import (
    classify_result,
    evaluate_user_response,
    improvement_feedback,
    success_feedback,
)
from src.states.evaluation import EvaluationState


def build_evaluation_graph():
    """Build and compile the evaluation LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke()``.
    """
    workflow = StateGraph(EvaluationState)

    # -- Nodes --
    workflow.add_node(
        "evaluate_user_response", evaluate_user_response,
    )
    workflow.add_node("success_feedback", success_feedback)
    workflow.add_node(
        "improvement_feedback", improvement_feedback,
    )

    # -- Entry point --
    workflow.add_edge(START, "evaluate_user_response")

    # -- Conditional routing --
    workflow.add_conditional_edges(
        "evaluate_user_response",
        classify_result,
        {
            "pass": "success_feedback",
            "fail": "improvement_feedback",
        },
    )

    # -- Both feedback nodes terminate --
    workflow.add_edge("success_feedback", END)
    workflow.add_edge("improvement_feedback", END)

    return workflow.compile()
