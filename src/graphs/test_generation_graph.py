"""
Test Case Generation graph (Parallelisation pattern).

    START --+-- generate_standard_cases --+
            |-- generate_edge_cases       +-- merge -- END
            +-- generate_error_cases      |
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.nodes.test_generation import (
    generate_edge_cases,
    generate_error_cases,
    generate_standard_cases,
    merge_testcases,
)
from src.states.test_generation import TestGenerationState


def build_test_generation_graph():
    """Build and compile the test-case-generation LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke()``.
    """
    workflow = StateGraph(TestGenerationState)

    # -- Nodes --
    workflow.add_node(
        "generate_standard_cases", generate_standard_cases,
    )
    workflow.add_node(
        "generate_edge_cases", generate_edge_cases,
    )
    workflow.add_node(
        "generate_error_cases", generate_error_cases,
    )
    workflow.add_node("merge_testcases", merge_testcases)

    # -- Parallel fan-out from START --
    workflow.add_edge(START, "generate_standard_cases")
    workflow.add_edge(START, "generate_edge_cases")
    workflow.add_edge(START, "generate_error_cases")

    # -- Fan-in to merge --
    workflow.add_edge("generate_standard_cases", "merge_testcases")
    workflow.add_edge("generate_edge_cases", "merge_testcases")
    workflow.add_edge("generate_error_cases", "merge_testcases")

    # -- To END --
    workflow.add_edge("merge_testcases", END)

    return workflow.compile()
