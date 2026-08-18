"""
Test Case Generation nodes (Parallelisation pattern).

    START --+-- generate_standard_cases --+
            |-- generate_edge_cases       +-- merge_testcases -- END
            +-- generate_error_cases      |
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from src.config.llm_client import get_langchain_llm
from src.prompts.test_generation import (
    EDGE_TC_SYSTEM,
    ERROR_TC_SYSTEM,
    STANDARD_TC_SYSTEM,
    TC_HUMAN,
)
from src.states.test_generation import (
    TestCase,
    TestCaseList,
    TestGenerationState,
)


# -- Helper --
def _test_cases_to_dicts(
    test_cases: list[TestCase],
) -> list[dict[str, str]]:
    """Convert a list of ``TestCase`` models to plain dicts."""
    return [
        {
            "input": tc.input,
            "expected_output": tc.expected_output,
            "testcase_type": tc.testcase_type,
            "difficulty": tc.difficulty,
        }
        for tc in test_cases
    ]


# -- Node: Standard / Valid cases --
def generate_standard_cases(
    state: TestGenerationState,
) -> dict:
    """Generate standard/valid test cases for the given question."""
    llm = get_langchain_llm()

    print("--- GENERATING STANDARD/VALID TEST CASES ---")
    num = state.no_of_test_cases

    system_msg = STANDARD_TC_SYSTEM.format(num_test_cases=num)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", TC_HUMAN),
    ])

    testcase_chain = (
        prompt | llm.with_structured_output(schema=TestCaseList)
    )

    try:
        result = testcase_chain.invoke({"question": state.question})
        test_cases = _test_cases_to_dicts(result.test_cases)
        print(
            f"--- NUMBER OF STANDARD/VALID GENERATED IS --> {num}"
        )
        print("--- GENERATED STANDARD/VALID TEST CASE ---")
        return {"standard_test_cases": test_cases}
    except Exception as exc:
        print(f"Error during standard valid cases generation: {exc}")
        return {
            "error_message": [
                f"Standard valid cases failed: {exc}"
            ]
        }


# -- Node: Edge cases --
def generate_edge_cases(
    state: TestGenerationState,
) -> dict:
    """Generate edge/boundary test cases for the given question."""
    llm = get_langchain_llm()

    print("--- GENERATING EDGE TEST CASES ---")
    num = state.no_of_test_cases

    system_msg = EDGE_TC_SYSTEM.format(num_test_cases=num)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", TC_HUMAN),
    ])

    testcase_chain = (
        prompt | llm.with_structured_output(schema=TestCaseList)
    )

    try:
        result = testcase_chain.invoke({"question": state.question})
        test_cases = _test_cases_to_dicts(result.test_cases)
        print(
            f"--- NUMBER OF EDGE CASE GENERATED IS --> "
            f"{len(test_cases)}"
        )
        print("--- GENERATED EDGE TEST CASES ---")
        return {"edge_test_cases": test_cases}
    except Exception as exc:
        print(f"Error during edge test cases generation: {exc}")
        return {
            "error_message": [
                f"Edge test cases creation failed: {exc}"
            ]
        }


# -- Node: Error / Invalid cases --
def generate_error_cases(
    state: TestGenerationState,
) -> dict:
    """Generate error/invalid test cases for the given question."""
    llm = get_langchain_llm()

    print("--- GENERATING ERROR/INVALID TEST CASES ---")
    num = state.no_of_test_cases

    system_msg = ERROR_TC_SYSTEM.format(num_test_cases=num)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", TC_HUMAN),
    ])

    testcase_chain = (
        prompt | llm.with_structured_output(schema=TestCaseList)
    )

    try:
        result = testcase_chain.invoke({"question": state.question})
        test_cases = _test_cases_to_dicts(result.test_cases)
        print(
            f"--- NUMBER OF ERROR/INVALID CASE GENERATED "
            f"is ---> {len(test_cases)}"
        )
        print("--- GENERATED ERROR/INVALID TEST CASES ---")
        return {"error_test_cases": test_cases}
    except Exception as exc:
        print(f"Error during error test cases generation: {exc}")
        return {
            "error_message": [
                f"Error test cases creation failed: {exc}"
            ]
        }


# -- Node: Merge --
def merge_testcases(
    state: TestGenerationState,
) -> dict:
    """Merge test cases from all parallel branches."""
    print("--- MERGING TEST CASES ---")

    merged: list[dict] = []
    merged.extend(state.standard_test_cases)
    merged.extend(state.edge_test_cases)
    merged.extend(state.error_test_cases)

    if not merged:
        print("Warning: No test cases were merged.")

    print(f"--- TOTAL MERGED TEST CASES: {len(merged)} ---")

    # JSON round-trip for plain-dict serialisability
    merged_json = json.dumps(merged, indent=4)
    return {"test_cases": json.loads(merged_json)}
