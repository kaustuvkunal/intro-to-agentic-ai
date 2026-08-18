"""State schemas for the Test Case Generation (Parallelisation) workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case."""

    input: str = Field(description="Input for the test case")
    expected_output: str = Field(description="Expected output")
    testcase_type: str = Field(
        description="Type: Standard Valid, Edge, Error/Invalid"
    )
    difficulty: str = Field(
        description="Difficulty of test case (easy or hard)"
    )


class TestCaseList(BaseModel):
    """Wrapper so ``llm.with_structured_output`` returns a list."""

    test_cases: list[TestCase] = Field(
        description="List of generated test cases"
    )


class TestGenerationState(BaseModel):
    """Holds all data exchanged between test-case-generation nodes.

    Fields
    ------
    question : dict
        The question dict to generate test cases for.
    no_of_test_cases : int
        Number of test cases to generate per category.
    test_cases : list[dict]
        Merged list of all generated test cases.
    standard_test_cases : list[dict]
        Standard / valid test cases.
    edge_test_cases : list[dict]
        Edge / boundary test cases.
    error_test_cases : list[dict]
        Error / invalid test cases.
    """

    question: dict[str, str] = Field(
        default_factory=dict,
        description="The question dict to generate test cases for",
    )
    no_of_test_cases: int = Field(
        description="Total number of test cases to generate per category"
    )
    test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Merged list of all generated test cases",
    )
    standard_test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Standard / valid test cases",
    )
    edge_test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Edge / boundary test cases",
    )
    error_test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Error / invalid test cases",
    )
