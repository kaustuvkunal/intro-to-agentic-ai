"""State schema for the Evaluation & Feedback (Conditional) workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationState(BaseModel):
    """Holds all data exchanged between evaluation nodes.

    Fields
    ------
    question : dict
        The programming question being evaluated.
    test_cases : list[dict]
        List of test cases to run against user code.
    user_response : str
        Learner's Python source code.
    result : str
        Binary outcome: ``"pass"`` or ``"fail"``.
    feedback : str
        Natural-language feedback message.
    passed_test_cases : list[dict]
        Test cases the learner passed.
    failed_test_cases : list[dict]
        Test cases the learner failed, with error details.
    """

    question: dict[str, str] = Field(
        description="The programming question being evaluated"
    )
    test_cases: list[dict[str, str]] = Field(
        description="List of test cases to run against user code"
    )
    user_response: str = Field(
        description="Learner's Python source code"
    )
    result: str = Field(
        default="pass",
        description="Binary outcome: 'pass' or 'fail'",
    )
    feedback: str = Field(
        default="",
        description="Natural-language feedback message",
    )
    passed_test_cases: list[dict[str, str]] = Field(
        default_factory=list,
        description="Test cases the learner passed",
    )
    failed_test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Test cases the learner failed, with error details",
    )
