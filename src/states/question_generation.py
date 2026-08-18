"""State schema for the Question Generation (Sequential) workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QPGenerationState(BaseModel):
    """Holds all data exchanged between question-generation nodes.

    Fields
    ------
    topic : str
        User-provided programming topic (e.g. *"Dictionaries"*).
    num_questions : int
        Total number of questions to generate.
    questions : list[dict]
        Generated question JSON objects.
    question_validation : str
        ``"VALID"`` / ``"INVALID"`` from ``validate_questions``.
    error_message : str
        Human-readable error when validation fails.
    """

    topic: str = Field(
        description="User-provided topic (e.g., 'Dictionaries')"
    )
    num_questions: int = Field(
        description="Total questions to generate"
    )
    questions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of JSON questions generated",
    )
    question_validation: str = Field(
        default="",
        description="'VALID' or 'INVALID' - result of validation",
    )
    error_message: str = Field(
        default="",
        description="Error message when validation fails",
    )
