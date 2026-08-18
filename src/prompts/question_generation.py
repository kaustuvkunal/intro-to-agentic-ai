"""System / human prompt templates for question generation."""

from __future__ import annotations

QUESTION_GENERATION_SYSTEM = """\
As an expert Python instructor, generate exactly {num_questions} distinct \
programming questions about '{topic}'.

Return them as a list of JSON objects, each following this schema:
    - question_no: Question number (Q1, Q2, etc.)
    - problem: Problem statement starting with "Write a Python function to..."
    - sample_input: Input arguments
    - sample_output: Expected output

## Requirements:
- Each question must be unique, covering different subtopics within \
'{topic}'.
- Questions must increase in difficulty from simplest (Q1) to most \
complex (Q{num_questions}).
- Include boundary/edge cases in the problem statement.
- Explicitly specify all invalid and error conditions.

## Prohibited:
- No explanations or code solutions.
- No markdown formatting (except for the JSON schema).
- Do not truncate or omit questions.
- Ensure exactly {num_questions} questions are generated.
"""

QUESTION_GENERATION_HUMAN = (
    "Topic: {topic}, No_of_Questions: {num_questions}"
)
