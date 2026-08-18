
"""
Question Generation nodes (Sequential pattern).

    generate_questions -> validate_questions -> END
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from src.config.llm_client import get_langchain_llm
from src.prompts.question_generation import (
    QUESTION_GENERATION_HUMAN,
    QUESTION_GENERATION_SYSTEM,
)
from src.states.question_generation import QPGenerationState


def generate_questions(
    state: QPGenerationState,
) -> QPGenerationState:
    """Generate programming questions for the given topic.

    Calls the provider-agnostic LLM via ``get_langchain_llm()`` so
    the actual provider is chosen at runtime from env / settings.

    Parameters
    ----------
    state : QPGenerationState
        Current workflow state with ``topic`` and ``num_questions``.

    Returns
    -------
    QPGenerationState
        Updated state with ``questions`` populated.
    """
    llm = get_langchain_llm()

    print(
        f"--- GENERATING QUESTIONS FOR THE TOPIC---> {state.topic}"
    )

    topic = state.topic
    num_questions = state.num_questions

    system_msg = QUESTION_GENERATION_SYSTEM.format(
        topic=topic,
        num_questions=num_questions,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", QUESTION_GENERATION_HUMAN),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "topic": topic,
        "num_questions": num_questions,
    })

    # -- Parse JSON response --
    print(response.content)
    try:
        questions = json.loads(response.content)
        if isinstance(questions, list) and all(
            isinstance(q, dict) for q in questions
        ):
            state.questions = questions
        else:
            print(
                "Warning: LLM response was not a list of dictionaries."
            )
            state.questions = []
    except json.JSONDecodeError:
        print("Error: Failed to parse LLM response as JSON.")
        state.questions = []
    except Exception as exc:
        print(
            f"An unexpected error occurred during JSON parsing: {exc}"
        )
        state.questions = []

    print("--- QUESTIONS GENERATED ---")
    print(state.questions)
    print("------")

    return state


def validate_questions(
    state: QPGenerationState,
) -> QPGenerationState:
    """Validate the generated questions.

    Rule-based checks:
        - At least one question must exist.
        - The count must match ``num_questions``.
        - Problems must be distinct.

    Parameters
    ----------
    state : QPGenerationState
        State populated by ``generate_questions``.

    Returns
    -------
    QPGenerationState
        State with ``question_validation`` and ``error_message`` set.
    """
    print(
        f"--- VALIDATING GENERATED QUESTIONS ---{state.topic}"
    )

    questions = state.questions

    # -- Must have at least one --
    if not questions:
        state.question_validation = "INVALID"
        state.error_message = "question generation failed."
        return state

    # -- Count must match --
    if len(questions) != state.num_questions:
        state.question_validation = "INVALID"
        state.error_message = (
            "Mismatch in number of questions generated."
        )
        print("--- GENERATED INVALID QUESTIONS ---")
        return state

    # -- Distinctness check --
    problems = [q.get("problem", "") for q in questions]
    if len(set(problems)) != len(problems):
        state.question_validation = "INVALID"
        state.error_message = "Duplicate questions detected."
        print("--- GENERATED INVALID QUESTIONS ---")
        return state

    print("--- GENERATED VALID QUESTIONS ---")
    state.question_validation = "VALID"
    state.error_message = ""
    return state
