"""
Entry point - runs all three workflows end-to-end.

Usage::

    # default (reads .env)
    uv run python -m src.uc1_assesment_main

    # override via env
    TOPIC="Dictionaries" NUM_QUESTIONS=3 \\
        uv run python -m src.uc1_assesment_main.main
"""

from __future__ import annotations

from pprint import pprint

from dotenv import load_dotenv

from src.config.settings import settings
from src.graphs.evaluation_graph import (
    build_evaluation_graph,
)
from src.graphs.question_generation_graph import (
    build_question_generation_graph,
)
from src.graphs.test_generation_graph import (
    build_test_generation_graph,
)
from src.states.evaluation import EvaluationState
from src.states.question_generation import QPGenerationState
from src.states.test_generation import TestGenerationState

# -- Load .env so settings picks up local overrides --
load_dotenv()


# -- Example learner response --
_USER_RESPONSE = (
    "def generate_even_squares(input_list):\n"
    "    if not all(isinstance(x, int) for x in input_list):\n"
    "        raise ValueError(\n"
    "            'All elements of the input list must be integers.'\n"
    "        )\n"
    "    squared_evens = [x ** 2 for x in input_list if x % 2 == 0]\n"
    "    return squared_evens\n"
)


def main() -> None:
    """Run the full assessment pipeline end-to-end."""
    chosen_topic = settings.topic
    num_questions = settings.num_questions
    num_test_cases = settings.num_test_cases

    # -- 1. Question Generation (Sequential) --
    print("=" * 60)
    print("  MODULE 1: QUESTION GENERATION (Sequential)")
    print("=" * 60)

    qg_graph = build_question_generation_graph()
    qg_result = qg_graph.invoke(
        QPGenerationState(
            topic=chosen_topic,
            num_questions=num_questions,
        )
    )

    question = qg_result["questions"][0]
    print(f"\nSelected question: {question['question_no']}\n")

    # -- 2. Test Case Generation (Parallelisation) --
    print("=" * 60)
    print("  MODULE 2: TEST CASE GENERATION (Parallel)")
    print("=" * 60)

    tg_graph = build_test_generation_graph()
    tg_result = tg_graph.invoke(
        TestGenerationState(
            question=question,
            no_of_test_cases=num_test_cases,
        )
    )

    test_case_list = tg_result["test_cases"]
    print(
        f"\nGenerated {len(test_case_list)} test cases.\n"
    )

    # -- 3. Evaluation & Feedback (Conditional) --
    print("=" * 60)
    print("  MODULE 3: EVALUATION & FEEDBACK (Conditional)")
    print("=" * 60)

    eval_graph = build_evaluation_graph()
    eval_result = eval_graph.invoke(
        EvaluationState(
            question=question,
            test_cases=test_case_list,
            user_response=_USER_RESPONSE,
            result="pass",
            feedback="",
            passed_test_cases=[],
            failed_test_cases=[],
        )
    )

    # -- Output --
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Result: {eval_result['result']}")
    pprint(f"Feedback: {eval_result['feedback']}")
    print(
        f"Provider: {settings.llm_provider}  |  "
        f"Model: {settings.llm_model}"
    )


if __name__ == "__main__":
    main()
