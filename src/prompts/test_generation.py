"""System / human prompt templates for test-case generation."""

from __future__ import annotations

# -- Standard / valid cases --
STANDARD_TC_SYSTEM = """\
Create {num_test_cases} test cases for the given programming QUESTION, \
focusing on typical valid scenarios.

Provide the output as a JSON object with a key 'test_cases', which \
includes a list of test case details:

Each test case should have:
- "input": A string version of valid input.
- "expected_output": A string version of expected output.
- "testcase_type": Always "Standard Valid".
- "difficulty": Either "easy" for simple or "hard" for complex.

QUESTION Format is JSON objects with following schema:
- question_no: Question number (Q1, Q2, etc.)
- problem: Problem statement starting with "Write a Python function to..."
- sample_input: Input arguments
- sample_output: Expected output

Specifications:
- Provide exactly {num_test_cases} test case JSON objects in 'test_cases'.
- No explanations, only the raw JSON data.
"""

# -- Edge cases --
EDGE_TC_SYSTEM = """\
Generate {num_test_cases} edge case test cases for the given \
programming QUESTION.

Return JSON with a 'test_cases' key containing test case objects with:
- input: stringified valid edge/boundary input
- expected_output: correct output (never errors)
- testcase_type: "Edge Case"
- difficulty: "easy" or "hard"

QUESTION Format is JSON objects with following schema:
- question_no: Question number (Q1, Q2, etc.)
- problem: Problem statement starting with "Write a Python function to..."
- sample_input: Input arguments
- sample_output: Expected output

STRICT RULES:
1. Only create edge cases if explicitly specified in the problem.
2. Inputs must be valid but test boundaries/extremes.
3. Never include error-triggering inputs.
4. Never make assumptions.
5. If no edge cases specified, return empty list.
6. Max {num_test_cases} test cases.
"""

# -- Error / invalid cases --
ERROR_TC_SYSTEM = """\
Generate {num_test_cases} test cases for the provided programming \
QUESTION, targeting invalid inputs that induce errors.

Output a JSON object with a single key 'test_cases', containing a \
list of test case objects with this schema:
- "input": stringified invalid input
- "expected_output": stringified error indication
- "testcase_type": "Error Case"
- "difficulty": "easy" or "hard"

QUESTION Format is JSON objects with following schema:
- question_no: Question number (Q1, Q2, etc.)
- problem: Problem statement starting with "Write a Python function to..."
- sample_input: Input arguments
- sample_output: Expected output

## Constraints:
- Inputs must be invalid, violating explicit problem constraints.
- Error conditions must be specified in the problem statement.
- Exclude empty strings/lists or valid edge cases as error cases.
- Expected output must reflect an error state.
- Return exactly {num_test_cases} test cases in 'test_cases'.
- Do not include code, explanations, or valid inputs.
"""

# -- Shared human template --
TC_HUMAN = "Question: {question}"
