REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

FORMAT_NUMERIC_ANSWER_PROMPT = """
Given an answer, please format it according to these rules:

1. If the answer contains a single number:
   - Remove any additional text or symbols (like %, $, etc.)
   - Present the number without commas
   - If it's a whole number, remove any decimal places
   - If it's a decimal, keep up to two decimal places

2. If the answer contains multiple numbers separated by '|':
   - Apply the above rules to each number
   - Keep the '|' separator between numbers

3. If the answer doesn't contain any numbers, return it as is.

Examples:
"74.60%" should become "74.60"
"3 | 20 | 32" should remain "3 | 20 | 32"
"$1,234,567.89" should become "1234567.89"
"42 years old" should become "42"

Your task is to format the given answer according to these rules.
"""