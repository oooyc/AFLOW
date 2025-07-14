REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

REVIEW_ANSWER_PROMPT = """
Review the refined answer for the given question. Ensure it meets the following criteria:

1. The answer is directly related to the question asked.
2. If the answer is a number, it should be provided without any additional text or symbols (e.g., no %, $, etc.).
3. The answer is in its simplest form (e.g., fractions are reduced, unnecessary decimal places are removed).
4. If multiple answers are possible, they are separated by a vertical bar (|) without spaces.

If the answer meets all criteria, return it as is. If not, modify the answer to meet these criteria and return the corrected version.
"""