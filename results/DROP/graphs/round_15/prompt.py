REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.
5. Double-check that the refined answer is consistent with the step-by-step solution and initial answer.
6. If the answer involves selecting between options, clearly state the chosen option.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

VERIFY_ANSWER_PROMPT = """
Given the original question and the proposed answer, please verify and, if necessary, correct the answer. Follow these guidelines:

1. Ensure the answer directly addresses the question asked.
2. If the answer should be a number, make sure it's provided as a number without additional text.
3. If multiple correct answers are possible, they should be separated by a vertical bar (|).
4. The answer should not include any explanations or reasoning.
5. If the question asks for a percentage, ensure the answer is in the correct percentage format (e.g., 74.60 instead of 0.746).
6. If the answer involves selecting between options, make sure the chosen option is clearly stated.
7. If the answer is correct and properly formatted, return it as is. If not, provide the corrected version.

Your task is to ensure the final answer is accurate, properly formatted, and consistent with the question requirements.
"""