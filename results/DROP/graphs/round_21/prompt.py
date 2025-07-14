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

VERIFY_NUMERICAL_ANSWER_PROMPT = """
Given the question and the refined answer, please verify and correct the numerical answer if necessary. Follow these guidelines:

1. Ensure the answer is a number or a set of numbers separated by a vertical bar (|).
2. Remove any units, percentage signs, or additional text from the answer.
3. If the answer should be a decimal, ensure it is properly formatted (e.g., 1.2 instead of 1.2%).
4. If the answer involves a calculation, double-check the math to ensure accuracy.
5. If the answer is correct and properly formatted, return it as is.
6. If the answer needs correction, provide the corrected numerical answer.

Your task is to verify and, if necessary, correct the numerical answer to ensure accuracy and proper formatting.
"""