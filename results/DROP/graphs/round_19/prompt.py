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
Given the question, step-by-step solution, and refined answer, please verify and correct the numerical answer if necessary. Follow these guidelines:

1. Check if the answer is consistent with the question and step-by-step solution.
2. Ensure the answer is a number or numbers separated by a vertical bar (|) if multiple answers are possible.
3. If the answer is incorrect or inconsistent, recalculate based on the step-by-step solution.
4. Remove any non-numeric characters or explanations, keeping only the numerical value(s).
5. If the question asks for a difference or comparison between values, double-check the calculation.
6. If no numerical answer is required, return the refined answer as is.

Your task is to provide the most accurate numerical answer based on the given information and calculations.
"""