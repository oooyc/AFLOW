REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

CHECK_NUMERICAL_ANSWER_PROMPT = """
Given the question and the refined answer, please perform a numerical check and correction if necessary. Follow these steps:

1. Identify if the answer is numerical.
2. If it is numerical, carefully review the question and the given answer to ensure its accuracy.
3. If you find any discrepancies or errors in the numerical answer, provide the correct numerical answer.
4. If the answer is correct or not numerical, return the original answer without changes.
5. Ensure the final answer follows the format guidelines: numbers only, no additional text, and use vertical bars (|) for multiple possible answers.

Your task is to verify and correct numerical answers while maintaining the correct answer format.
"""