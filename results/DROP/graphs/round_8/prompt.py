REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.

"""

MATH_CORRECTION_PROMPT = """
Given the question and the refined answer, perform the following tasks:

1. Identify if the answer involves a mathematical calculation or a numerical value.
2. If it does, verify the calculation or the numerical value for accuracy.
3. If you find any discrepancies or errors, correct them and provide the accurate result.
4. If the answer is already correct or doesn't involve calculations, return the original answer.
5. Ensure the final answer follows the format guidelines:
   - Provide only the number without additional text for numerical answers.
   - Use a vertical bar (|) to separate multiple correct answers.
   - Do not include any explanations or reasoning in the final answer.

Your task is to ensure mathematical accuracy while maintaining the concise answer format.

"""