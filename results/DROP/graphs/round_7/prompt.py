FORMAT_ANSWER_PROMPT = """
Given the question and the best solution, please format the final answer following these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text or units.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information and format it accordingly.
"""

ERROR_CHECK_PROMPT = """
Given the question and the formatted answer, please perform an error check and correction if necessary. Follow these guidelines:

1. Verify that the answer directly addresses the question asked.
2. Ensure that the answer is in the correct format (e.g., only a number for numerical answers, no additional text).
3. Check for any obvious errors in calculation or reasoning.
4. If you detect an error, provide the corrected answer. If no error is found, return the original answer.

Your task is to catch and correct any potential errors in the formatted answer, ensuring the highest possible accuracy.
"""