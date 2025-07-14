REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

VERIFY_ANSWER_PROMPT = """
Given the original question and the refined answer, please verify and correct the answer if necessary. Follow these guidelines:

1. Check if the answer directly addresses the question.
2. Ensure the answer format is correct (number only for numerical answers, multiple answers separated by |).
3. Verify that the answer is consistent with the information provided in the question.
4. If the answer is incorrect or needs adjustment, provide the corrected answer following the same formatting rules.

Your task is to ensure the final answer is accurate, concise, and properly formatted.
"""