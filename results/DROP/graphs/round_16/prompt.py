REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.

Your task is to extract or refine the most accurate answer based on the provided information.
"""

EXTRACT_VALIDATE_PROMPT = """
Given the question and the refined answer, please extract and validate the numerical answer. Follow these guidelines:

1. Extract only the numerical value from the refined answer.
2. Ensure the extracted number is relevant to the question asked.
3. If the answer is a decimal, maintain the original decimal places.
4. If the answer is a whole number, do not add decimal places.
5. If no valid numerical answer can be extracted, respond with "Unable to extract a valid numerical answer."

Provide only the extracted and validated numerical answer without any additional text or explanations.
"""