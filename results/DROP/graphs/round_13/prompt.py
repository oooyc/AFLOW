REFINE_ANSWER_PROMPT = """
Given the question, step-by-step solution, and initial answer, please refine and format the final answer. Follow these guidelines:

1. Ensure the answer is concise and directly addresses the question.
2. If the answer is a number, provide only the number without additional text.
3. If multiple correct answers are possible, separate them with a vertical bar (|).
4. Remove any explanations or reasoning from the final answer.
5. Double-check the numerical calculations and ensure they are correct.
6. If the question asks for a specific unit (e.g., years, yards), make sure the answer includes only the number without the unit.

Your task is to extract or refine the most accurate answer based on the provided information.

"""