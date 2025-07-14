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
Given the question and the proposed answer, please verify and format the answer according to the following rules:

1. If the answer is a single number, ensure it's provided without any units or additional text (e.g., "5" instead of "5%").
2. If the answer consists of multiple numbers, separate them with a vertical bar (|) without spaces (e.g., "3|20|32").
3. Remove any explanatory text or reasoning from the answer.
4. Ensure the answer directly addresses the question without any extraneous information.
5. If the original answer doesn't meet these criteria, modify it to comply.

Provide only the verified and formatted answer without any additional explanation.
"""