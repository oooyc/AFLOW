REFINE_ANSWER_PROMPT = """
Given the problem, step-by-step solution, and initial answer, please refine and format the answer to ensure it is concise, accurate, and directly addresses the question. Follow these guidelines:

1. Review the step-by-step solution for accuracy and completeness.
2. Ensure the final answer is clearly stated and matches the question asked.
3. If the answer is a numerical value, include only the number without additional explanation.
4. If the answer requires a brief explanation, keep it concise and to the point.
5. Remove any unnecessary information or verbosity.

Provide the refined answer in a clear and concise format.
"""