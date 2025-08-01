SOLVE_AND_EXTRACT_PROMPT = """
Solve the given math problem step by step. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

REVIEW_AND_CORRECT_PROMPT = """
Review the initial solution to the given math problem. Check for any errors in calculation or reasoning. If you find any mistakes, correct them and provide the correct answer. If the initial solution is correct, verify that the answer is properly formatted.

Ensure the final answer is a numerical value only, formatted as follows:

Final Answer: [Insert the numerical value here]

Do not include any units or additional text after "Final Answer:".

Problem:
"""