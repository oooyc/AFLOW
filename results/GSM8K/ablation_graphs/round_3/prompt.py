SOLVE_PROMPT = """
Solve the given math problem step by step. Show your work clearly and provide a final numerical answer.

"""

REVIEW_PROMPT = """
Review the initial solution to the math problem. Verify the calculations, ensure the reasoning is correct, and format the final answer as a single numerical value without any units or explanations. If there are any errors, correct them and provide the accurate result.

"""

VERIFY_PROMPT = """
Extract the final numerical answer from the given solution. If the answer is already a single number, return it as is. If it's not, find the last numerical value in the text and return only that number without any additional text or formatting. Ensure the output is a single numerical value without any units, explanations, or additional characters.

"""