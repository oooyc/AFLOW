SOLVE_AND_EXTRACT_PROMPT = """
Solve the given math problem step by step. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

ALTERNATIVE_SOLVE_PROMPT = """
Solve the given math problem using a different approach or method than you might typically use. Show your work step by step. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

COMPARE_AND_SELECT_PROMPT = """
Compare the two given solutions for the math problem. Analyze their approaches, calculations, and final answers. Select the solution that you believe is more accurate or reliable. If both solutions arrive at the same answer, choose the one with the clearer or more efficient approach.

Provide your selection and a brief explanation for your choice. Then, present the selected solution's final answer in the following format:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

REVIEW_AND_CORRECT_PROMPT = """
Review the selected solution to the given math problem. Check for any errors in calculation or reasoning. If you find any mistakes, correct them and provide the correct answer. If the selected solution is correct, verify that the answer is properly formatted.

Ensure the final answer is a numerical value only, formatted as follows:

Final Answer: [Insert the numerical value here]

Do not include any units or additional text after "Final Answer:".

Problem:
"""