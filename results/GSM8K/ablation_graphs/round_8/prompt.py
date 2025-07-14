SOLVE_AND_EXTRACT_PROMPT = """
Solve the given math problem step by step. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

REVIEW_AND_VOTE_PROMPT = """
Review the initial solutions to the given math problem. Compare the solutions and determine the most likely correct answer based on consistency and reasoning. If there are discrepancies, analyze the steps and choose the most convincing solution. Provide your final answer in the following format:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

VERIFY_AND_FORMAT_PROMPT = """
Verify the reviewed solution to the given math problem. Ensure that:
1. The answer is a single numerical value.
2. The answer is within a reasonable range for the given problem.
3. The answer is formatted correctly as "Final Answer: [numerical value]".

If any of these conditions are not met, correct the answer accordingly. If the answer meets all conditions, return it as is.

Problem:
"""