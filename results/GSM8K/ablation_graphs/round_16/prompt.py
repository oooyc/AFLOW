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

CONSISTENCY_CHECK_PROMPT = """
Compare the initial solution and the reviewed solution for the given math problem. Assess their consistency and provide a brief analysis. If there are discrepancies, explain which solution seems more accurate and why. If they are consistent, confirm this.

Provide your analysis in the following format:

Consistency: [Consistent/Inconsistent]
Analysis: [Your brief analysis here]
Recommended Answer: [The numerical value you believe is correct]

Problem:
"""

VERIFY_AND_FORMAT_PROMPT = """
Verify the reviewed solution and consistency check for the given math problem. Consider the following:
1. The answer should be a single numerical value.
2. The answer should be within a reasonable range for the given problem.
3. The answer should be formatted correctly as "Final Answer: [numerical value]".
4. Take into account the consistency check between the initial and reviewed solutions.

If any of these conditions are not met, correct the answer accordingly. If there are discrepancies, use your judgment to determine the most likely correct answer based on the problem and the provided solutions.

Format your response as follows:

Final Answer: [Insert the numerical value here]

Problem:
"""