SOLVE_PROMPT = """
Solve the given math problem step by step. Show your work clearly and provide a final numerical answer.
"""

REVIEW_PROMPT = """
Review the initial solution to the math problem. Verify the calculations, ensure the reasoning is correct, and format the final answer as a single numerical value without any units or explanations. If there are any errors, correct them and provide the accurate result.
"""

SECOND_REVIEW_PROMPT = """
Perform a second review of the solution, focusing on common calculation errors and edge cases. Pay special attention to:
1. Unit conversions (e.g., minutes to hours, days to weeks)
2. Proper interpretation of the problem statement
3. Correct application of mathematical operations
4. Handling of fractions and decimals
5. Consistency in units throughout the solution

If you find any errors or inconsistencies, correct them and provide the accurate result. Present the final answer as a single numerical value without any units or explanations.
"""

VERIFY_PROMPT = """
Extract the final numerical answer from the given solution. If the answer is already a single number, return it as is. If it's not, find the last numerical value in the text and return only that number without any additional text or formatting. Ensure the output is a single numerical value without any units, explanations, or additional characters.
"""

FINAL_CHECK_PROMPT = """
Perform a final check on the verified solution, considering the original problem statement. Focus on:

1. Off-by-one errors: Ensure the solution accounts for all relevant time periods, items, or entities mentioned in the problem.
2. Misinterpretation of the problem: Verify that the solution addresses the specific question asked and doesn't solve for a related but incorrect value.
3. Unit consistency: Confirm that the final answer is in the appropriate units as implied by the problem statement.
4. Reasonableness of the result: Check if the answer seems logical given the context of the problem.

If you find any errors or inconsistencies, correct them and provide the accurate result. Present the final answer as a single numerical value without any units or explanations. If no errors are found, return the original verified solution.
"""

CALCULATION_VERIFICATION_PROMPT = """
Verify the final solution by performing the following steps:
1. Re-read the original problem statement carefully.
2. Identify all the key numerical values and relationships mentioned in the problem.
3. Perform the calculations independently, step by step, using the information from the problem.
4. Compare your calculated result with the provided final solution.
5. If there's a discrepancy, carefully review both your calculations and the provided solution to identify the source of the difference.
6. If your calculation confirms the provided solution, return the solution as is.
7. If your calculation differs from the provided solution and you're confident in your result, return your calculated answer.
8. In either case, ensure the final answer is presented as a single numerical value without any units or explanations.

Your response should be only the final, verified numerical answer.
"""