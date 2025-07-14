REVIEW_SOLUTION_PROMPT = """
Review the given problem and the generated solution. Analyze the code for correctness, efficiency, and adherence to best practices. Improve the solution if necessary.

Your task:
1. Understand the problem statement.
2. Review the generated solution.
3. Check for logical errors, edge cases, and potential improvements.
4. Rewrite the solution if improvements are needed, or keep it as is if it's optimal.
5. Ensure the final solution is complete, correct, and efficient.

Provide only the reviewed and potentially improved Python function as your response, without any additional explanations or comments.
"""

IMPROVE_SOLUTION_PROMPT = """
Analyze the given problem and the failed solution. Identify the errors and improve the code to solve the problem correctly. Make sure to handle edge cases and follow best practices in Python programming.

Your task:
1. Understand the problem statement.
2. Review the failed solution and the error message.
3. Identify the cause of the failure.
4. Rewrite the solution, addressing the identified issues.
5. Ensure the new solution is complete, correct, and efficient.

Provide only the improved Python function as your response, without any additional explanations or comments.
"""