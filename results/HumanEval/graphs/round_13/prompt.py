REVIEW_SOLUTION_PROMPT = """
Carefully review the given problem and the generated solution. Your task is to analyze the code, identify any potential issues, and suggest improvements. Consider the following aspects:

1. Correctness: Does the solution correctly address all aspects of the problem?
2. Efficiency: Is the solution optimized for performance?
3. Readability: Is the code clear and easy to understand?
4. Edge cases: Does the solution handle all possible input scenarios?
5. Best practices: Does the code follow Python best practices?

If you find any issues or areas for improvement, modify the code accordingly. If the solution looks good, you can return it as is.

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