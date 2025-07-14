CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Ensure the function is correct, efficient, and follows best practices. Include comments to explain the logic.

Problem:
{problem}

Function signature:
{entry_point}

Your task:
1. Implement the function to solve the problem.
2. Add comments to explain the logic and any important steps.
3. Ensure the function handles edge cases and potential errors.
4. Optimize the solution for efficiency where possible.

Please provide only the function implementation without any additional text or explanations outside the code.
"""