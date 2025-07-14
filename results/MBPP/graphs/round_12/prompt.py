CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

REVIEW_PROMPT = """
Review and improve the following Python function. Ensure it correctly solves the given problem, uses appropriate data structures and algorithms, and follows best practices for code readability and efficiency. If improvements are necessary, provide the updated function.

Problem:
{input}

Provide the reviewed and improved function below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""