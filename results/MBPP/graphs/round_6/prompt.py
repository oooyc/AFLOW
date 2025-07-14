CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

CODE_REVIEW_PROMPT = """
Review the following Python code for the given problem. Identify any potential issues, improve code quality, and optimize if necessary. Ensure the function name and signature remain unchanged. Provide the reviewed and improved code.

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