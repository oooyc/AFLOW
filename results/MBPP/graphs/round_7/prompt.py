CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

REVIEW_PROMPT = """
Review and improve the generated solution for the given problem. Check for:
1. Correct implementation of the problem requirements
2. Proper error handling and edge cases
3. Code efficiency and optimization
4. Adherence to Python best practices and PEP 8 style guide

Problem:
{input}

Provide the reviewed and improved solution below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""