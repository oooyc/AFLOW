CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

ANALYZE_IMPROVE_PROMPT = """
Analyze the following Python function and suggest improvements for better performance, readability, and correctness. Consider edge cases, potential errors, and optimization opportunities. Provide an improved version of the function.

Problem:
{input}

Provide the improved function below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""