ANALYZE_PROBLEM_PROMPT = """
Analyze the given problem and generate a set of test cases that cover various scenarios. Consider edge cases, typical inputs, and potential pitfalls. Your analysis should include:
1. A brief description of the problem
2. Key considerations for implementation
3. A list of at least 3 test cases with expected outputs

Problem:
{input}

Provide your analysis below:
"""

CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Problem analysis:
{analysis['response']}

Generate the complete function below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""