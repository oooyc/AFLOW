CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Ensure the function signature matches the problem description. Include docstrings and comments to explain the logic. The solution should be efficient and handle edge cases. Pay special attention to the order of operations and ensure that the function evaluates the expression correctly.
"""

REVIEW_PROMPT = """
Review the generated solution and suggest improvements. Consider the following:
1. Does the function correctly implement the order of operations?
2. Are all required operators (+, -, *, //, **) handled properly?
3. Is there proper error handling for cases like division by zero?
4. Does the function handle the case where the operator list is empty?
5. Is the code efficient and readable?
Provide specific suggestions for improvement.
"""

ANALYZE_AND_IMPROVE_PROMPT = """
Analyze the failed solution and provide suggestions for improvement. Consider the following:
1. Identify logical errors or misunderstandings of the problem.
2. Check for edge cases that might not be handled correctly.
3. Ensure the order of operations is correctly implemented.
4. Suggest optimizations or alternative approaches if applicable.
5. Provide specific code snippets or pseudocode to address the issues found.
"""