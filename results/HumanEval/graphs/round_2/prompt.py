CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Ensure the function signature matches the problem description. Include docstrings and comments to explain the logic. The solution should be efficient and handle edge cases.
"""

ANALYZE_AND_IMPROVE_PROMPT = """
Analyze the failed solution and provide suggestions for improvement. Consider the following:
1. Identify logical errors or misunderstandings of the problem.
2. Check for edge cases that might not be handled correctly.
3. Suggest optimizations or alternative approaches if applicable.
4. Provide specific code snippets or pseudocode to address the issues found.
"""