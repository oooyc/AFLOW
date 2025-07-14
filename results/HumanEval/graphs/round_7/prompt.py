CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Ensure the function signature matches the problem description. Include docstrings and comments to explain the logic. The solution should be efficient and handle edge cases.
"""

SELF_REVIEW_PROMPT = """
Review the generated solution and provide feedback on its correctness, efficiency, and adherence to best practices. Consider the following:
1. Does the solution correctly implement the problem requirements?
2. Are there any potential edge cases that are not handled?
3. Can the code be optimized for better performance?
4. Does the code follow Python best practices and style guidelines?
5. Are there any logical errors or potential bugs?
Provide specific suggestions for improvements if necessary.
"""

ANALYZE_AND_IMPROVE_PROMPT = """
Analyze the failed solution and provide suggestions for improvement. Consider the following:
1. Identify logical errors or misunderstandings of the problem.
2. Check for edge cases that might not be handled correctly.
3. Suggest optimizations or alternative approaches if applicable.
4. Provide specific code snippets or pseudocode to address the issues found.
"""