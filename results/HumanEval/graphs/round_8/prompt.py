CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Ensure the function signature matches the problem description. Include docstrings and comments to explain the logic. The solution should be efficient and handle edge cases.
"""

COMPREHENSIVE_ANALYSIS_PROMPT = """
Perform a comprehensive analysis of the problem and the failed solution. Consider the following aspects:

1. Problem understanding: Identify any potential misinterpretations of the problem statement.
2. Logical errors: Pinpoint any flaws in the solution's logic or algorithm.
3. Edge cases: Determine if all possible input scenarios are properly handled.
4. Efficiency: Evaluate the time and space complexity of the solution.
5. Coding style: Assess the readability and adherence to Python best practices.
6. Test case failures: Analyze why specific test cases are failing.
7. Alternative approaches: Suggest different algorithms or data structures that might be more suitable.

Provide a detailed breakdown of these aspects and suggest specific improvements for each identified issue.
"""