CODE_GENERATE_PROMPT = """
Generate a Python function that solves the given problem. Make sure to handle edge cases and use efficient algorithms where possible.

Problem:
{problem}

Entry point:
{entry_point}

Generate the solution:
"""

IMPROVED_CODE_GENERATE_PROMPT = """
The previous solution failed some test cases. Please generate an improved Python function that solves the given problem. Pay extra attention to edge cases and efficiency.

Problem:
{problem}

Entry point:
{entry_point}

Previous solution:
{previous_solution}

Generate an improved solution:
"""