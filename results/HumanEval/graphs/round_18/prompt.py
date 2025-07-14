REVIEW_PROMPT = """
Review the following generated solution for the given problem. If the solution is correct and optimal, respond with 'No improvements needed'. If there are any issues or potential improvements, describe them in detail.

Problem: {problem}

Generated Solution:
{solution}

Your review:
"""