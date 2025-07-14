REVIEW_PROMPT = """
Review the following problem and the generated solution. If you find any issues or potential improvements, provide an improved solution. If the solution is already optimal, state that no improvements are needed.

Problem: {input}

Your response should be in the following format:
Review: [Your review comments]
Improved Solution: [The improved solution if any, or "No improvements needed" if the original solution is optimal]

Ensure that your review is thorough and that any improved solution addresses the problem correctly and efficiently.
"""