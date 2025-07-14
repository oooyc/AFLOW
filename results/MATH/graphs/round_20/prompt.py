SUMMARIZE_PROBLEM_PROMPT = """
Provide a concise summary of the given mathematical problem, highlighting the key concepts and formulas involved. Your summary should:

1. Restate the problem briefly.
2. Identify the main mathematical topics or areas relevant to the problem.
3. List any important formulas or theorems that may be useful in solving the problem.
4. Mention any specific constraints or conditions given in the problem.

Keep your summary clear and concise, focusing on the essential elements needed to understand and approach the problem.
"""

REFINE_ANSWER_PROMPT = """
Given the mathematical problem, the problem summary, and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Briefly mention the key concepts identified in the summary.
3. Explain the approach and any formulas or concepts used.
4. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
5. Interpret the code output and incorporate it into your explanation.
6. Provide a final answer, enclosed in \boxed{} LaTeX notation.
7. Ensure all mathematical notation is in LaTeX format.

Your response should be comprehensive, mathematically rigorous, and easy to follow.
"""

GENERATE_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step, taking into account the provided problem summary. Follow these guidelines:

1. State the problem clearly.
2. Reference the key concepts and formulas mentioned in the summary.
3. Outline the approach and any relevant formulas or concepts.
4. Provide detailed calculations, using LaTeX notation for mathematical expressions.
5. Explain each step of your reasoning.
6. Present the final answer enclosed in \boxed{} LaTeX notation.
7. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
"""

DETAILED_SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given mathematical problem, incorporating the information from the problem summary. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and formulas involved, referencing the summary.
3. A detailed, logical progression of steps to solve the problem.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the answer in the context of the original problem.

Aim for clarity, accuracy, and thoroughness in your solution.
"""