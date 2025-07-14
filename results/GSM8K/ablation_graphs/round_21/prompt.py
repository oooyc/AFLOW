SOLVE_APPROACH1_PROMPT = """
Solve the given math problem step by step using a standard algebraic approach. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

SOLVE_APPROACH2_PROMPT = """
Solve the given math problem step by step using a visual or diagrammatic approach, if applicable. If not applicable, use an alternative method different from the standard algebraic approach. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

SOLVE_APPROACH3_PROMPT = """
Solve the given math problem step by step using estimation or approximation techniques, then refine the answer for accuracy. After solving, extract the final numerical answer and format it as follows:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""

COMPARE_AND_SELECT_PROMPT = """
Compare the three solutions provided for the given math problem. Analyze each solution for correctness, completeness, and consistency with the problem statement. Select the most accurate and reliable solution, or if all solutions agree, confirm their consistency.

If the solutions differ, explain the differences and justify your selection of the most accurate answer. If all solutions agree, state this consistency.

Provide the final answer in the following format:

Final Answer: [Insert the numerical value here]

Ensure that only the numerical value is provided after "Final Answer:", without any units or additional text.

Problem:
"""