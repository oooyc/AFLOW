REFINE_PROMPT = """
Given the problem, step-by-step thought process, and initial answer, please refine and format the final answer. Ensure that:
1. The answer is concise and directly addresses the question.
2. Only include numerical values or short phrases as required by the question.
3. Do not include explanations or additional context unless specifically asked.
4. If the answer involves multiple items, separate them with a vertical bar (|).

Problem: {input}

Provide the refined answer:
"""