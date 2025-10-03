BASE_PROMPT = """
Solve the problem step-by-step. Verify your final answer matches all constraints. 
Structure your response:
1. Analyze key elements and constraints
2. Show calculations/reasoning clearly
3. Confirm solution satisfies original problem
4. Box the final answer as \boxed{answer}

Problem: {input}
"""