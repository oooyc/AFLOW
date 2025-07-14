FORMAT_ANSWER_PROMPT = """
Given the question and the best answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is a clear, brief statement without additional explanation or reasoning. If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.
"""

async def __call__(self, problem: str):
    """
    Implementation of the workflow
    """
    solutions = []
    for _ in range(3):
        initial_response = await self.answer_generate(input=problem)
        thought_process = initial_response['thought']
        initial_answer = initial_response['answer']
        solutions.append(initial_answer)

    ensemble_result = await self.sc_ensemble(solutions=solutions)
    best_answer = ensemble_result['response']

    refined_solution = await self.custom(
        input=f"Question: {problem}\nBest answer: {best_answer}",
        instruction=prompt_custom.FORMAT_ANSWER_PROMPT
    )

    return refined_solution['response'], self.llm.cost_manager.total_cost