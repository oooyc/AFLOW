FORMAT_ANSWER_PROMPT = """
Given the question and the verified answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is a clear, brief statement without additional explanation or reasoning. If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.
"""

FACT_CHECK_PROMPT = """
Given the question and the best answer, carefully analyze the information provided and perform a fact-check. If you find any inconsistencies or errors in the best answer, provide a corrected version. If the best answer appears accurate, simply restate it. Focus on verifying key facts, names, dates, and other critical information related to the question.

Question: {question}
Best answer: {best_answer}

Your task:
1. Analyze the question and the best answer.
2. Identify any potential errors or inconsistencies.
3. If errors are found, provide a corrected answer.
4. If no errors are found, restate the best answer.

Provide your response in a concise manner, focusing solely on the corrected or verified answer without additional explanations.
"""