FACT_CHECK_PROMPT = """
Given the question and the initial answer, please verify the accuracy of the answer. If you find any inaccuracies or have additional relevant information, provide a corrected or more accurate answer. If the initial answer is correct and complete, simply restate it.

Focus on:
1. Correcting any factual errors
2. Providing more specific information if the initial answer is too vague
3. Ensuring the answer directly addresses the question asked

Do not include any explanations or reasoning in your response. Provide only the verified or corrected answer.
"""

FORMAT_ANSWER_PROMPT = """
Given the question and the verified answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is a clear, brief statement without additional explanation or reasoning. If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.

Ensure that the final answer is in the correct form (singular/plural, capitalization) to directly match the question.
"""