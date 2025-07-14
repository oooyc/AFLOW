FORMAT_ANSWER_PROMPT = """
Given the question and the best answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is a clear, brief statement without additional explanation or reasoning. If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.
"""