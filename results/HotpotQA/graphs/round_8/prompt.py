FORMAT_ANSWER_PROMPT = """
Given the question and the best answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is a clear, brief statement without additional explanation or reasoning. If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.
"""

VERIFY_ANSWER_PROMPT = """
Given the question and the formatted answer, verify if the answer is appropriate and make adjustments if necessary. Follow these guidelines:

1. If the answer is a name, profession, or short phrase, ensure it's presented without any additional context or sentence structure.
2. If the answer is "The context does not provide...", replace it with a more concise statement like "Information not available" or "Unknown".
3. If the answer is too long or includes unnecessary explanation, trim it down to the essential information.
4. Ensure the answer directly addresses the question asked.

Provide the final, verified answer without any additional explanation or reasoning.
"""