REFINE_AND_FORMAT_ANSWER_PROMPT = """
Given the question and the best answer, refine and format the answer to be concise, accurate, and directly addressing the question. Ensure the answer is clear and brief, without additional explanation or reasoning. Focus on extracting the most relevant information.

If the answer is a name, profession, or short phrase, provide only that information. Do not include any prefixes like "The answer is" or "The profession is". Just provide the refined answer itself.
"""

EXTRACT_CORE_ANSWER_PROMPT = """
Given the question and the refined answer, extract the core information that directly answers the question. Provide only the essential details without any additional context or explanation. The output should be as concise as possible, ideally limited to a few words or a short phrase.

For names, provide only the name.
For professions, state only the profession.
For short phrases, give only that phrase.
For years or dates, provide only the year or date.

Ensure the extracted answer is the most relevant and accurate response to the question asked.
"""