FORMAT_ANSWER_PROMPT = """
Given the question and the verified answer, format the final answer to be concise, accurate, and directly addressing the question. Ensure the answer is clear and brief, without unnecessary explanation. Follow these guidelines:

1. For questions about people, provide only the name or profession.
2. For questions about dates or numbers, provide only the specific date or number.
3. For questions asking about differences or comparisons, provide a brief, one-sentence explanation highlighting the key difference.
4. For questions about facts or events, provide a concise statement of the fact or event.

Do not include any prefixes like "The answer is" or "The difference is". Provide only the essential information to answer the question.

Question: {question}
Verified answer: {verified_answer}

Format your response accordingly:
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