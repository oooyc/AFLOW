FORMAT_ANSWER_PROMPT = """
Given the question and the double-verified answer, format the final answer to be extremely concise and directly addressing the question. Provide only the essential information without any explanation or additional context. For names, professions, or short phrases, give only that specific information.

Examples:
- For a person's name: "John Doe"
- For a profession: "Doctor"
- For a short phrase: "Once in a blue moon"

Do not include any prefixes or explanatory text. Provide only the answer itself.
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

ADDITIONAL_VERIFICATION_PROMPT = """
Perform an additional verification on the previously fact-checked answer. Focus on:

1. Consistency with the question
2. Accuracy of specific details (names, dates, places)
3. Completeness of the answer

If any issues are found, provide a corrected version. If the answer is accurate and complete, restate it. Be concise and direct in your response, providing only the verified or corrected answer without explanations.
"""