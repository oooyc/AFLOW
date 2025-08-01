FORMAT_ANSWER_PROMPT = """
Given the question and the final verified answer, format the final answer to be extremely concise and directly addressing the question. Provide only the essential information without any explanation or additional context. For names, professions, short phrases, or numerical answers, give only that specific information.

Examples:
- For a person's name: "John Doe"
- For a profession: "Doctor"
- For a short phrase: "Once in a blue moon"
- For a numerical answer: "6" or "six"

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

STYLE_CHECK_PROMPT = """
Review the fact-checked answer, focusing specifically on architectural and historical style terms. Verify the accuracy of any mentioned styles, ensuring they are correctly identified and placed in the proper historical context. Pay special attention to:

1. Architectural styles (e.g., Gothic, Baroque, Neoclassical)
2. Historical periods and their associated styles
3. Relationships between different styles (e.g., which style influenced or preceded another)

If you find any errors or misidentifications in the architectural or historical style terms, provide a corrected version. If the answer is accurate, simply restate it. Be concise and direct in your response, providing only the verified or corrected answer without explanations.
"""

ADDITIONAL_VERIFICATION_PROMPT = """
Perform an additional verification on the previously style-checked answer. Focus on:

1. Consistency with the question
2. Accuracy of specific details (names, dates, places, styles)
3. Completeness of the answer

If any issues are found, provide a corrected version. If the answer is accurate and complete, restate it. Be concise and direct in your response, providing only the verified or corrected answer without explanations.
"""

NUMERICAL_VERIFICATION_PROMPT = """
Review the verified answer, focusing specifically on numerical information. If the answer contains a number, verify its accuracy and format it appropriately. Pay attention to:

1. Consistency between numerical and word representations (e.g., "6" and "six")
2. Appropriate use of numerals or words based on context
3. Accuracy of the numerical information in relation to the question

If the answer contains a number, provide it in both numeral and word form (e.g., "6 (six)"). If no numerical information is present, simply restate the answer. Be concise and direct in your response, providing only the verified or corrected answer without explanations.
"""