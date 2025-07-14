FORMAT_ANSWER_PROMPT = """
Given the question and the double-verified answer, format the final answer to be extremely concise and directly addressing the question. Provide only the essential information without any explanation or additional context. For names, professions, or short phrases, give only that specific information.

Examples:
- For a person's name: "John Doe"
- For a profession: "Doctor"
- For a short phrase: "Once in a blue moon"

Do not include any prefixes or explanatory text. Provide only the answer itself.
"""

NAME_EXTRACTION_PROMPT = """
Given the question and the best answer, focus on extracting and verifying any names mentioned. Pay special attention to:

1. Full names (first name, middle name, last name)
2. Nicknames or aliases
3. Correct spelling of names

If the answer contains a name, ensure it is accurate and complete. If a name is incorrect or incomplete, provide the correct full name. If no name is present or needed, simply restate the answer.

Provide your response in a concise manner, focusing solely on the corrected or verified answer without additional explanations.
"""

ENTITY_DISAMBIGUATION_PROMPT = """
Given the question and the name-extracted answer, perform entity disambiguation to ensure the correct identification of individuals, companies, or other entities mentioned. Consider:

1. Different entities with similar names
2. Context-specific identities (e.g., a person's role in a specific project)
3. Common misconceptions or frequently confused entities

If there's any ambiguity or potential misidentification, clarify the entity's identity based on the context of the question. If no disambiguation is needed, restate the answer.

Provide your response in a concise manner, focusing solely on the disambiguated or verified answer without additional explanations.
"""

IDENTITY_CHECK_PROMPT = """
Given the question and the disambiguated answer, perform a thorough identity check, especially for individuals who may have changed their names, use stage names, or have multiple identities. Consider:

1. Birth names vs. stage names or pseudonyms
2. Name changes due to personal reasons (e.g., marriage, gender transition)
3. Professional names vs. legal names

If the answer involves a person's identity and there's a discrepancy or additional information about their name, provide the most accurate and relevant name based on the context of the question. If no changes are needed, restate the answer.

Provide your response in a concise manner, focusing solely on the verified or corrected identity without additional explanations.
"""

FACT_CHECK_PROMPT = """
Given the question and the identity-checked answer, carefully analyze the information provided and perform a fact-check. If you find any inconsistencies or errors in the answer, provide a corrected version. If the answer appears accurate, simply restate it. Focus on verifying key facts, names, dates, and other critical information related to the question.

Your task:
1. Analyze the question and the answer.
2. Identify any potential errors or inconsistencies.
3. If errors are found, provide a corrected answer.
4. If no errors are found, restate the answer.

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
4. Ensure the answer is a single word, number, or short phrase when appropriate

If any issues are found, provide a corrected version. If the answer is accurate and complete, restate it. Be concise and direct in your response, providing only the verified or corrected answer without explanations.
"""

MISSING_INFO_CHECK_PROMPT = """
Analyze the given answer in the context of the question. Determine if any specific information requested in the question is missing from the answer. If information is missing, clearly state what is missing. If all requested information is present, simply restate the answer.

Focus on:
1. Names of people, places, or things
2. Dates or time periods
3. Numerical values or quantities
4. Specific facts or details directly related to the question

Provide a concise response, either restating the answer if complete or explicitly mentioning the missing information.
"""

INFER_MISSING_INFO_PROMPT = """
Given the question and the current answer which lacks specific information, attempt to infer or deduce the missing details based on the context of the question and any available information. Consider:

1. Common knowledge related to the topic
2. Logical deductions based on the question's context
3. Typical patterns or relationships in similar scenarios

If a reasonable inference can be made, provide it as the answer. If no reliable inference is possible, state that the specific information cannot be determined based on the available context.

Provide a concise response, focusing solely on the inferred information or the statement that it cannot be determined.
"""