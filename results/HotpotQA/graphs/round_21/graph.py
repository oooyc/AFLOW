from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.round_21.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        solutions = []
        for _ in range(3):
            initial_response = await self.answer_generate(input=problem)
            thought_process = initial_response['thought']
            initial_answer = initial_response['answer']
            solutions.append(initial_answer)

        ensemble_result = await self.sc_ensemble(solutions=solutions)
        best_answer = ensemble_result['response']
        
        # New step: Check if it's a yes/no question and perform fact-checking
        yes_no_check = await self.custom(
            input=f"Question: {problem}\nBest answer: {best_answer}",
            instruction=prompt_custom.YES_NO_FACT_CHECK_PROMPT
        )
        
        if yes_no_check['response'].lower() in ['yes', 'no']:
            best_answer = yes_no_check['response']
        
        name_extracted_answer = await self.custom(
            input=f"Question: {problem}\nBest answer: {best_answer}",
            instruction=prompt_custom.NAME_EXTRACTION_PROMPT
        )
        
        identity_check_answer = await self.custom(
            input=f"Question: {problem}\nName-extracted answer: {name_extracted_answer['response']}",
            instruction=prompt_custom.IDENTITY_CHECK_PROMPT
        )
        
        profession_check_answer = await self.custom(
            input=f"Question: {problem}\nIdentity-checked answer: {identity_check_answer['response']}",
            instruction=prompt_custom.PROFESSION_CHECK_PROMPT
        )
        
        fact_checked_answer = await self.custom(
            input=f"Question: {problem}\nProfession-checked answer: {profession_check_answer['response']}",
            instruction=prompt_custom.FACT_CHECK_PROMPT
        )
        
        style_check_answer = await self.custom(
            input=f"Question: {problem}\nFact-checked answer: {fact_checked_answer['response']}",
            instruction=prompt_custom.STYLE_CHECK_PROMPT
        )
        
        additional_verification = await self.custom(
            input=f"Question: {problem}\nStyle-checked answer: {style_check_answer['response']}",
            instruction=prompt_custom.ADDITIONAL_VERIFICATION_PROMPT
        )
        
        refined_solution = await self.custom(
            input=f"Question: {problem}\nDouble-verified answer: {additional_verification['response']}",
            instruction=prompt_custom.FORMAT_ANSWER_PROMPT
        )
        
        return refined_solution['response'], self.llm.cost_manager.total_cost
                    