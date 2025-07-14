from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.round_5.prompt as prompt_custom
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
        
        fact_checked_answer = await self.custom(
            input=f"Question: {problem}\nBest answer: {best_answer}",
            instruction=prompt_custom.FACT_CHECK_PROMPT
        )
        
        refined_solution = await self.custom(
            input=f"Question: {problem}\nVerified answer: {fact_checked_answer['response']}",
            instruction=prompt_custom.FORMAT_ANSWER_PROMPT
        )
        
        return refined_solution['response'], self.llm.cost_manager.total_cost
                    