from typing import Literal
import metagpt.ext.aflow.scripts.optimized.DROP.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.DROP.graphs.round_12.prompt as prompt_custom
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
        for _ in range(3):  # Generate 3 solutions
            step_by_step = await self.answer_generate(input=problem)
            solutions.append(step_by_step['answer'])
        
        ensemble_solution = await self.sc_ensemble(solutions=solutions)
        
        refined_solution = await self.custom(input=f"Question: {problem}\nBest solution: {ensemble_solution['response']}", instruction=prompt_custom.REFINE_ANSWER_PROMPT)
        
        final_solution = await self.custom(input=f"Answer: {refined_solution['response']}", instruction=prompt_custom.FORMAT_NUMERIC_ANSWER_PROMPT)

        return final_solution['response'], self.llm.cost_manager.total_cost
                    