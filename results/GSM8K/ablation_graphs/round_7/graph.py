from typing import Literal
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.round_7.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        initial_solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_AND_EXTRACT_PROMPT)
        reviewed_solution = await self.custom(input=problem + f"\nInitial solution: {initial_solution['response']}", instruction=prompt_custom.REVIEW_AND_CORRECT_PROMPT)
        verified_solution = await self.custom(input=problem + f"\nReviewed solution: {reviewed_solution['response']}", instruction=prompt_custom.VERIFY_RANGE_PROMPT)
        return verified_solution['response'], self.llm.cost_manager.total_cost
                    