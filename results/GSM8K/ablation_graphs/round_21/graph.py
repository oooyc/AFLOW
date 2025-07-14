from typing import Literal
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.round_21.prompt as prompt_custom
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
        solution1 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_APPROACH1_PROMPT)
        solution2 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_APPROACH2_PROMPT)
        solution3 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_APPROACH3_PROMPT)
        
        combined_solutions = f"Solution 1: {solution1['response']}\nSolution 2: {solution2['response']}\nSolution 3: {solution3['response']}"
        
        final_solution = await self.custom(input=problem + "\n" + combined_solutions, instruction=prompt_custom.COMPARE_AND_SELECT_PROMPT)
        
        return final_solution['response'], self.llm.cost_manager.total_cost
                    