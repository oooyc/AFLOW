from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_3.prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the graph
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        
        # Add a review step using the Custom operator
        review_result = await self.custom(input=f"Problem: {problem}\nGenerated Solution: {solution['response']}", instruction=prompt_custom.REVIEW_PROMPT)
        
        if "Improved Solution:" in review_result['response']:
            final_solution = review_result['response'].split("Improved Solution:")[1].strip()
        else:
            final_solution = solution['response']
        
        return final_solution, self.llm.cost_manager.total_cost
