from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_14.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the graph
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="Generate a Python function to solve the given problem.")
        
        # Review and improve the generated solution
        reviewed_solution = await self.custom(input=f"Problem: {problem}\nGenerated solution: {solution['response']}", instruction=prompt_custom.REVIEW_SOLUTION_PROMPT)
        
        # Test the reviewed solution
        test_result = await self.test(problem=problem, solution=reviewed_solution['response'], entry_point=entry_point)
        
        if test_result['result']:
            return test_result['solution'], self.llm.cost_manager.total_cost
        else:
            # If the test fails, try to improve the solution
            improved_solution = await self.custom(input=f"Problem: {problem}\nFailed solution: {reviewed_solution['response']}\nError: {test_result['solution']}", instruction=prompt_custom.IMPROVE_SOLUTION_PROMPT)
            return improved_solution['response'], self.llm.cost_manager.total_cost
