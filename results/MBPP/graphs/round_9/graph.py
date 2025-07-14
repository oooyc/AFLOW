from typing import Literal
import examples.aflow.scripts.optimized.MBPP.graphs.template.operator as operator
import examples.aflow.scripts.optimized.MBPP.graphs.round_9.prompt as prompt_custom
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
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
        
        # Test the generated solution
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            # If the test fails, try to fix the solution
            fix_instruction = f"The following solution failed the test cases. Please fix it:\n\n{solution['response']}"
            fixed_solution = await self.custom(input=problem, instruction=fix_instruction)
            return fixed_solution['response'], self.llm.cost_manager.total_cost
        
        return solution['response'], self.llm.cost_manager.total_cost
