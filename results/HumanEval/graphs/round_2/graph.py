from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_2.prompt as prompt_custom
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
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            # If the test fails, use Custom to analyze and improve the solution
            analysis = await self.custom(input=f"Problem: {problem}\nFailed solution: {solution['response']}\nTest result: {test_result['solution']}", instruction=prompt_custom.ANALYZE_AND_IMPROVE_PROMPT)
            improved_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"{prompt_custom.CODE_GENERATE_PROMPT}\n\nConsider this analysis: {analysis['response']}")
            return improved_solution['response'], self.llm.cost_manager.total_cost
        
        return solution['response'], self.llm.cost_manager.total_cost
