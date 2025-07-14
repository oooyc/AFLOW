from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_20.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        # Generate multiple solutions
        solutions = []
        for _ in range(3):
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_SOLUTION_PROMPT)
            solutions.append(solution['response'])

        # Use ScEnsemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Review and improve the selected solution
        reviewed_solution = await self.custom(input=f"Problem: {problem}\nGenerated solution: {best_solution['response']}", instruction=prompt_custom.REVIEW_SOLUTION_PROMPT)
        
        # Test the reviewed solution
        test_result = await self.test(problem=problem, solution=reviewed_solution['response'], entry_point=entry_point)
        
        if test_result['result']:
            return test_result['solution'], self.llm.cost_manager.total_cost
        else:
            # If the test fails, try to improve the solution
            improved_solution = await self.custom(input=f"Problem: {problem}\nFailed solution: {reviewed_solution['response']}\nError: {test_result['solution']}", instruction=prompt_custom.IMPROVE_SOLUTION_PROMPT)
            return improved_solution['response'], self.llm.cost_manager.total_cost
