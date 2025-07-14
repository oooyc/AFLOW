from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_7.prompt as prompt_custom
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
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
            solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Self-review step
        review = await self.custom(input=f"Problem: {problem}\nGenerated solution: {best_solution['response']}", instruction=prompt_custom.SELF_REVIEW_PROMPT)
        
        # Improve the solution based on the review
        improved_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"{prompt_custom.CODE_GENERATE_PROMPT}\n\nConsider this review: {review['response']}")
        
        test_result = await self.test(problem=problem, solution=improved_solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            analysis = await self.custom(input=f"Problem: {problem}\nFailed solution: {improved_solution['response']}\nTest result: {test_result['solution']}", instruction=prompt_custom.ANALYZE_AND_IMPROVE_PROMPT)
            final_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"{prompt_custom.CODE_GENERATE_PROMPT}\n\nConsider this analysis: {analysis['response']}")
            return final_solution['response'], self.llm.cost_manager.total_cost
        
        return improved_solution['response'], self.llm.cost_manager.total_cost
