from typing import Literal
import examples.aflow.scripts.optimized.HumanEval.graphs.template.operator as operator
import examples.aflow.scripts.optimized.HumanEval.graphs.round_9.prompt as prompt_custom
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
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
        
        # Add a review step
        review = await self.custom(input=f"Problem: {problem}\nGenerated solution: {solution['response']}", instruction=prompt_custom.REVIEW_PROMPT)
        
        if "No issues found" not in review['response']:
            # If issues are found, generate an improved solution
            improved_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"{prompt_custom.CODE_GENERATE_PROMPT}\n\nConsider this review: {review['response']}")
            solution = improved_solution
        
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            # Comprehensive analysis of the problem and failed solution
            analysis = await self.custom(input=f"Problem: {problem}\nFailed solution: {solution['response']}\nTest result: {test_result['solution']}", instruction=prompt_custom.COMPREHENSIVE_ANALYSIS_PROMPT)
            
            # Generate improved solution based on the comprehensive analysis
            improved_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"{prompt_custom.CODE_GENERATE_PROMPT}\n\nConsider this analysis: {analysis['response']}")
            
            # Test the improved solution
            improved_test_result = await self.test(problem=problem, solution=improved_solution['response'], entry_point=entry_point)
            
            if improved_test_result['result']:
                return improved_solution['response'], self.llm.cost_manager.total_cost
            else:
                # If the improved solution still fails, return the original solution
                return solution['response'], self.llm.cost_manager.total_cost
        
        return solution['response'], self.llm.cost_manager.total_cost
