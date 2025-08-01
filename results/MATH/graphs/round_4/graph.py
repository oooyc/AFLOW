from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.graphs.round_4.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        # Use Programmer to generate and execute Python code
        code_solution = await self.programmer(problem=problem)
        
        # Use Custom to refine and format the answer
        refined_solution = await self.custom(input=problem + f"\nCode output: {code_solution['output']}", instruction=prompt_custom.REFINE_ANSWER_PROMPT)
        
        # Generate multiple solutions using Custom with different approaches
        solutions = []
        for approach in ['algebraic', 'geometric', 'analytical']:
            solution = await self.custom(input=problem, instruction=prompt_custom.GENERATE_SOLUTION_PROMPT.format(approach=approach))
            solutions.append(solution['response'])
        
        # Add the refined solution to the list of solutions
        solutions.append(refined_solution['response'])
        
        # Use ScEnsemble to select the most consistent solution
        final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        return final_solution['response'], self.llm.cost_manager.total_cost
                    