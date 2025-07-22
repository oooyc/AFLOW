from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

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
        self.node_evaluations = []
        self.custom = operator.Custom(self.llm, eval_log=self.node_evaluations)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT, rate_input=False)
        # Format and box final answer
        formatted = await self.custom(input=solution['response'], instruction=prompt_custom.FORMAT_PROMPT, rate_input=True)
        return formatted['response'], self.llm.get_usage_summary()["total_cost"]
