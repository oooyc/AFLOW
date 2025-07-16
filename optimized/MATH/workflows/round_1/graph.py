# optimized/MATH/workflows/round_1/graph.py
from typing import Literal
import optimized.MATH.workflows.template.operator as operator
import optimized.MATH.workflows.round_1.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]