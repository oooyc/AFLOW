from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.graphs.round_18.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        initial_response = await self.answer_generate(input=problem)
        refined_solution = await self.custom(input=problem + f"\nThought process: {initial_response['thought']}\nInitial answer: {initial_response['answer']}", instruction=prompt_custom.REFINE_PROMPT)
        return refined_solution['response'], self.llm.cost_manager.total_cost
                    