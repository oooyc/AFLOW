from typing import Literal
import metagpt.ext.aflow.scripts.optimized.DROP.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.DROP.graphs.round_9.prompt as prompt_custom
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
        step_by_step = await self.answer_generate(input=problem)
        refined_solution = await self.custom(input=f"Question: {problem}\nStep-by-step solution: {step_by_step['thought']}\nInitial answer: {step_by_step['answer']}", instruction=prompt_custom.REFINE_ANSWER_PROMPT)
        verified_solution = await self.custom(input=f"Question: {problem}\nRefined answer: {refined_solution['response']}", instruction=prompt_custom.VERIFY_ANSWER_PROMPT)
        return verified_solution['response'], self.llm.cost_manager.total_cost
                    