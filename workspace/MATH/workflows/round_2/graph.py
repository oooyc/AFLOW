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
        self.sc_ensemble = operator.ScEnsemble(self.llm, eval_log=self.node_evaluations)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple candidate solutions
        candidates = []
        for _ in range(3):
            response = await self.custom(
                input=problem,
                instruction=prompt_custom.BASE_PROMPT,
                rate_input=False
            )
            candidates.append(response['response'])
        
        # Select most consistent answer via self-consistency
        result = await self.sc_ensemble(
            solutions=candidates,
            problem=problem,
            rate_input=True
        )
        return result['response'], self.llm.get_usage_summary()["total_cost"]
