# -*- coding: utf-8 -*-
# @Date    : 4/1/2025 21:01 PM
# @Author  : didi
# @Desc    : Run baseline to test prompts 

import asyncio
from typing import Literal
from scripts.operators import AnswerGenerate
from scripts.workflow import Workflow
from scripts.async_llm import LLMsConfig
from benchmarks.hotpotqa import HotpotQABenchmark
from scripts.logs import logger
from scripts.async_llm import create_llm_instance

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class PoWorkflow(Workflow):
    def __init__(
        self,
        test_prompt: str,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.prompt = test_prompt
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.response = AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        full_input = self.prompt + problem
        solution = await self.response(input=full_input)
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]

async def main():
    mini_llm_config = LLMsConfig.default().get("gpt-4o-mini")
    spo_prompt = "You are an ultra-precise answer machine. Follow these rules without exception:" \
        "1. Provide ONLY the exact answer requested - nothing more" \
        "2. Format answers as a single line in bold text" \
        "3. Never include:" \
        "- Explanations" \
        "- Context" \
        "- Reasoning" \
        "- Qualifiers" \
        "- Hedging language" \
        "- Introductory phrases" \
        "- Additional details" \
        "- Bullet points" \
        "4. If a specific number, name, date, or term is requested, provide only that" \
        "5. Use maximum 5 words unless absolutely impossible" \
        "6. Never apologize or explain your brevity" \
        "Your sole purpose is to deliver the minimum viable answer with perfect accuracy." 
    spo_test_workflow = PoWorkflow(test_prompt=spo_prompt, name="spo_test", llm_config=mini_llm_config, dataset="HotpotQA")
    # file path refer to the dataset you want to use.
    # log_path refer to the folder of output csv.
    test_hotpotqa_benchmark = HotpotQABenchmark(name="HotpotQA", file_path="data/hotpotqa_validate.jsonl", log_path="")
    
    results = await test_hotpotqa_benchmark.run_baseline(spo_test_workflow)

if __name__ == "__main__":
    asyncio.run(main())
