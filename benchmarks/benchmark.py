import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    # async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 50, validation_n = None, round = None):
    #     semaphore = asyncio.Semaphore(max_concurrent_tasks)

    #     async def sem_evaluate(i, problem, validation_n = None, round = None):
    #         async with semaphore:
    #             return await self.evaluate_problem(i, problem, agent, validation_n=validation_n, round=round)

    #     tasks = [sem_evaluate(i, problem, validation_n=validation_n, round=round) for i, problem in enumerate(data)]
    #     return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def evaluate_all_problems(self, data: List[dict], agent_factory: Callable, max_concurrent_tasks: int = 50, validation_n = None, round = None):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)

            # highlight-start
            # --- 关键改动: 在这里创建独立的实例 ---
            async def sem_evaluate(i, problem, validation_n = None, round = None):
                async with semaphore:
                    # agent_factory 就是我们从Evaluator传过来的lambda函数
                    # 每次调用，都会生成一个全新的、干净的graph实例
                    graph_instance = agent_factory()
                    return await self.evaluate_problem(i, problem, graph_instance, validation_n=validation_n, round=round)
            # highlight-end

            tasks = [sem_evaluate(i, problem, validation_n=validation_n, round=round) for i, problem in enumerate(data)]
            return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(self, agent_factory: Callable, va_list: List[int], max_concurrent_tasks: int = 50, validation_n = None, round = None):
        data = await self.load_data(va_list)
        # 将 agent_factory 透传下去
        results = await self.evaluate_all_problems(data, agent_factory, max_concurrent_tasks, validation_n = validation_n, round = round)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

