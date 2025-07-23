# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple

from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
        }

    # async def graph_evaluate(
    #     self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False, validation_n = None, round = None
    # ) -> Tuple[float, float, float]:
    #     if dataset not in self.dataset_configs:
    #         raise ValueError(f"Unsupported dataset: {dataset}")

    #     data_path = self._get_data_path(dataset, is_test)
    #     benchmark_class = self.dataset_configs[dataset]
    #     benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

    #     # Use params to configure the graph and benchmark
    #     configured_graph = await self._configure_graph(dataset, graph, params)
        
    #     if is_test:
    #         va_list = None  # For test data, generally use None to test all
    #     else:
    #         va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
    #     return await benchmark.run_evaluation(configured_graph, va_list, validation_n=validation_n, round=round,)

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False, validation_n=None, round=None
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        self.benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # highlight-start
        # --- 改动点 1: 不再返回实例，而是返回一个配置好的“工厂” ---
        # configured_graph 现在是一个可以被调用来创建新实例的函数或类
        graph_factory = self._configure_graph(dataset, graph, params)
        # highlight-end
        
        if is_test:
            va_list = None
        else:
            va_list = None
        
        # highlight-start
        # --- 改动点 2: 将工厂传递给 benchmark ---
        return await self.benchmark.run_evaluation(graph_factory, va_list, validation_n=validation_n, round=round)
        # highlight-end

    def _configure_graph(self, dataset, graph, params: dict):
            # highlight-start
            # --- 改动点 3: 返回一个可以创建实例的 lambda 函数 ---
            dataset_config = params.get("dataset", {})
            llm_config = params.get("llm_config", {})
            
            # 这个lambda函数捕获了所有必要的配置信息
            # 每次调用它，都会创建一个全新的、配置好的Workflow实例
            return lambda: graph(name=dataset, llm_config=llm_config, dataset=dataset_config)
            # highlight-end

    # async def _configure_graph(self, dataset, graph, params: dict):
    #     # Here you can configure the graph based on params
    #     # For example: set LLM configuration, dataset configuration, etc.
    #     dataset_config = params.get("dataset", {})
    #     llm_config = params.get("llm_config", {})
    #     return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"data/datasets/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
