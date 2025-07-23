import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
import os

import json
import asyncio

class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str, failure_report_path: str = "failure_report.json"):
        super().__init__(name, file_path, log_path)
        # 主数据结构，用于实时构建报告
        self.failure_report_data: dict = {}
        # 全局总尝试次数计数器
        self.global_total_attempts: int = 0
        self.failure_report_path = os.path.join(log_path, failure_report_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    # @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        # 可以在这里返回中间输出和评分，就是处理graph的return
        # 这一步讲node_evaluations清空，避免重复记录,非常重要！！！！
        # graph.node_evaluations.clear() 
        return await graph(input_text)

    async def evaluate_problem(self, i: int, problem: dict, graph: Callable, validation_n=None, round=None) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]
        
        # round_key = f"round_{round}"
        # validation_key = f"validation_{validation_n}"
        # problem_key = f"problem_{i}"

        # highlight-start
        # --- 更新数据结构，为每个层级添加两种失败计数器 ---
        # round_data = self.failure_report_data.setdefault(
        #     round_key, 
        #     {"round_total_attempts": 0, "round_failed_attempts": 0, "round_failed_problems": 0}
        # )
        # validation_data = round_data.setdefault(
        #     validation_key, 
        #     {"validation_total_attempts": 0, "validation_failed_attempts": 0, "validation_failed_problems": 0}
        # )
        # highlight-end

        max_attempts = 5
        wait_seconds = 1
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            # self.global_total_attempts += 1
            # round_data["round_total_attempts"] += 1
            # validation_data["validation_total_attempts"] += 1
            # 这里可能需要处理graph的返回值！！！！！！！
            try:
                # `graph` 是一个为当前问题专门创建的独立实例
                output, cost = await self._generate_output(graph, input_text)
                
                # highlight-start
                # --- 关键改动: 在调用完成后，通过属性访问日志 ---
                node_evaluations = graph.node_evaluations
                # highlight-end
                
                uni_score, extracted_output = self.calculate_score(expected_output, output)

                if uni_score == 0:
                    self.log_mismatch(
                        problem=input_text, 
                        expected_output=expected_output, 
                        prediction=output, 
                        extracted_output=extracted_output,
                        valuation_log=node_evaluations # <--- 传递获取到的、独立的日志
                    )
                
                # 返回值中也不需要包含 node_evaluations
                return input_text, output, expected_output, uni_score, cost

            except Exception as e:
                last_exception = e
                
                # highlight-start
                # # --- 版本B逻辑: 每次尝试失败，增加 "failed_attempts" 计数 ---
                # round_data["round_failed_attempts"] += 1
                # validation_data["validation_failed_attempts"] += 1
                # # highlight-end
                
                # problem_data = validation_data.setdefault(problem_key, {"failed_attempts": {}})
                # problem_data["failed_attempts"][str(attempt)] = str(e)
                
                logger.warning(f"问题 {i} [Round {round}, Val {validation_n}] 第 {attempt}/{max_attempts} 次尝试失败。失败原因： {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(wait_seconds)
        
        # --- 循环结束后执行 ---
        # highlight-start
        # --- 版本A逻辑: 所有尝试都失败后，增加 "failed_problems" 计数 ---
        # round_data["round_failed_problems"] += 1
        # validation_data["validation_failed_problems"] += 1
        # highlight-end
        
        logger.error(f"问题 {i} [Round {round}, Val {validation_n}] 所有尝试均失败，跳过。")
        return input_text, str(last_exception), expected_output, 0.0, 0.0

    def write_per_round_report(self, round_num: int):
        """
        将指定轮次的失败报告写入独立的JSON文件。
        文件名将基于 `failure_report_path` 自动生成。
        """
        round_key = f"round_{round_num}"
        
        # 1. 获取当前轮次的数据
        round_data_to_save = self.failure_report_data.get(round_key)
        
        if not round_data_to_save:
            logger.warning(f"在第 {round_num} 轮没有找到任何失败数据，跳过写入文件。")
            return

        # 2. 生成新的文件名，例如 "report.json" -> "report_round_1.json"
        base, ext = os.path.splitext(self.failure_report_path)
        output_path = f"{base}_round_{round_num}{ext}"
        
        # 3. 写入JSON文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(round_data_to_save, f, indent=4, ensure_ascii=False)
            logger.info(f"第 {round_num} 轮的失败报告已成功写入到: {output_path}")
        except Exception as e:
            logger.error(f"写入第 {round_num} 轮失败报告时发生错误: {e}")
            
    def clear_round_data(self, round_num: int):
        """
        (可选) 清理指定轮次的内存数据，以节省空间。
        """
        round_key = f"round_{round_num}"
        if round_key in self.failure_report_data:
            del self.failure_report_data[round_key]
            logger.info(f"已清理第 {round_num} 轮的内存数据。")

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
