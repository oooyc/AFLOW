# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import time
from typing import List, Literal, Dict
import random
from pydantic import BaseModel, Field
import json
from scripts.prompts.prompt import (
    CORRECTION_PROMPT_TEMPLATE,
)
import os
from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
import traceback
import re
from scripts.optimizer_utils.diversity_utils import WorkflowSimilarity, analyze_and_present_families
from scripts.exception import WorkflowSyntaxError, WorkflowAttributeError
from scripts.prompts.prompt import SYNTAX_CORRECTION_PROMPT_TEMPLATE

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default=None, description="modification")
    graph: str = Field(default=None, description="graph")
    prompt: str = Field(default=None, description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds
        self.workflow_similarity = WorkflowSimilarity(self.root_path)
        self.graph_utils = GraphUtils(self.root_path)
        # 加载graphs、写入graphs，组装优化prompt，实现优化，取出各个tag的内容做一个写入
        self.data_utils = DataUtils(self.root_path)
        # 加载results.json，维护一个top score数组，实现graph的select
        self.experience_utils = ExperienceUtils(self.root_path)
        # load每一个round的experience进行处理，对于选中轮的experience会做一个format，另外还会check modification是否重复
        self.evaluation_utils = EvaluationUtils(self.root_path)
        # 调用evaluator进行评估
        self.convergence_utils = ConvergenceUtils(self.root_path)
        # 收敛性评估，比较复杂，综合考虑这一轮topk和上一轮topk得分的workflow自身的不稳定性，当这一轮topk的平均得分与上一轮topk平均得分的差值小于两者综合的不稳定性，就认为收敛了

    async def _correct_attribute_error(self, error: WorkflowAttributeError) -> bool:
        """修复因 prompt 变量被注释而引发的 AttributeError。"""
        round_to_fix = error.round_number
        # 从错误对象中直接获取需要修复的文件路径
        file_to_fix_path = os.path.join(self.root_path, 'workflows', f'round_{round_to_fix}', 'prompt.py')
        
        logger.info(f"启动工作流 Round {round_to_fix} 的属性错误自动修复程序 (文件: {file_to_fix_path})...")
        
        try:
            # 从原始错误信息中提取缺失的属性名
            missing_attr_match = re.search(r"has no attribute '(.+?)'", str(error.original_error))
            if not missing_attr_match:
                logger.error("无法从错误信息中提取属性名。")
                return False
            
            missing_attr = missing_attr_match.group(1)

            with open(file_to_fix_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式找到被注释的变量并取消注释
            # pattern 会匹配如: '# ANALYSIS_PROMPT = """' 或 '#ANALYSIS_PROMPT = """'
            pattern = re.compile(rf'#\s*({missing_attr}\s*=\s*["\']{{3}})', re.MULTILINE)
            
            new_content, num_replacements = pattern.subn(r'\1', content)

            if num_replacements > 0:
                with open(file_to_fix_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"✅ 成功取消对变量 '{missing_attr}' 的注释。文件已覆写。")
                return True
            else:
                logger.error(f"❌ 在文件中未找到被注释的变量 '{missing_attr}'。修复失败。")
                return False

        except Exception as e:
            logger.error(f"❌ 自动修复属性错误失败: {e}")
            traceback.print_exc()
            return False

    async def _correct_workflow_syntax_error(self, error: WorkflowSyntaxError) -> bool:
        """
        处理 WorkflowSyntaxError，调用 LLM 修复并覆写文件。
        """
        round_to_fix = error.round_number
        logger.info(f"启动工作流 Round {round_to_fix} 的自动修复程序...")

        try:
            graph_path = f"{self.root_path}/workflows"
            directory = os.path.join(graph_path, f"round_{round_to_fix}")
            
            # 1. 读取损坏的文件
            prompt_code, graph_code = self.graph_utils.read_graph_files(round_to_fix, graph_path)
            
            # 2. 构建修复提示
            missing_module_match = re.search(r"name '(\w+)' is not defined", str(error.original_error))
            missing_module = missing_module_match.group(1) if missing_module_match else "unknown"

            correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
                error_type=type(error.original_error).__name__,
                error_message=str(error.original_error),
                graph_code=self.graph_utils.extract_solve_graph(graph_code)[0],
                prompt_code=prompt_code,
                missing_module=missing_module
            )

            # 3. 调用 LLM 进行修复 (逻辑类似于优化)
            graph_formatter = XmlFormatter.from_model(GraphOptimize)
            response = await self.optimize_llm.call_with_format(correction_prompt, graph_formatter)
            
            # 4. 覆写文件
            self.graph_utils.write_graph_files(directory, response, round_to_fix, self.dataset)
            logger.info(f"✅ 工作流 Round {round_to_fix} 已被自动修复并覆写。")
            return True
        
        except Exception as e:
            logger.error(f"❌ 自动修复工作流 Round {round_to_fix} 失败: {e}")
            traceback.print_exc()
            return False

    async def _correct_syntax_error(self, error: SyntaxError, round_to_fix: int) -> bool:
        """处理通用的 SyntaxError，调用 LLM 修复并覆写文件。"""
        # error.filename 包含了出错文件的完整路径
        file_to_fix = error.filename
        if file_to_fix is None:
            logger.error("SyntaxError 未提供文件名，无法修复。")
            return False
            
        logger.info(f"启动工作流 Round {round_to_fix} 的通用语法修复程序 (文件: {file_to_fix})...")
        
        try:
            with open(file_to_fix, 'r', encoding='utf-8') as f:
                broken_code = f.read()
            
            # 构建修复提示
            correction_prompt = SYNTAX_CORRECTION_PROMPT_TEMPLATE.format(
                error_message=str(error),
                line_number=error.lineno,
                error_text=error.text,
                full_code=broken_code
            )
            
            # 直接调用 LLM 获取修复后的代码字符串，不使用 XML 格式
            fixed_code = await self.optimize_llm(correction_prompt)
            
            # 覆写文件
            with open(file_to_fix, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
                
            logger.info(f"✅ 文件 {os.path.basename(file_to_fix)} 已被 LLM 自动修复并覆写。")
            return True

        except Exception as e:
            logger.error(f"❌ 自动修复通用语法错误失败: {e}")
            traceback.print_exc()
            return False

    def _select_diverse_parent_workflow(self) -> Dict:
        """
        封装了完整的“多样性分析 -> 家族划分 -> 概率选举 -> 经验池化”流程
        """
        logger.info("Starting diverse parent selection process...")
        workflows_path = f"{self.root_path}/workflows"
        
        # 1. 加载所有需要的数据
        candidate_workflows = self.data_utils.load_all_candidate_workflows()
        if not candidate_workflows:
            raise ValueError("Candidate workflow pool is empty. Cannot proceed.")
        
        processed_experience = self.experience_utils.load_experience(path=workflows_path)
        # 假设 failure logs 也由 experience_utils 或 data_utils 加载
        # 您需要根据实际情况实现 load_all_failure_logs
        failure_logs = self.data_utils.load_all_failure_logs(workflows_path) 
        
        # 2. 运行相似度分析，找到所有家族
        logger.info(f"Analyzing {len(candidate_workflows)} workflows to form families...")
        all_families, _ = self.workflow_similarity.find_similar_workflows(
            [wf['round'] for wf in candidate_workflows], 
            processed_experience,
            failure_logs,
            self.graph_utils,
            workflows_path,
            similarity_threshold=0.7 # 您可以配置这个阈值
        )
        
        # 3. 进行概率性代表选举，并为代表附加池化经验
        logger.info("Electing representatives from families and pooling experience...")
        representative_pool = analyze_and_present_families(
            candidate_workflows,
            all_families,
            processed_experience,
            failure_logs
        )
        
        # 4. 从最终的代表池中，进行全局概率选择，确定本轮的父节点
        logger.info("Selecting final parent from the diverse pool of representatives...")
        final_sample = self.data_utils.select_round(representative_pool)
        
        logger.info(f"Final selected parent: Round {final_sample['round']}. It carries the wisdom of its family.")
        return final_sample

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            # 这里的retry是针对生成的retry
            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})"
                                    f"失败类型：{type(e).__name__}，原因：{str(e) or '无详细信息'}，"
                                    f"Traceback：{traceback.format_exc()}"
                                   )
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(1, graph_path)
            # self.graph = self.graph_utils.load_graph(self.round, graph_path)
            # avg_score = 0.58
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True, round=round)
        # Create a loop until the generated graph meets the check conditions
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            # =======================> 核心替换 <=======================
            # 旧逻辑:
            # top_rounds = self.data_utils.get_top_rounds(self.sample)
            # sample = self.data_utils.select_round(top_rounds)
            
            # 新逻辑:
            sample = self._select_diverse_parent_workflow()
            # =======================> 替换结束 <=======================

            # 读取选定父节点的原始文件
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)
            all_experience = self.experience_utils.load_experience()
            # 使用池化后的经验和日志
            family_info = sample['families'] # 需要在 analyze_and_select_representatives 中附加 family 信息
            experience = self.experience_utils.format_pooled_experience(sample['round'], family_info, all_experience)
            sample_size = min(3, len(sample["pooled_logs"]))
            random_samples = random.sample(sample["pooled_logs"], sample_size)
            log_data = json.dumps(random_samples, indent=3, ensure_ascii=False)

            operator_description = self.graph_utils.load_operators_description(self.operators)

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
            )
            # print(graph_optimize_prompt)
            # Replace ActionNode with AsyncLLM and XmlFormatter
            try:
                # Create XmlFormatter based on GraphOptimize model
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                
                # Call the LLM with formatter
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt, 
                    graph_formatter
                )
                
                # If we reach here, response is properly formatted and validated
                logger.info(f"Graph optimization response received successfully")
            except FormatError as e:
                # Handle format validation errors
                logger.error(f"Format error in graph optimization: {str(e)}")
                # Try again with a fallback approach - direct call with post-processing
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                
                # Try to extract fields using basic parsing
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error("Failed to extract fields from raw response, retrying...")
                    continue

            # Check if the modification meets the conditions
            check = self.experience_utils.check_modification(
                all_experience, response["modification"], sample["round"]
            )

            # If `check` is True, break the loop; otherwise, regenerate the graph
            if check:
                break

        # Save the graph and evaluate
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        # --- 4. 评估与修复阶段 (EVALUATION & CORRECTION LOOP) ---
        avg_score = 0.0
        max_eval_attempts = 3
        for attempt in range(max_eval_attempts):
            try:
                logger.info(f"Evaluating new workflow Round {self.round+1} (Attempt {attempt + 1}/{max_eval_attempts})")
                
                # 加载图现在也放在 try 块中，以便捕获 SyntaxError
                self.graph = self.graph_utils.load_graph(self.round+1, graph_path)
                
                global_results_data = self.data_utils.load_results(graph_path)
                avg_score = await self.evaluation_utils.evaluate_graph(
                    self, directory, validation_n, global_results_data, 
                    initial=False, round=self.round+1
                )
                
                logger.info(f"✅ Evaluation successful for Round {self.round+1}. Score: {avg_score}")
                break 
            
            # 捕获 NameError (缺少 import)
            except WorkflowSyntaxError as e:
                logger.warning(f"Caught fixable syntax error in Round {e.round_number}. Initiating self-correction...")
                success = await self._correct_workflow_syntax_error(e)
                if not success: break
            
            # 捕获 AttributeError (变量被注释)
            except WorkflowAttributeError as e:
                logger.warning(f"Caught fixable attribute error in Round {e.round_number}. Initiating self-correction...")
                success = await self._correct_attribute_error(e)
                if not success: break

            # 捕获通用 SyntaxError (代码格式错误)
            except SyntaxError as e:
                logger.warning(f"Caught generic syntax error in Round {self.round+1}. Initiating LLM-based correction...")
                success = await self._correct_syntax_error(e, self.round+1)
                if not success: break
                
            # 捕获其他所有异常
            except Exception as e:
                logger.error(f"❌ Unhandled error during evaluation of Round {self.round+1}: {e}")
                traceback.print_exc()
                avg_score = 0.0
                break

            # 如果修复成功，日志会在修复函数内部打印，这里直接继续下一次尝试
            if success:
                logger.info(f"Correction successful. Retrying evaluation...")
                continue
            else:
                logger.error(f"❌ Correction failed for Round {self.round+1}. Assigning score 0.")
                avg_score = 0.0
                break

        # --- 5. 收尾阶段 (FINALIZATION PHASE) ---
        self.experience_utils.update_experience(directory, experience, avg_score)
        return avg_score

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re
            
            # Initialize result dictionary with default values
            result = {
                "modification": "",
                "graph": "",
                "prompt": ""
            }
            
            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()
            
            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None
            
            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self):
        rounds = [3,4,5,6,7,8]  # You can choose the rounds you want to test here.
        data = []

        graph_path = f"{self.root_path}/workflows"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)