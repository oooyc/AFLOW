# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of aflow
import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Type
from pydantic import BaseModel, Field, create_model
from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.MBPP.workflows.template.operator_an import *
from workspace.MBPP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function


from scripts.operators import Operator



class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    def _create_custom_model(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建模型类"""
        if rate_input:
            base_fields = ({
                "score": (int, Field(..., description="A score from 1 to 10 evaluating the input quality.")),
                "justification": (str, Field(..., description="A brief justification for the score.")),
                "response": (str, Field(default="", description="Your solution for this problem."))
            })
            return create_model("DynamicCustomResponseModel", **base_fields)
        else:
            return GenerateOp


    async def __call__(self, input, instruction, rate_input=False):
        prompt = instruction + input
        if rate_input:
            prompt = VALUATION_PROMPT + prompt

        # 动态创建模型类
        op_class = self._create_custom_model(rate_input)

        response = await self._fill_node(op_class, prompt, mode="single_fill", rate_input=rate_input)
        return response
    
class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate", eval_log: Optional[list] = None):
        # 修改：__init__ 以接收 eval_log
        super().__init__(llm, name, eval_log)

    # 新增：动态创建模型的私有方法
    def _create_custom_code_generate_op(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建代码生成模型类"""
        if rate_input:
            # 假设 GenerateOp 只有一个 'response' 字段
            return create_model(
                'DynamicCustomCodeGenerateOp',
                score=(int, Field(..., description="A score from 1 to 10 evaluating the input quality.")),
                justification=(str, Field(..., description="A brief justification for the score.")),
                response=(str, Field(default="", description="Your solution for this problem"))
            )
        else:
            # GenerateOp 似乎定义在 op_prompt.py 中，这里直接使用
            return GenerateOp

    # 修改：__call__ 方法
    async def __call__(self, problem, entry_point, instruction, rate_input: bool = False):
        prompt = instruction + problem

        # 新增：根据 rate_input 添加评估提示
        if rate_input:
            prompt = VALUATION_PROMPT + prompt
        
        # 新增：动态获取 op_class
        op_class = self._create_custom_code_generate_op(rate_input)
        
        # 修改：调用 _fill_node 时传入 op_class 和 rate_input
        response = await self._fill_node(
            op_class=op_class, 
            prompt=prompt, 
            mode="code_fill", 
            rate_input=rate_input,
            function_name=entry_point
        )
        return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    def _create_sc_ensemble_op(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建ScEnsembleOp模型类"""
        if rate_input:
            class newScEnsembleOp(BaseModel):
                score: int = Field(..., description="A score from 1 to 10 evaluating the input quality.")
                justification: str = Field(..., description="A brief justification for the score.")
                thought: str = Field(default="", description="The explanation of the most consistent solution.")
                solution_letter: str = Field(default="", description="The letter of most consistent solution.")
            return newScEnsembleOp
        else:
            return ScEnsembleOp
        
    async def __call__(self, solutions: List[str], problem: str, rate_input=False):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        if rate_input:
            prompt = VALUATION_PROMPT + prompt

        # 动态创建Op类
        op_class = self._create_sc_ensemble_op(rate_input)
        
        response = await self._fill_node(
            op_class=op_class,  # 使用动态创建的类
            prompt=prompt,
            mode="xml_fill",
            rate_input=rate_input
        )

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test", eval_log: Optional[list] = None):
        # 修改：__init__ 以接收 eval_log
        super().__init__(llm, name, eval_log)

    # exec_code 方法保持不变
    def exec_code(self, solution, entry_point):
        test_cases = extract_test_cases_from_jsonl(entry_point)

        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    # 新增：动态创建模型的私有方法
    def _create_reflection_test_op(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建ReflectionTestOp模型类"""
        if rate_input:
            return create_model(
                'DynamicReflectionTestOp',
                score=(int, Field(..., description="A score from 1 to 10 evaluating the input quality.")),
                justification=(str, Field(..., description="A brief justification for the score.")),
                reflection_and_solution=(str, Field(default="", description="Corrective solution for code execution errors or test case failures"))
            )
        else:
            return ReflectionTestOp

    # 修改：__call__ 方法
    async def __call__(self, problem, solution, entry_point, test_loop: int = 3, rate_input: bool = False):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        # 新增：在循环外获取 op_class
        op_class = self._create_reflection_test_op(rate_input)

        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            
            prompt = ""
            if "exec_fail_case" in result:
                exec_error = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {exec_error}",
                    test_fail="executed unsucessfully",
                )
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )

            # 新增：应用 rate_input 逻辑
            if rate_input:
                prompt = VALUATION_PROMPT + prompt
            
            # 修改：调用 _fill_node 时传入 op_class 和 rate_input
            response = await self._fill_node(
                op_class=op_class,
                prompt=prompt,
                mode="code_fill",
                rate_input=rate_input
            )
            # 注意这里假设你的 ReflectionTestOp 在 code_fill 模式下返回的 solution 字段是 reflection_and_solution
            # 如果 response 中没有这个 key，可能需要调整
            solution = response.get("response", "")
            if not solution:
                solution = response.get("reflection_and_solution", solution)

        # 最终检查
        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}