# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of ags
import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Type
from pydantic import BaseModel, Field, create_model
from workspace.DROP.workflows.template.operator_an import *
from workspace.DROP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio
import re

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
    
# class AnswerGenerate(Operator):
#     def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
#         super().__init__(llm, name)

#     async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
#         prompt = ANSWER_GENERATION_PROMPT.format(input=input)
#         response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
#         return response
    

# 假设 AnswerGenerateOp 定义如下，如果未定义，需要先定义
# class AnswerGenerateOp(BaseModel):
#     thought: str = Field(default="", description="The step by step thinking process")
#     answer: str = Field(default="", description="The final answer to the question")

class AnswerGenerate(Operator):
    # __init__ 方法不需要改变，直接继承即可
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    # 新增：动态创建模型的私有方法
    def _create_answer_generate_op(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建AnswerGenerateOp模型类"""
        if rate_input:
            # Pydantic v2 的 create_model 语法
            return create_model(
                'DynamicAnswerGenerateOp',
                score=(int, Field(..., description="A score from 1 to 10 evaluating the input quality.")),
                justification=(str, Field(..., description="A brief justification for the score.")),
                thought=(str, Field(default="", description="The step by step thinking process")),
                answer=(str, Field(default="", description="The final answer to the question"))
            )
        else:
            return AnswerGenerateOp

    # 修改：__call__ 方法
    async def __call__(self, input: str, rate_input: bool = False) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        
        # 新增：根据 rate_input 添加评估提示
        if rate_input:
            prompt = VALUATION_PROMPT + prompt

        # 新增：动态获取 op_class
        op_class = self._create_answer_generate_op(rate_input)

        # 修改：调用 _fill_node 时传入 op_class 和 rate_input
        response = await self._fill_node(
            op_class=op_class, 
            prompt=prompt, 
            mode="xml_fill",
            rate_input=rate_input
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