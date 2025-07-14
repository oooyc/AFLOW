# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of aflow
import ast
import builtins
import concurrent
import random
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Any

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.operator_an import *
from metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.op_prompt import *
from examples.aflow.w_action_node.utils import test_case_2_test_function
from metagpt.actions.action_node import ActionNode
from metagpt.llm import LLM
from metagpt.logs import logger
import re
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import threading
import logging
import time


class Operator:
    def __init__(self, name, llm: LLM):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Custom(Operator):
    def __init__(self, llm: LLM, name: str = "Custom"):
        super().__init__(name, llm)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        node = await ActionNode.from_pydantic(GenerateOp).fill(context=prompt, llm=self.llm, mode="single_fill")
        response = node.instruct_content.model_dump()
        return response
