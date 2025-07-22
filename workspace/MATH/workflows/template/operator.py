# workspace/MATH/workflows/template/operator.py
import concurrent
import sys
import traceback
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter, WrapperXmlFormatter
from workspace.MATH.workflows.template.operator_an import *
from workspace.MATH.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio
import re

class Operator:
    def __init__(self, llm: AsyncLLM, name: str, eval_log: Optional[list] = None):
        self.name = name
        self.llm = llm
        self.eval_log = eval_log

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, function_name="solve", rate_input=False):
        # Create appropriate formatter based on mode
        if rate_input:
            match = re.search(r'Your main task:(.*)', prompt, re.DOTALL)
            node_inputs = match.group(1).strip()  # 提取并去除首尾空白
            formatter = self._create_formatter_wrapper(op_class, mode, function_name=function_name)
        else:
            formatter = self._create_formatter(op_class, mode, function_name=function_name)
            
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response, input_rating = await self.llm.call_with_format(prompt, formatter, rate_input=rate_input)
            else:
                # Fallback to direct call if no formatter is needed
                response, input_rating = await self.llm(prompt)
                
            # 如果有日志本，就把自己的评估记录下来
            if self.eval_log is not None and rate_input:
                log_entry = {
                    "node_name": self.name,
                    "input_rating": input_rating,
                    # 还可以记录节点的输入是什么，便于追溯，当时估计会导致上下文过长，还是别加了
                    # "node_inputs": node_inputs
                }
                self.eval_log.append(log_entry)

            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    
    def _create_formatter_wrapper(self, op_class, mode=None, function_name='solve') -> Optional[BaseFormatter]:
        # 定义外层XML的模板
        class ResponseWrapper(BaseModel):
            score: str
            justification: str
            task_output: str

        if mode == "code_fill":
            # 外层用XML解析，内层用CodeFormatter解析
            return WrapperXmlFormatter(op_class=ResponseWrapper, inner_formatter=CodeFormatter(function_name=function_name))
        elif mode == "xml_fill":
            # 内外都是XML，内层formatter用任务本身的op_class
            inner_xml_formatter = XmlFormatter.from_model(op_class)
            return WrapperXmlFormatter(op_class=ResponseWrapper, inner_formatter=inner_xml_formatter)
        elif mode == "single_fill":
            return WrapperXmlFormatter(op_class=ResponseWrapper, inner_formatter=TextFormatter())
        else:
            return None

    def _create_formatter(self, op_class, mode=None, function_name='solve') -> Optional[BaseFormatter]:
        # """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    async def __call__(self, input, instruction, rate_input=False):
        prompt = instruction + input
        if rate_input:
            prompt = VALUATION_PROMPT + prompt
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill", rate_input=rate_input)
        return response

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode, rate_input):
        """
        Asynchronous method to generate code.
        """

        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        if rate_input:
            score_prompt = VALUATION_PROMPT
            prompt = score_prompt + prompt
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve", rate_input=rate_input)
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None", rate_input=False):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill", rate_input=rate_input)
            code = code_response.get("code")
            if not code:
                return {"code": code, "response": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble", eval_log: Optional[list] = None):
        super().__init__(llm, name, eval_log)

    async def __call__(self, solutions: List[str], problem: str, rate_input=False):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        if rate_input:
            prompt = VALUATION_PROMPT + prompt

        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill", rate_input=rate_input)

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}