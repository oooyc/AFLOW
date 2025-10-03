# workspace/MATH/workflows/template/operator.py
import concurrent
import sys
import traceback
from typing import List, Optional, Type, Tuple
from pydantic import BaseModel, Field, create_model

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter, CodeWithRatingFormatter, TextWithRatingFormatter
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

        formatter = self._create_formatter(op_class, mode, function_name=function_name, rate_input=rate_input)
            
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(prompt, formatter, rate_input=rate_input)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(prompt)
                
            # 如果有日志本，就把自己的评估记录下来
            if self.eval_log is not None and rate_input:
                log_entry = {
                    "node_name": self.name,
                    "input_rating": {response.get("score", "NA"): response.get("justification", "There was no upstream node to rate.")},
                    # 还可以记录节点的输入是什么，便于追溯，当时估计会导致上下文过长，还是别加了
                    # "node_inputs": node_inputs
                }
                self.eval_log.append(log_entry)
                response.pop("score", None)  # Remove score from response to match expected format
                response.pop("justification", None)  # Remove justification from response to match expected format
            # Convert to expected format based on the original implementation

            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
            
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    

    def _create_formatter(self, op_class, mode=None, function_name='solve', rate_input=False) -> Optional[BaseFormatter]:
        # """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            if rate_input:
                return CodeWithRatingFormatter(op_class, function_name=function_name)
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            if rate_input:
                return TextWithRatingFormatter(op_class)
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


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
            
    def _create_code_generate_op(self, rate_input: bool) -> Type[BaseModel]:
        """动态创建CodeGenerateOp模型类"""
        if rate_input:
            base_fields = {
                "score": (int, Field(..., description="A score from 1 to 10 evaluating the input quality.")),
                "justification": (str, Field(..., description="A brief justification for the score.")),
                "code": (str, Field(..., description="Your complete code solution for this problem."))
            }
            return create_model("DynamicCodeGenerateOp", **base_fields)
        else:
            return CodeGenerateOp

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

        op_class = self._create_code_generate_op(rate_input)

        response = await self._fill_node(op_class, prompt, mode, function_name="solve", rate_input=rate_input)
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
            code = code_response.get("response")
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
    
# class AnswerGenerateOp(BaseModel):
#     thought: str = Field(default="", description="The step by step thinking process")
#     answer: str = Field(default="", description="The final answer to the question")

class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


# class ReflectionTestOp(BaseModel):
#     reflection_and_solution: str = Field(
#         default="", description="Corrective solution for code execution errors or test case failures"
#     )

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

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

    async def __call__(self, problem, solution, entry_point, test_loop: int = 3):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]

        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}