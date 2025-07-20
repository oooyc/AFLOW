# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field

class InputRatingOp(BaseModel):
    score: int = Field(..., description="A score from 1 to 10 evaluating the input quality.")
    justification: str = Field(..., description="A brief justification for the score.")


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The explanation of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

