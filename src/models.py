from pydantic import BaseModel, Field
from typing import Any, Optional


class ParameterType(BaseModel):
    data_type: str = Field(min_length=1, alias="type")


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterType]
    returns: Optional[ParameterType] = None


class InputPrompt(BaseModel):
    prompt: str = Field(min_length=1)


class FunctionCall(BaseModel):
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any]
