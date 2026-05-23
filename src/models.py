from pydantic import BaseModel, Field
from enum import Enum


class FunctionParameter(BaseModel):
    type: str = Field(min_length=1)

class
