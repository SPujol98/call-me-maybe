from enum import Enum, auto


class State(Enum):
    STRUCTURAL = auto()
    FUNCTION_NAME = auto()
    PARAM_KEY = auto()
    PARAM_NUMBER = auto()
    PARAM_STRING = auto()
