from enum import Enum, auto


class State(Enum):
    STRUCTURAL = auto()
    FUNCTION_NAME = auto()
    PARAM_KEY = auto()
    PARAM_NUMBER = auto()
    PARAM_STRING = auto()


class StructuralPhase(Enum):
    OPENING = auto()
    AFTER_NAME = auto()
    PARAM_SEPARATOR = auto()
    PARAM_SEPARATOR_NO_COMMA = auto()
    VALUE_SEPARATOR = auto()
    CLOSING = auto()
