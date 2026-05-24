import json

from src.models import FunctionDefinition, InputPrompt
from pydantic import ValidationError


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    func_list: list[FunctionDefinition] = []
    try:
        with open(path) as f:
            data = json.load(f)
            for d in data:
                func_list.append(FunctionDefinition.model_validate(d))
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
    except ValidationError as e:
        print(f"Error: {e}")
    return func_list


def load_input_prompts(path: str) -> list[InputPrompt]:
    input_list: list[InputPrompt] = []
    try:
        with open(path) as f:
            data = json.load(f)
            for d in data:
                input_list.append(InputPrompt.model_validate(d))
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
    except ValidationError as e:
        print(f"Error: {e}")
    return input_list
