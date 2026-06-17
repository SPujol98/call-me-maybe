import json
import sys
import os

from src.models import FunctionDefinition, InputPrompt
from pydantic import ValidationError


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    if not os.path.exists(path):
        print(f"Error: functions definition file not found: {path}",
              file=sys.stderr)
        sys.exit(1)
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"Error: {path} must contain a JSON array", file=sys.stderr)
        sys.exit(1)
    return [FunctionDefinition(**item) for item in data]


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
