import json
import sys
import os

from src.models import FunctionDefinition, InputPrompt
from pydantic import ValidationError


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON.
    """
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
    try:
        return [FunctionDefinition(**item) for item in data]
    except ValidationError as e:
        print(f"Error: invalid function definition schema: {e}",
              file=sys.stderr)
        sys.exit(1)


def load_input_prompts(path: str) -> list[InputPrompt]:
    """
    Load and validate test prompts from a JSON.
    """
    if not os.path.exists(path):
        print(f"Error: input prompts file not found: {path}",
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
    try:
        return [InputPrompt(**item) for item in data]
    except ValidationError as e:
        print(f"Error: invalid input prompt schema: {e}",
              file=sys.stderr)
        sys.exit(1)
