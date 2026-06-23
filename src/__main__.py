import argparse
import numpy as np
import json
import os

from src.json_parser import load_function_definitions, load_input_prompts
from src.monitor import Monitor
from src.logits_processor import filter_logits
from src.models import FunctionCall, FunctionDefinition
from llm_sdk import Small_LLM_Model  # type: ignore


def build_prompt(functions: list[FunctionDefinition]) -> str:
    """
    Build the system coontext listing available functions and
    single few-shot example of the target JSON output format.
    """
    context = "You are a function calling engine. Available functions:\n"
    for fn in functions:
        context += f"- {fn.name}: {fn.description}\n"
    context += '\nExample:\n'
    context += ('{"prompt": "Greet alice", "name": "fn_greet",'
                ' "parameters": {"name": "alice"}}\n')
    return context


def cast_parameters(parameters: dict[str, object],
                    function: FunctionDefinition) -> dict[str, object]:
    """
    Cast values declared as "number" to float.
    """
    for key, value in parameters.items():
        if key not in function.parameters:
            continue
        param_type = function.parameters[key].data_type
        if param_type == "number":
            parameters[key] = float(value)  # type: ignore
    return parameters


def main() -> None:
    """
    Run the constrained decoding pipeline: for each prompt, generate a
    a function call token by token, skipping the model whenever the monitor
    already knows the only valid next token.
    """
    arg_parser = argparse.ArgumentParser(description="Call Me Maybe: A "
                                         "constrained decoding pipeline for "
                                         "strict function calling "
                                         "JSON generation.")
    arg_parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json",
                            help="Path to the JSON file with "
                            "function definitions")
    arg_parser.add_argument("--input",
                            default="data/input/function_calling_tests.json",
                            help="Path to the JSON file with test prompts")
    arg_parser.add_argument("--output",
                            default="data/output/function_calls.json",
                            help="Path to the output JSON file")
    args = arg_parser.parse_args()
    f_definitions = load_function_definitions(args.functions_definition)
    i_prompts = load_input_prompts(args.input)
    model = Small_LLM_Model()
    monitor = Monitor(f_definitions, model)
    result_list: list[FunctionCall] = []
    for prompt in i_prompts:
        monitor.start(prompt.prompt)
        full_prompt = build_prompt(f_definitions)
        input_ids = model.encode(full_prompt).tolist()[0]

        while not monitor.end_checker():
            valid = monitor.get_valid_tokens()
            if len(valid) == 1:
                token_id = next(iter(valid))
            else:
                input_ids = input_ids[-512:]
                logits = model.get_logits_from_input_ids(input_ids)
                filtered = filter_logits(logits, valid)
                token_id = int(np.argmax(filtered))
            input_ids.append(token_id)
            monitor.update(token_id)
            word = model.decode([token_id])
            print(word, end="", flush=True)
        print()
        decoded = model.decode(monitor.all_generated_ids)
        try:
            result_json = json.loads(decoded)
        except json.JSONDecodeError:
            continue
        current_fn = next(fn for fn in f_definitions
                          if fn.name == result_json["name"])
        params = cast_parameters(result_json["parameters"], current_fn)
        result_list.append(FunctionCall(
            prompt=prompt.prompt,
            name=result_json["name"],
            parameters=params))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps([fc.model_dump() for fc in result_list], indent=2))


if __name__ == "__main__":
    main()
