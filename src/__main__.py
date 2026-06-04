import argparse
import numpy as np
import json
import os

from src.json_parser import load_function_definitions, load_input_prompts
from src.monitor import Monitor
from src.logits_processor import filter_logits
from src.models import FunctionCall
from llm_sdk import Small_LLM_Model


def main() -> None:
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
        print(f"Processing: {prompt.prompt}")
        monitor.start()
        input_ids = model.encode(prompt.prompt).tolist()[0]
        while not monitor.end_checker():
            logits = model.get_logits_from_input_ids(input_ids)
            valid = monitor.get_valid_tokens()
            filtered = filter_logits(logits, valid)
            token_id = int(np.argmax(filtered))
            input_ids.append(token_id)
            monitor.update(token_id)
        result_json = json.loads(model.decode(monitor.generated_ids))
        result_list.append(FunctionCall(
            prompt=prompt.prompt,
            name=result_json["name"],
            parameters=result_json["parameters"]))
        print(f"Done: {result_json['name']}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps([fc.model_dump() for fc in result_list], indent=2))


if __name__ == "__main__":
    main()
