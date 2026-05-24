import argparse
from src.json_parser import load_function_definitions, load_input_prompts


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

    print(f_definitions)
    print(i_prompts)


if __name__ == "__main__":
    main()
