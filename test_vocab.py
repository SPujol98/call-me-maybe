import json

from llm_sdk import Small_LLM_Model
from src.monitor import Monitor
from src.json_parser import load_function_definitions


def prueba() -> None:
    model = Small_LLM_Model()
    functions = load_function_definitions("data/input/"
                                          "functions_definition.json")
    monitor = Monitor(functions, model)
    monitor.test()
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    ids = model.encode('{"name": ').tolist()[0]
    print(f"After quote: {monitor.function_prefix_map.get((1,), set())}")
    id_to_tok = {v: k for k, v in monitor.vocab.items()}
    tokens = {id_to_tok.get(i, '?') for i in monitor.function_prefix_map.get((1,), set())}
    print(f"After quote tokens: {tokens}")
    '''
    vocab_path = model.get_path_to_vocab_file()

    functions = ['"fn_add_numbers"',
                 '"fn_greet"',
                 '"fn_reverse_string"',
                 '"fn_get_square_root"',
                 '"fn_substitute_string_with_regex"'
                 ]

    # build_prefix_map(functions)
    with open(vocab_path) as f:
        vocab = json.load(f)
    print(vocab)



    id_to_token = {v: k for k, v in vocab.items()}
    for fn in functions:
        ids = model.encode(fn).tolist()[0]
        tokens = [id_to_token[i] for i in ids]
        print(f"{fn} -> {tokens}")


    for id in [8822, 2891, 32964, 1889, 3744, 43277, 3904]:
        print(f"{id} -> {repr(id_to_token.get(id, 'NOT FOUND'))}")


def build_prefix_map(n_funcs: list[str]) -> dict[tuple, str]:

    prefix_map: dict[tuple, str] = {}
    model = Small_LLM_Model()
    for fn in n_funcs:
        ids = model.encode(fn).tolist()[0]
        for i in range(len(ids)):
            prefix_map.setdefault(tuple(ids[:i]), set()).add(ids[i])
    print(prefix_map)
    return prefix_map
'''

if __name__ == "__main__":
    prueba()
