import json

from llm_sdk import Small_LLM_Model


def prueba() -> None:
    model = Small_LLM_Model()
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
    print(type(vocab))


'''
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
