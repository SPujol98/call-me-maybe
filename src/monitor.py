import json

from collections import deque
from src.models import FunctionDefinition
from src.state import State
from llm_sdk import Small_LLM_Model


class Monitor:
    def __init__(self, functions: list[FunctionDefinition],
                 model: Small_LLM_Model) -> None:
        self.model = model
        self.state: State = State.STRUCTURAL
        self.generated_ids: list[int] = []
        self.prefix_map: dict[tuple,
                              set[int]] = self._build_prefix_map(functions)
        self.current_function: FunctionDefinition | None = None
        self.current_param_index: int = 0
        self.vocab: dict[str, int] = self._load_vocab()

    def _build_prefix_map(self, n_funcs:
                          list[FunctionDefinition]) -> dict[tuple, set[int]]:
        prefix_map: dict[tuple, set[int]] = {}
        for fn in n_funcs:
            ids = self.model.encode(f'"{fn.name}"').tolist()[0]
            for i in range(len(ids)):
                prefix_map.setdefault(tuple(ids[:i]), set()).add(ids[i])
        return prefix_map
    
    def _load_vocab(self) -> dict[str, int]:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            return json.load(f)

    def update(self, token_id: int) -> None:
        self.generated_ids.append(token_id)

    def get_valid_tokens(self) -> set[int]:
        if self.state == State.FUNCTION_NAME:
            return self.prefix_map.get(tuple(self.generated_ids), set())
        elif self.state == State.PARAM_KEY:
        elif self.state == State.PARAM_NUMBER:
            valid = set()
            for ch in "0123456789.-":
                valid.add(self.vocab[ch])
            return valid
        elif self.state == State.PARAM_STRING:
            return set(self.vocab.values()) - {self.vocab['"']}
        elif self.state == State.STRUCTURAL:
        