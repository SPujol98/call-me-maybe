import json

from collections import deque
from src.models import FunctionDefinition
from src.state import State, StructuralPhase
from llm_sdk import Small_LLM_Model  # type: ignore


class Monitor:
    def __init__(self, functions: list[FunctionDefinition],
                 model: Small_LLM_Model) -> None:
        self.model = model
        self.state: State = State.STRUCTURAL
        self.generated_ids: list[int] = []
        self.functions: list[FunctionDefinition] = functions
        self.prefix_map: dict[tuple,
                              set[int]] = self._build_prefix_map(functions)
        self.current_function: FunctionDefinition | None = None
        self.current_param_index: int = 0
        self.vocab: dict[str, int] = self._load_vocab()
        self.structural_queue: deque[int] = deque()
        self.structural_phase: StructuralPhase = StructuralPhase.OPENING

    def start(self) -> None:
        self.state = State.STRUCTURAL
        self.generated_ids = []
        self.current_function = None
        self.current_param_index = 0
        self.structural_queue = deque()
        self.structural_phase = StructuralPhase.OPENING
        self._enqueue_strucutural(StructuralPhase.OPENING)

    def _add_to_prefix_map(self, prefix_map: dict[tuple, set[int]],
                           text: str) -> None:
        ids = self.model.encode(text).tolist()[0]
        for i in range(len(ids)):
            prefix_map.setdefault(tuple(ids[:i]), set()).add(ids[i])

    def _build_prefix_map(self, n_funcs:
                          list[FunctionDefinition]) -> dict[tuple, set[int]]:
        prefix_map: dict[tuple, set[int]] = {}
        for fn in n_funcs:
            self._add_to_prefix_map(prefix_map, f'"{fn.name}"')
            for key in fn.parameters.keys():
                self._add_to_prefix_map(prefix_map, f'"{key}"')
        return prefix_map

    def _load_vocab(self) -> dict[str, int]:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            return json.load(f)

    def update(self, token_id: int) -> None:
        self.generated_ids.append(token_id)

        if self.state == State.STRUCTURAL:
            if self.structural_queue and token_id == self.structural_queue[0]:
                self.structural_queue.popleft()
            if not self.structural_queue:
                self._transition_from_structural()
        elif self.state == State.FUNCTION_NAME:
            if token_id == self.vocab['"']:
                self.state = State.STRUCTURAL
                self._enqueue_strucutural(StructuralPhase.AFTER_NAME)
                quote_id = self.vocab['"']
                for i in range(len(self.generated_ids) - 2, -1, -1):
                    if self.generated_ids[i] == quote_id:
                        name_tokens = self.generated_ids[i+1:-1]
                        break
                id_to_token = {v: k for k, v in self.vocab.items()}
                name = "".join(id_to_token[i] for i in name_tokens)
                self.current_function = next(fn for fn in
                                             self.functions if fn.name == name)
        elif self.state == State.PARAM_KEY:
            if token_id == self.vocab['"']:
                self._enqueue_strucutural(StructuralPhase.VALUE_SEPARATOR)
        elif self.state == State.PARAM_STRING:
            if token_id == self.vocab['"']:
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                self._enqueue_strucutural(StructuralPhase.PARAM_SEPARATOR)
        elif self.state == State.PARAM_NUMBER:
            if token_id not in {self.vocab[ch] for ch in "0123456789.-"}:
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                self._enqueue_strucutural(StructuralPhase.PARAM_SEPARATOR)

    def _transition_from_structural(self) -> None:
        if self.structural_phase == StructuralPhase.OPENING:
            self.state = State.FUNCTION_NAME
            self.structural_phase = StructuralPhase.AFTER_NAME
        elif self.current_function is None:
            return
        elif self.structural_phase == StructuralPhase.AFTER_NAME:
            self.state = State.PARAM_KEY
            self.structural_phase = StructuralPhase.PARAM_SEPARATOR
        elif self.structural_phase == StructuralPhase.PARAM_SEPARATOR:
            if self.current_function and (self.current_param_index
                                          >= len(self.
                                                 current_function.parameters)):
                self.state = State.STRUCTURAL
                self.structural_phase = StructuralPhase.CLOSING
            else:
                self.state = State.PARAM_KEY
        elif self.structural_phase == StructuralPhase.VALUE_SEPARATOR:
            key_option = list(self.current_function.
                              parameters.keys())[self.current_param_index]
            if (self.current_function.
                    parameters[key_option].data_type == "string"):
                self.state = State.PARAM_STRING
            elif (self.current_function.
                  parameters[key_option].data_type == "number"):
                self.state = State.PARAM_NUMBER
        elif self.structural_phase == StructuralPhase.CLOSING:
            pass

    def _enqueue_strucutural(self, str_phase: StructuralPhase) -> None:
        if str_phase == StructuralPhase.OPENING:
            ids = self.model.encode('{"name": ').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif self.current_function is None:
            return
        elif str_phase == StructuralPhase.AFTER_NAME:
            ids = self.model.encode('", "parameters": {').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.CLOSING:
            ids = self.model.encode('}').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.PARAM_SEPARATOR:
            separator = list(self.current_function.
                             parameters.keys())[self.current_param_index]
            ids = self.model.encode(f', "{separator}":').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.VALUE_SEPARATOR:
            ids = self.model.encode(':').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)

    def get_valid_tokens(self) -> set[int]:
        if self.state == State.FUNCTION_NAME:
            return self.prefix_map.get(tuple(self.generated_ids), set())
        elif self.state == State.PARAM_KEY:
            return self.prefix_map.get(tuple(self.generated_ids), set())
        elif self.state == State.PARAM_NUMBER:
            valid = set()
            for ch in "0123456789.-":
                valid.add(self.vocab[ch])
            return valid
        elif self.state == State.PARAM_STRING:
            return set(self.vocab.values()) - {self.vocab['"']}
        elif self.state == State.STRUCTURAL:
            if self.structural_queue:
                return {self.structural_queue[0]}
            return set()
        return set()

    def end_checker(self) -> bool:
        if (self.structural_phase == StructuralPhase.CLOSING
                and not self.structural_queue):
            return True
        return False

    def test(self) -> None:
        print(self.current_function)
        print(self.current_param_index)
        print(self.structural_queue)
