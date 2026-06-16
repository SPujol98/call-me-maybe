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
        self.all_generated_ids: list[int] = []
        self.functions: list[FunctionDefinition] = functions
        self.function_prefix_map: dict[tuple[int, ...], set[int]] = (
            self._build_function_prefix_map(functions)
            )
        self.bool_prefix_map: dict[tuple[int, ...], set[int]] = (
            self._build_bool_prefix_map())
        self.current_prompt: str = ""
        self.param_number_count: int = 0
        self.param_string_token_count: int = 0
        self.current_function: FunctionDefinition | None = None
        self.current_param_index: int = 0
        self.vocab: dict[str, int] = self._load_vocab()
        self.quote_containing_tokens: set[int] = {
            tid for tok, tid in self.vocab.items()
            if '"' in tok and tok != '"'
        }
        self.structural_queue: deque[int] = deque()
        self.structural_phase: StructuralPhase = StructuralPhase.OPENING

    def start(self, prompt: str) -> None:
        self.state = State.STRUCTURAL
        self.current_prompt = prompt
        self.generated_ids = []
        self.all_generated_ids = []
        self.current_function = None
        self.current_param_index = 0
        self.param_number_count = 0
        self.param_string_token_count = 0
        self.structural_queue = deque()
        self.structural_phase = StructuralPhase.OPENING
        self._enqueue_strucutural(StructuralPhase.OPENING)

    def _add_to_prefix_map(self, prefix_map: dict[tuple[int, ...], set[int]],
                           text: str) -> None:
        ids = self.model.encode(text).tolist()[0]
        for i in range(len(ids)):
            prefix_map.setdefault(tuple(ids[:i]), set()).add(ids[i])

    def _build_function_prefix_map(
            self, n_funcs: list[FunctionDefinition]) -> dict[tuple[int, ...],
                                                             set[int]]:
        function_prefix_map: dict[tuple[int, ...], set[int]] = {}
        for fn in n_funcs:
            self._add_to_prefix_map(function_prefix_map, f'"{fn.name}"')
        return function_prefix_map

    def _build_bool_prefix_map(self) -> dict[tuple[int, ...], set[int]]:
        bool_prefix_map: dict[tuple[int, ...], set[int]] = {}
        self._add_to_prefix_map(bool_prefix_map, "true")
        self._add_to_prefix_map(bool_prefix_map, "false")
        return bool_prefix_map

    def _load_vocab(self) -> dict[str, int]:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            result: dict[str, int] = json.load(f)
            return result

    def update(self, token_id: int) -> None:
        self.generated_ids.append(token_id)
        self.all_generated_ids.append(token_id)

        if self.state == State.STRUCTURAL:
            if self.structural_queue and token_id == self.structural_queue[0]:
                self.structural_queue.popleft()
            if not self.structural_queue:
                self._transition_from_structural()
        elif self.state == State.FUNCTION_NAME:
            if token_id == self.vocab['"'] and len(self.generated_ids) > 1:
                self.state = State.STRUCTURAL
                quote_id = self.vocab['"']
                name_tokens: list[int] = []
                for i in range(len(self.generated_ids) - 2, -1, -1):
                    if self.generated_ids[i] == quote_id:
                        name_tokens = self.generated_ids[i+1:-1]
                        break
                id_to_token = {v: k for k, v in self.vocab.items()}
                name = "".join(id_to_token[i] for i in name_tokens)
                self.current_function = next(fn for fn in
                                             self.functions if fn.name == name)
                self._enqueue_strucutural(StructuralPhase.AFTER_NAME)
        elif self.state == State.PARAM_STRING:
            self.param_string_token_count += 1
            if token_id == self.vocab['"']:
                if self.current_function is None:
                    return
                self.param_string_token_count = 0
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                is_last = self.current_param_index >= len(
                    self.current_function.parameters)
                if self.current_function and is_last:
                    self.structural_phase = StructuralPhase.CLOSING_DOUBLE
                    self._enqueue_strucutural(StructuralPhase.CLOSING_DOUBLE)
                else:
                    self.structural_phase = StructuralPhase.PARAM_SEPARATOR
                    self._enqueue_strucutural(StructuralPhase.PARAM_SEPARATOR)
        elif self.state == State.PARAM_BOOL:
            if not self.bool_prefix_map.get(tuple(self.generated_ids), set()):
                if self.current_function is None:
                    return
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                is_last = self.current_param_index >= len(
                    self.current_function.parameters)
                if self.current_function and is_last:
                    self.structural_phase = StructuralPhase.CLOSING_DOUBLE
                    self._enqueue_strucutural(StructuralPhase.CLOSING_DOUBLE)
                else:
                    self.structural_phase = StructuralPhase.PARAM_SEPARATOR
                    self._enqueue_strucutural(StructuralPhase.PARAM_SEPARATOR)
        elif self.state == State.PARAM_NUMBER:
            if token_id == self.vocab['}']:
                self.param_number_count = 0
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                self.structural_phase = StructuralPhase.CLOSING
                self._enqueue_strucutural(StructuralPhase.CLOSING)
            elif token_id == self.vocab[',']:
                self.param_number_count = 0
                self.current_param_index += 1
                self.state = State.STRUCTURAL
                if self.current_function and (self.
                                              current_param_index
                                              >= len(self.
                                                     current_function.
                                                     parameters)):
                    self.structural_phase = StructuralPhase.CLOSING
                else:
                    self.structural_phase = (StructuralPhase.
                                             PARAM_SEPARATOR_NO_COMMA)
                    self._enqueue_strucutural(StructuralPhase.
                                              PARAM_SEPARATOR_NO_COMMA)
            else:
                self.param_number_count += 1

    def _transition_from_structural(self) -> None:
        if self.structural_phase == StructuralPhase.OPENING:
            self.generated_ids = []
            self.state = State.FUNCTION_NAME
            self.structural_phase = StructuralPhase.AFTER_NAME
        elif self.current_function is None:
            return
        elif self.structural_phase == StructuralPhase.AFTER_NAME:
            self.state = State.STRUCTURAL
            self.structural_phase = StructuralPhase.PARAM_SEPARATOR_NO_COMMA
            self._enqueue_strucutural(StructuralPhase.PARAM_SEPARATOR_NO_COMMA)
        elif self.structural_phase == StructuralPhase.PARAM_SEPARATOR:
            if self.current_function and (self.current_param_index
                                          >= len(self.
                                                 current_function.parameters)):
                self.state = State.STRUCTURAL
                self.structural_phase = StructuralPhase.CLOSING
            else:
                key_option = list(self.current_function.parameters.
                                  keys())[self.current_param_index]
                if (self.current_function.
                        parameters[key_option].data_type == "string"):
                    self.state = State.PARAM_STRING
                elif (self.current_function.parameters[key_option].
                      data_type == "boolean"):
                    self.generated_ids = []
                    self.state = State.PARAM_BOOL
                else:
                    self.state = State.PARAM_NUMBER
        elif self.structural_phase == StructuralPhase.PARAM_SEPARATOR_NO_COMMA:
            self.state = State.STRUCTURAL
            key_option = list(self.current_function.parameters.
                              keys())[self.current_param_index]
            if (self.current_function.
                    parameters[key_option].data_type == "string"):
                self.state = State.PARAM_STRING
            elif (self.current_function.parameters[key_option].
                    data_type == "boolean"):
                self.generated_ids = []
                self.state = State.PARAM_BOOL
            else:
                self.state = State.PARAM_NUMBER
        elif self.structural_phase == StructuralPhase.CLOSING:
            pass

    def _enqueue_strucutural(self, str_phase: StructuralPhase) -> None:
        if str_phase == StructuralPhase.OPENING:
            escaped = self.current_prompt.replace('"', '\\"')
            text = f'{{"prompt": "{escaped}", "name":'
            ids = self.model.encode(text).tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif self.current_function is None:
            return
        elif str_phase == StructuralPhase.AFTER_NAME:
            ids = self.model.encode(', "parameters": {').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.CLOSING:
            ids = self.model.encode('}').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.CLOSING_DOUBLE:
            ids = self.model.encode('}}').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.PARAM_SEPARATOR:
            separator = list(self.current_function.
                             parameters.keys())[self.current_param_index]
            param_type = self.current_function.parameters[separator].data_type
            if param_type == "string":
                ids = self.model.encode(f', "{separator}":"').tolist()[0]
            else:
                ids = self.model.encode(f', "{separator}":').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)
        elif str_phase == StructuralPhase.PARAM_SEPARATOR_NO_COMMA:
            separator = list(self.current_function.
                             parameters.keys())[self.current_param_index]
            param_type = self.current_function.parameters[separator].data_type
            if param_type == "string":
                ids = self.model.encode(f'"{separator}":"').tolist()[0]
            else:
                ids = self.model.encode(f'"{separator}":').tolist()[0]
            for id in ids:
                self.structural_queue.append(id)

    def get_valid_tokens(self) -> set[int]:
        if self.state == State.FUNCTION_NAME:
            return self.function_prefix_map.get(tuple(self.generated_ids),
                                                set())
        elif self.state == State.PARAM_NUMBER:
            if self.current_function is None:
                return set()
            valid: set[int] = set()
            is_last = (self.current_param_index + 1 >= len(
                self.current_function.parameters))
            exit_token = self.vocab['}'] if is_last else self.vocab[',']

            if self.param_number_count > 10:
                return {exit_token}

            for ch in "0123456789.-":
                valid.add(self.vocab[ch])
            valid.add(exit_token)
            return valid
        elif self.state == State.PARAM_STRING:
            if self.param_string_token_count > 50:
                return {self.vocab['"']}
            return set(self.vocab.values()) - self.quote_containing_tokens
        elif self.state == State.PARAM_BOOL:
            return self.bool_prefix_map.get(tuple(self.generated_ids),
                                            set())
        elif self.state == State.STRUCTURAL:
            if self.structural_queue:
                return {self.structural_queue[0]}
            return set()
        return set()

    def end_checker(self) -> bool:
        if ((self.structural_phase == StructuralPhase.CLOSING or
             self.structural_phase == StructuralPhase.CLOSING_DOUBLE)
                and not self.structural_queue):
            return True
        return False
