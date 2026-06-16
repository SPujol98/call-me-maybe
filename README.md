*This project has been created as part of the 42 curriculum by spujol-s.*

---

# 📞 Call Me Maybe — Constrained Decoding for LLM Function Calling

> Force a small language model to generate valid JSON function calls — every single time — by hijacking its token selection process at each step.

---

## 📋 Description

**Call Me Maybe** is a constrained decoding pipeline built in Python that wraps a small LLM (Qwen3-0.6B) and guarantees 100% valid JSON function-call output, regardless of what the model would naturally generate.

The idea is straightforward: instead of trusting the model to produce correct JSON, you intercept it at every token generation step and restrict its choices to only those tokens that keep the output structurally valid and schema-compliant. If a field expects a number, the model can only generate digits. If a field expects a string, it generates characters — but only until it closes the string with a quote. The model never gets the chance to hallucinate invalid output.

The pipeline reads a set of function definitions (name, description, parameter types) and a list of natural-language prompts. For each prompt it generates a JSON object of the form:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": {
    "name": "shrek"
  }
}
```

Every generated JSON is guaranteed to parse without errors.

---

## ⚙️ Instructions

### Requirements

- Python 3.12+
- `uv` (package manager)
- The model weights are downloaded automatically on first run (~1.5 GB)

### Installation

```bash
make install
```

### Run

```bash
make run
```

This processes `data/input/function_calling_tests.json` using the function definitions in `data/input/functions_definition.json` and writes results to `data/output/function_calls.json`.

### Custom paths

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/output.json
```

### Lint

```bash
make lint         # flake8 + mypy
make lint-strict  # flake8 + mypy --strict
```

### Clean

```bash
make clean
```

---

## 🗂️ Project Structure

```
CallMeMaybe/
├── Makefile
├── README.md
├── pyproject.toml
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── llm_sdk/               # Provided SDK (not authored)
└── src/
    ├── __main__.py        # Entry point and generation loop
    ├── models.py          # Pydantic models (FunctionDefinition, FunctionCall...)
    ├── json_parser.py     # Input file loaders
    ├── logits_processor.py # Token masking (filter_logits)
    ├── monitor.py         # State machine — the core of the project
    └── state.py           # State and phase enums
```

---

## 🧠 Algorithm Explanation

The core of the project is a **finite-state machine** implemented in `Monitor` that runs in lockstep with the model's token generation loop.

At each step:

1. The model produces logits (raw scores) over its entire vocabulary (~150k tokens).
2. The monitor's `get_valid_tokens()` returns the set of tokens that are allowed at this exact point in the output.
3. `filter_logits()` sets all other tokens to `-inf`, making them impossible to sample.
4. The highest-scoring remaining token is selected (`argmax`).
5. The monitor's `update()` records the token and advances its internal state.

The monitor tracks two orthogonal dimensions of state:

- **`State`** — what kind of content is currently being generated: `STRUCTURAL` (fixed scaffolding), `FUNCTION_NAME`, `PARAM_NUMBER`, `PARAM_STRING`, or `PARAM_BOOL`.
- **`StructuralPhase`** — which part of the JSON structure we're in: `OPENING`, `AFTER_NAME`, `PARAM_SEPARATOR`, `PARAM_SEPARATOR_NO_COMMA`, `CLOSING`, or `CLOSING_DOUBLE`.

### Structural forcing via a queue

Fixed JSON tokens (like `{"prompt": "...", "name":` or `, "parameters": {`) are pre-encoded into a `deque` called `structural_queue`. While the queue is non-empty, the only valid token is `structural_queue[0]` — the model is forced to generate exactly those tokens in sequence, token by token. Once the queue empties, the monitor transitions to the next phase.

### Function name selection

A prefix map is built at startup: for each function name, every valid prefix is stored as a key mapping to the set of tokens that can legally extend it. During `FUNCTION_NAME` state, only tokens present in this map for the current prefix are allowed — the model can only produce valid function names, and it terminates the name when it generates the closing `"`.

### Parameter value handling

- **Numbers**: valid tokens are digits `0-9`, `.`, `-`, and the appropriate exit token (`,` or `}`). After 10 digits, only the exit token is allowed to prevent infinite generation.
- **Strings**: all vocabulary tokens are valid except those containing embedded `"` characters (which would prematurely close the string). After 50 tokens, only `"` is forced to close the string.
- **Booleans**: a separate prefix map for `"true"` and `"false"` — same mechanism as function names. Generation ends when no further prefix extension is possible.

### Prompt embedding for string extraction

A key insight is that the model must be able to "copy" values from the prompt into the parameters. To make this easy, the structural opening embeds the original prompt directly inside the JSON being generated:

```
{"prompt": "Greet shrek", "name":
```

This means when the model reaches `"name":` in the parameters, the string "shrek" is only a few tokens back in context — making it trivial to copy.

---

## 🏗️ Design Decisions

### State machine over regex or grammar

A grammar-based approach (e.g., a JSON grammar with schema constraints) would be more general but much harder to implement efficiently for this use case. The state machine is simpler and gives direct control over exactly which tokens are valid at each transition.

### Prefix maps instead of character-by-character matching

The tokenizer uses BPE (Byte Pair Encoding), which means a single token can encode multiple characters (e.g., `"shrek"` might be one token). Matching character by character would break this. Prefix maps work at the token level: they track which tokens can legally extend the generated sequence toward a valid target string.

### Forcing structural tokens vs. letting the model choose

All JSON structure (keys, separators, braces) is forced through the `structural_queue`. The model only makes free choices for: function name (constrained by prefix map), parameter values (constrained by type), and string content. This eliminates the vast majority of possible failure modes.

### Closing logic for different parameter types

- `PARAM_NUMBER`: the model itself generates `}` or `,` to close the value — one structural `}` is then enqueued to close the outer object.
- `PARAM_STRING` / `PARAM_BOOL`: the model generates `"` / completes `true`/`false` without generating any `}` — two `}}` are enqueued to close both the parameters object and the root object.

### Token sliding window

To avoid memory issues on long prompts, `input_ids` is trimmed to the last 512 tokens before each forward pass. This keeps memory bounded while retaining enough context for accurate predictions.

---

## 📊 Performance Analysis

### Accuracy

All 11 test prompts produce valid, parseable JSON with correct function selection and parameter extraction. The constrained decoding guarantee is absolute: structurally invalid output is impossible by construction.

Semantic accuracy (whether the extracted values are correct) depends on the model's ability to copy or infer values from the prompt. With the prompt embedded in the JSON scaffold, extraction accuracy on simple prompts is high. For complex regex parameters, the model approximates rather than computing the exact regex — this is a model capability limitation, not a pipeline one.

### Speed

Each token requires one full forward pass of the model (~1-2 seconds on CPU). For a prompt with simple numeric parameters (2-3 tokens per parameter) this is fast (~20-30 seconds total). For string-heavy prompts where the model generates up to 50 tokens before the safety limit kicks in, a single prompt can take several minutes on CPU. A GPU would reduce this by 10-50x.

### Reliability

The `end_checker()` method guarantees the loop terminates: every state has a defined exit condition enforced by the constrained token set, and safety limits (`> 10` for numbers, `> 50` for strings) provide a hard upper bound on generation length.

---

## 🧩 Challenges Faced

### Multi-character tokens and quote detection

Early implementations checked `token_id == vocab['"']` to detect string closing. This broke for BPE tokens that encode a quote alongside other characters (e.g., a token that decodes to `."` or `x"`). The fix was precomputing `quote_containing_tokens` — the set of all token IDs whose string representation contains `"` but is not exactly `"` — and excluding them from `PARAM_STRING`'s valid set.

### Model hallucinating instead of extracting

The biggest challenge was getting the 0.6B model to extract string values from the prompt rather than generating generic content. Initial prompts produced template-like output (`{type: str}`, markdown blocks). The breakthrough was embedding the user prompt directly inside the JSON being constructed — placing "shrek" immediately before `"name":` in the parameters gives the model an almost trivial copy task rather than a hard reasoning one.

### Double closing braces

`PARAM_NUMBER` exits when the model generates `}` (closing the parameters object), so only one more `}` is needed. `PARAM_STRING` and `PARAM_BOOL` exit without generating any `}`, so two are needed. Getting this asymmetry right required adding `CLOSING_DOUBLE` as a distinct structural phase.

### PARAM_KEY state causing key duplication

An early design had a `PARAM_KEY` state where the model chose which parameter key to generate. This caused the model to sometimes pick the wrong key (generating `{"b": 2, "b": 3}` instead of `{"a": 2, "b": 3}`). Since keys are fully determined by the function definition and `current_param_index`, the right fix was eliminating `PARAM_KEY` entirely and encoding the key directly into `PARAM_SEPARATOR` and `PARAM_SEPARATOR_NO_COMMA`.

---

## 🧪 Testing Strategy

Testing was done iteratively, expanding coverage one prompt type at a time:

1. **Numeric parameters** (`fn_add_numbers`): verified correct function selection and integer extraction. Confirmed `{"a": 2, "b": 3}` and `{"a": 265, "b": 345}` parse correctly.
2. **Single string parameter** (`fn_greet`): verified string extraction from prompt context. This was the hardest to get right and drove the prompt embedding design decision.
3. **Multi-string parameters** (`fn_reverse_string`, `fn_substitute_string_with_regex`): verified that 3-parameter functions with all-string signatures work correctly.
4. **Mixed types** (`fn_get_square_root`): single numeric parameter, verified float-capable output.
5. **Boolean and float** (added test functions): added temporary `fn_test_bool` and `fn_test_float` to `functions_definition.json` with prompts like "Test with flag true" — confirmed `true` without quotes and `3.14` as float.

Debug instrumentation (state and count prints) was added and removed throughout to track the monitor's transitions in real time without overwhelming output volume.

---

## 💡 Example Usage

### Input (`functions_definition.json`)

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

### Input (`function_calling_tests.json`)

```json
[
  { "prompt": "Greet shrek" }
]
```

### Run

```bash
make run
```

### Output (`data/output/function_calls.json`)

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  }
]
```

---

## 📚 Resources

### References

- [Hugging Face — Constrained Beam Search](https://huggingface.co/blog/constrained-beam-search)
- [Outlines — Structured Text Generation](https://github.com/outlines-dev/outlines)
- [Qwen3 Model Card — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte Pair Encoding — Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [JSON specification — ECMA-404](https://www.ecma-international.org/publications-and-standards/standards/ecma-404/)
- [Logits and Softmax in LLMs — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)

### AI Usage

Generative AI was used throughout this project as a Socratic mentor rather than a code generator — it asked questions and pushed back, I wrote the code:

- **Architecture design**: Discussed the state machine approach, the tradeoffs between grammar-based and FSM-based constrained decoding, and the decision to eliminate `PARAM_KEY` in favor of fully structural key forcing.
- **Debugging token issues**: Helped reason through why `token_id == vocab['"']` was insufficient for BPE tokenizers and identified the precomputed `quote_containing_tokens` set as the correct fix.
- **Prompt engineering**: Helped identify why the model hallucinated on string parameters and validated the approach of embedding the user prompt directly in the JSON scaffold to make value extraction a near-trivial copy task.
- **Type checking**: Helped resolve `mypy --strict` errors related to generic types (`tuple[int, ...]`, `set[int]`) and `None`-guards for optional attributes.
