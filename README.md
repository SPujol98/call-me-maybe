# Call Me Maybe — Constrained Decoding for LLM Function Calling

Force a small language model to generate valid JSON function calls — every time — by controlling which tokens it's allowed to produce at each step.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Score](https://img.shields.io/badge/Score-108%2F100-brightgreen)

![demo](assets/demo.gif)

---

## Overview

LLMs are bad at reliably producing structured output, especially small ones. A 0.6B model prompted to generate JSON might get it right 30% of the time — which is useless for any real pipeline.

Call Me Maybe takes a different approach: instead of prompting and hoping, it intercepts the model at every token generation step and mathematically forbids any token that would break the output structure. If the current field expects a number, only digits are allowed. If it expects a string, only non-quote tokens are valid. The model never gets the chance to hallucinate invalid output — structurally incorrect JSON is impossible by construction.

```
"What is the sum of 2 and 3?"
        ↓
{"prompt": "What is the sum of 2 and 3?", "name":"fn_add_numbers", "parameters": {"a":2,"b":3}}
```

---

## How It Works

The core is a **finite-state machine** (`Monitor`) that runs in lockstep with the model's token generation loop:

1. The model produces logits (raw scores) over its full vocabulary (~150k tokens)
2. `get_valid_tokens()` returns the set of tokens allowed at this exact point
3. `filter_logits()` sets all other tokens to `-inf` — impossible to sample
4. The highest-scoring remaining token wins (`argmax`)
5. `update()` records the token and advances the state machine

The monitor tracks two things simultaneously:

- **`State`** — what's being generated right now: `STRUCTURAL` (fixed scaffolding), `FUNCTION_NAME`, `PARAM_NUMBER`, `PARAM_STRING`, or `PARAM_BOOL`
- **`StructuralPhase`** — where in the JSON structure we are: `OPENING`, `AFTER_NAME`, `PARAM_SEPARATOR`, `PARAM_SEPARATOR_NO_COMMA`, `CLOSING`, or `CLOSING_DOUBLE`

### Structural forcing

Fixed JSON scaffolding (`{"prompt": "...", "name":`, `, "parameters": {`, closing braces) gets pre-encoded into a `deque` called `structural_queue`. While the queue is non-empty, the only valid token is `structural_queue[0]` — the model is forced through the structure token by token. Once the queue empties, the monitor transitions to the next phase.

### Function name selection

At startup, a prefix map is built for every known function name: each valid prefix maps to the set of tokens that can legally extend it. During `FUNCTION_NAME` state, only tokens in this map for the current prefix are allowed. The model can't hallucinate a function name — it's guided character by character toward one of the valid options.

### Parameter handling

- **Numbers** — only digits, `.`, `-`, and the appropriate exit token (`,` or `}`). After 10 digits, only the exit token is allowed.
- **Strings** — all tokens except those containing embedded `"` characters. After 50 tokens, only `"` is forced to close the string.
- **Booleans** — same prefix map mechanism as function names, built for `"true"` and `"false"`.

### Prompt embedding

Getting a 0.6B model to extract string values from a prompt is harder than it sounds. The fix was embedding the original prompt directly inside the JSON scaffold being constructed:

```
{"prompt": "Greet shrek", "name":
```

When the model reaches `"name":` in the parameters, "shrek" is only a few tokens back in context — making it a near-trivial copy task instead of a reasoning problem.

---

## Design Decisions

**State machine over grammar** — a grammar-based approach would be more general but harder to control precisely for this use case. The FSM gives direct access to exactly which tokens are valid at each transition, which is what constrained decoding needs.

**Prefix maps over character matching** — Qwen3 uses BPE tokenization, where a single token can encode multiple characters ("shrek" might be one token). Character-by-character matching breaks this. Prefix maps operate at the token level and handle multi-character tokens naturally.

**Forced structure, free values** — all JSON keys, separators, and braces go through the `structural_queue`. The model only makes free choices for function name (constrained by prefix map), parameter values (constrained by type), and string content. This eliminates the vast majority of failure modes upfront.

**Closing asymmetry** — `PARAM_NUMBER` exits when the model generates `}` (closing the parameters object), so only one more `}` is needed. `PARAM_STRING` and `PARAM_BOOL` exit without generating any `}`, so two are needed. This required `CLOSING_DOUBLE` as a distinct structural phase.

**Single-token skip** — when `get_valid_tokens()` returns exactly one token, the forward pass is skipped entirely and that token is forced directly. This saves ~60-75% of model calls on structural tokens, which are the majority.

---

## Bugs Worth Mentioning

**Quote detection in BPE** — early versions checked `token_id == vocab['"']` to detect string closing. This broke for BPE tokens that encode a quote alongside other characters (e.g. a token decoding to `."` or `x"`). Fixed by precomputing `invalid_string_tokens` — all token IDs whose decoded string contains `"` but isn't exactly `"` — and excluding them from `PARAM_STRING`'s valid set.

**Key duplication from PARAM_KEY** — an early design had a `PARAM_KEY` state where the model chose which parameter key to generate. This caused duplicate keys (`{"b": 2, "b": 3}` instead of `{"a": 2, "b": 3}`). Since keys are fully determined by the function definition and `current_param_index`, the fix was eliminating `PARAM_KEY` entirely and encoding keys directly into `PARAM_SEPARATOR` and `PARAM_SEPARATOR_NO_COMMA`.

---

## Performance

| Metric | Result |
|:---|:---|
| JSON validity | 100% — invalid output is structurally impossible |
| Function selection | Deterministic — hallucinated names can't be generated |
| Model calls saved | ~60-75% via single-token skip on structural tokens |
| CPU latency | ~20-30s for numeric prompts, up to several minutes for long strings |

---

## Installation & Usage

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv)
- Model weights download automatically on first run (~1.5 GB)

### Install

```bash
make install
```

### Run

```bash
make run
```

Processes `data/input/function_calling_tests.json` against `data/input/functions_definition.json` and writes results to `data/output/function_calls.json`.

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

---

## Project Structure

```
CallMeMaybe/
├── Makefile
├── pyproject.toml
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── llm_sdk/                    # Provided SDK (not authored)
└── src/
    ├── __main__.py             # Entry point and generation loop
    ├── models.py               # Pydantic models
    ├── json_parser.py          # Input file loaders with Pydantic validation
    ├── logits_processor.py     # filter_logits — token masking
    ├── monitor.py              # State machine — core of the pipeline
    └── state.py                # State and StructuralPhase enums
```

---

## Input / Output Format

**`functions_definition.json`**
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

**`function_calling_tests.json`**
```json
[{ "prompt": "Greet shrek" }]
```

**Output**
```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

---

## Resources

- [Constrained Beam Search — Hugging Face](https://huggingface.co/blog/constrained-beam-search)
- [Outlines — Structured Text Generation](https://github.com/outlines-dev/outlines)
- [Qwen3-0.6B — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte Pair Encoding — Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding)
