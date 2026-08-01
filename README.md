# lfm-tools — Local Tool-Calling LLM Server (LiquidAI LFM2 + MCP)

An OpenAI-compatible `/v1/chat/completions` API, backed by a **local**
LiquidAI LFM2 GGUF model running via `llama-cpp-python`, that calls tools
exposed by a local **MCP (Model Context Protocol)** server. Runs fully
offline — no cloud LLM calls.

> **If you are an AI coding agent working in this repo:** read this file in
> full before making changes, and also read `AGENT.md` for terse
> do/don't rules. Section "Known Gotchas" below documents several
> non-obvious constraints discovered through real debugging — breaking any
> of them silently reintroduces bugs that took multiple iterations to fix.

---

## Architecture

```text
┌──────────────────────┐
│   client_example.py  │  OpenAIClient + ToolCallingAgent
│  (reference client)  │  drives the multi-turn tool-calling loop
└──────────┬────────────┘
           │ HTTP (OpenAI-compatible JSON)
           ▼
┌──────────────────────────────────────────┐
│   openai_api_server_local_llm.py          │
│   FastAPI server                          │
│                                            │
│   LocalLLMManager                         │
│     - loads LFM2-1.2B-Tool GGUF via       │
│       llama-cpp-python                    │
│     - builds the RAW prompt manually      │
│       (LFM2's literal special-token       │
│       format — see below)                 │
│     - parses tool calls out of raw text   │
│                                            │
│   MCPClient                               │
│     - spawns mcp_server.py as a subprocess│
│     - stdio JSON-RPC per the MCP protocol │
└──────────┬─────────────────────────────────┘
           │ MCP stdio protocol
           ▼
┌──────────────────────┐
│    mcp_server.py      │  Tool implementations:
│   MCP server           │   - get_weather (simulated)
│  (stdio subprocess)    │   - calculate (safe eval)
└──────────────────────┘   - get_time (pytz)
```

**Key design point:** the FastAPI server does *not* delegate tool-call
formatting to `llama-cpp-python`'s chat-template auto-detection. It builds
the exact LFM2 prompt string by hand (`build_raw_prompt`) and calls
`Llama.create_completion()` directly, not `create_chat_completion()`. This
was a deliberate fix after repeated format mismatches — see Known Gotchas.

---

## Repository layout

| File | Role |
|---|---|
| `openai_api_server_local_llm.py` | FastAPI server: OpenAI-compatible endpoint, LLM prompt building/parsing, MCP client wrapper |
| `mcp_server.py` | MCP server exposing 3 tools over stdio: `get_weather`, `calculate`, `get_time` |
| `client_example.py` | Reference client implementing the full tool-calling loop (`ToolCallingAgent`) against the server |
| `pyproject.toml` | Dependencies (managed with `uv` or `pip`) |
| `AGENT.md` | Terse directive rules for AI coding agents working in this repo |
| `models/` | **Not tracked in git** — GGUF model files go here (see Setup) |

---

## Setup from scratch

### 1. Prerequisites

- Python ≥ 3.12
- Windows/Linux/macOS (developed/tested primarily on Windows 10)
- Less than 0.7 GB disk for the model file

### 2. Install dependencies

```bash
# using uv (recommended)
uv sync

# or plain pip
pip install -e .
```

`pyproject.toml` dependencies (as currently pinned):

```toml
dependencies = [
    "accelerate>=1.14.0",
    "fastapi>=0.139.2",
    "gunicorn>=26.0.0",
    "llama-cpp-python>=0.3.34",
    "mcp>=1.28.1,<2.0.0",   # IMPORTANT: see Known Gotchas — mcp 2.0+ breaks this code
    "pydantic>=2.13.4",
    "pytz>=2026.2",
    "requests>=2.34.2",
    "torch>=2.13.0",
    "transformers>=5.14.1",
    "uvicorn[standard]>=0.51.0",
]
```

### 3. Download the model

This project uses **`LiquidAI/LFM2-1.2B-Tool`** (GGUF build), a model
purpose-built by Liquid AI for concise, precise tool calling — chosen over
smaller general-purpose LFM2.5 variants after they proved unreliable at
compound/multi-tool queries.

Download a GGUF quantization from
[`LiquidAI/LFM2-1.2B-Tool-GGUF`](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF)
into a local `models/` folder:

```text
models/LFM2-1.2B-Tool-Q4_K_M.gguf
```

`Q4_K_M` has been empirically verified to produce correct tool-call
formatting and argument extraction on all test queries below, at roughly
1/4 the size of `F16`. `F16` also works if you have the disk/RAM budget and
want maximum accuracy headroom. Update the `model_path` in the `lifespan`
handler (`openai_api_server_local_llm.py`) to match whichever file you
download.

### 4. Run

Two options, both start the MCP server automatically as a subprocess —
you do not need to run `mcp_server.py` separately:

```bash
# Terminal 1: start the API server (spawns mcp_server.py internally)
python openai_api_server_local_llm.py

# Terminal 2: run the reference client examples
python client_example.py
```

Server listens on `http://localhost:8000`.

### 5. Sanity check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/tools
```

`health` should report `"model_loaded": true` and `"mcp_connected": true`
before sending chat requests.

---

## API

### `POST /v1/chat/completions`

OpenAI-compatible. Standard `messages` array (`role`: `user` / `assistant`
/ `tool`), optional `tools` (falls back to the MCP server's tool list if
omitted). Returns `tool_calls` in the assistant message when the model
decides a tool is needed — the caller is responsible for executing them
and appending `role: "tool"` result messages, then calling again (see
`ToolCallingAgent.run()` in `client_example.py` for the reference loop).

> **Note:** `stream: true` is accepted but not actually implemented as
> token streaming — the server always blocks until generation completes,
> then returns one full message. Treat `stream` as a no-op for now.

### `GET /v1/tools`

Lists tools discovered from the MCP server (name, description, JSON Schema
parameters).

### `POST /v1/tools/execute`

Directly executes one tool via MCP: `{"name": "...", "arguments": {...}}`.
Used by `client_example.py`'s `execute_tool_call()` — in a production
setup you'd likely have the client (or an agent) talk to MCP directly
instead of proxying through this endpoint.

### `GET /health`, `GET /v1/models`, `GET /`

Status/introspection endpoints.

---

## The LFM2 tool-calling format (critical to understand before editing prompt logic)

LFM2-1.2B-Tool was fine-tuned on a **specific literal prompt format**
using special tokens. `build_raw_prompt()` reproduces it exactly:

```text
<|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_weather", "description": "...", "parameters": {...}}, ...]<|tool_list_end|><|im_end|>
<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_weather(location="Paris, France")]<|tool_call_end|><|im_end|>
<|im_start|>tool
<|tool_response_start|>{"location": "Paris, France", "temperature": 22, ...}<|tool_response_end|><|im_end|>
<|im_start|>assistant
```

Key facts:

- Tool calls are **Pythonic**: `[func(arg="value"), func2(arg="value")]` — a
  Python-style list, possibly with **multiple calls in one turn** (this is
  how compound queries like "weather in Paris and time there" are handled
  — the model emits both calls in a single bracket, not across separate
  turns).
- `<|tool_call_start|>`/`<|tool_call_end|>` are real vocabulary tokens.
  **`llama-cpp-python` strips special tokens from detokenized text by
  default**, so they usually will *not* appear literally in
  `create_completion()`'s output text — `parse_tool_calls_from_content()`
  treats the wrapper tokens as optional for this reason. Don't "fix" the
  parser to require them; that reintroduces a bug that silently broke tool
  detection entirely.
- Recommended decoding is **greedy, `temperature=0`** — this is hardcoded
  in `generate_response()`, not exposed as a request parameter, per
  Liquid AI's own guidance for this model.
- Tool results are re-injected as `role: "tool"` with content wrapped in
  `<|tool_response_start|>...<|tool_response_end|>` — the raw JSON string
  from MCP goes in as-is, **do not `json.dumps()` it again** (it's already
  a JSON string coming out of `MCPClient.call_tool()`; double-encoding was
  a real bug hit during development).

---

## Known Gotchas (read before "fixing" something that looks wrong)

1. **`mcp` package must stay below 2.0.0.** `mcp>=2.0.0` is a breaking
   rewrite that removes the `@app.list_tools()` / `@app.call_tool()`
   decorator API `mcp_server.py` uses. If `pip install mcp` resolves to
   2.x, the MCP subprocess crashes silently on startup and the API server
   just reports `Connection closed` with no other diagnostic. Keep the
   `<2.0.0` pin in `pyproject.toml`.

2. **Don't add `chat_format=` when constructing `Llama(...)`.** An earlier
   version hardcoded `chat_format="chatml"`, which conflicted with LFM2's
   own tool-calling conventions and caused the model to echo placeholder
   text instead of real tool calls. The current code deliberately omits
   `chat_format` and bypasses chat-template handling entirely by using
   `create_completion()` with a hand-built prompt instead of
   `create_chat_completion()`.

3. **No leading `<|startoftext|>` in the manually built prompt.**
   `llama-cpp-python`'s `create_completion()` adds BOS itself by default;
   including it again in `build_raw_prompt()` triggers a
   "duplicate leading `<|startoftext|>`" warning and can degrade output
   quality. `build_raw_prompt()` starts directly at `<|im_start|>system`.

4. **`MCPClient.call_tool()` returns a JSON *string*, not a dict.** Callers
   that need a dict must `json.loads()` it themselves; callers re-embedding
   it into a message `content` field must use it as-is, not
   `json.dumps()` it again.

5. **Multi-tool-call turns are natively supported by the model**, and
   `parse_tool_calls_from_content()` extracts all of them via regex
   iteration — no need for a separate "planner" or intent-routing stage.
   (An LFM2.5-Encoder-based routing approach was prototyped and abandoned
   in favor of switching to `LFM2-1.2B-Tool`, which handles compound
   queries natively and more reliably.)

6. **`temperature`/`top_k`/`top_p`/`repeat_penalty` in `ChatCompletionRequest`
   are currently accepted but ignored** by `generate_response()`, which
   hardcodes `temperature=0` per the model's documented recommendation.
   If you need those knobs back, thread them through explicitly rather
   than assuming the Pydantic schema fields do anything.

---

## Verified test matrix

These four cases are the standing regression set (see
`client_example.py`'s `example_*` functions) — re-run after any prompt,
parsing, or model change:

| Query | Expected behavior |
|---|---|
| "Hello! Please introduce yourself." | No tool call; plain-text response |
| "What's the weather like in London?" | 1 tool call (`get_weather`) → synthesized answer |
| "Please calculate 1234 * 5678 for me" | 1 tool call (`calculate`) → synthesized answer with correct number |
| "What's the weather in Paris and what time is it there?" | **2 tool calls in one turn** (`get_weather` + `get_time`) → single combined answer |

---

## Suggested next steps / open items

- If real token streaming is wanted, `generate_response()` needs a
  generator-based path using `create_completion(..., stream=True)` instead
  of the current blocking call.
- `calculate`'s `eval()` sandboxing (character whitelist) is adequate for
  local/offline use but not hardened for any future network-exposed
  deployment.
- Consider adding a `.gitignore` entry for `models/*.gguf` if this repo is
  tracked in git — GGUF files are large and shouldn't be committed.
