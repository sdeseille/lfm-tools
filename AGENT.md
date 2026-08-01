# AGENT.md — Directives for AI agents working in lfm-tools

Terse companion to README.md. README explains *why*; this file states
*rules*. If they ever conflict, this file wins for behavior, README wins
for context/rationale.

## Before touching anything

- Read `openai_api_server_local_llm.py` in full before editing it. It is
  small enough to hold in context and dense with non-obvious constraints.
- Run the regression set (below) before AND after any change to prompt
  building, parsing, or model loading. Do not skip the "before" run —
  you need a baseline to know if you broke something.

## Hard constraints — do not violate without explicit user approval

1. **`mcp` dependency must stay `<2.0.0`.**
   `mcp>=2.0.0` removes the `@app.list_tools()` / `@app.call_tool()`
   decorator API that `mcp_server.py` uses. Never widen this pin. If a
   task requires `mcp>=2.0.0` features, stop and ask the user first —
   it requires rewriting `mcp_server.py`'s API entirely.

2. **Never add `chat_format=` to the `Llama(...)` constructor.**
   The server bypasses `llama-cpp-python`'s chat-template handling
   entirely on purpose (`create_completion()`, not
   `create_chat_completion()`). Adding `chat_format` reintroduces a
   format conflict that previously broke tool-call parsing silently.

3. **Never add `<|startoftext|>` to `build_raw_prompt()`.**
   `create_completion()` adds BOS automatically. Re-adding it duplicates
   BOS and degrades output quality (llama.cpp will warn about this if it
   happens — treat that warning as a regression, not noise).

4. **`generate_response()` must call `create_completion()` with
   `temperature=0`.** This is the documented setting for
   `LFM2-1.2B-Tool`. Do not wire `request.temperature` through unless the
   user explicitly asks for it — it was deliberately hardcoded.

5. **`parse_tool_calls_from_content()` must treat
   `<|tool_call_start|>`/`<|tool_call_end|>` as OPTIONAL in the regex.**
   `llama-cpp-python` strips special tokens from output text by default,
   so they usually will not be present. Making them required silently
   breaks all tool-call detection — this exact regression happened once
   already.

6. **Never re-`json.dumps()` a value coming from `MCPClient.call_tool()`.**
   It already returns a JSON string. Wrapping it again double-encodes it;
   downstream `json.loads()` calls then return a string instead of a
   dict, causing `TypeError: string indices must be integers`.

7. **Do not reintroduce an LLM-external "router" or "intent classifier"
   stage** (e.g. an encoder-based multi-label tool router) without being
   asked. This was prototyped and deliberately abandoned — the current
   model (`LFM2-1.2B-Tool`) handles single- and multi-tool detection
   natively and reliably. If multi-tool accuracy regresses, first check
   whether prompt/parsing changes broke it before reaching for
   architecture changes.

## Always test after changes to these files

| Changed file/function | Minimum required test |
|---|---|
| `build_raw_prompt()` | Full regression set (below) |
| `parse_tool_calls_from_content()` | Full regression set, check `🔍 RAW MODEL OUTPUT` debug logs for each case |
| `create_system_prompt()` | Full regression set |
| `LocalLLMManager.load_model()` / GGUF path / quantization | Full regression set + confirm no `RuntimeWarning` in server startup logs |
| `mcp_server.py` (tool schemas or handlers) | Restart server fully (subprocess is spawned at startup) + full regression set |
| `pyproject.toml` dependency versions | Full regression set; for `mcp` specifically, also confirm server startup log shows `✓ Connected to MCP server` with no `Connection closed` error |

## Regression set (run via `python client_example.py` against a running server)

1. `"Hello! Please introduce yourself."` → **no tool call**, plain text.
2. `"What's the weather like in London?"` → **1 tool call** (`get_weather`), then a synthesized answer using the returned temperature/condition.
3. `"Please calculate 1234 * 5678 for me"` → **1 tool call** (`calculate`), final answer states the correct number (`7006652`) — watch for the model mangling multi-digit numbers when re-typing results.
4. `"What's the weather in Paris and what time is it there?"` → **2 tool calls in one turn** (`get_weather` + `get_time`), single combined final answer referencing both. This is the compound-query case — if it regresses to only firing one tool, treat it as the top-priority bug.

All four must pass before considering a change complete. Do not report a
task "done" on partial pass.

## Debugging workflow (use before guessing)

- Server-side exceptions are swallowed into HTTP 500 unless printed. The
  `chat_completions()` except block already calls `traceback.print_exc()`
  — read the **server** terminal output, not just the client's `500` error.
- `generate_response()` prints `🔍 RAW MODEL OUTPUT: ...` before parsing.
  Always check this first when tool detection seems wrong — it tells you
  immediately whether the model emitted the right Pythonic call and the
  bug is in parsing, or the model itself failed to call the right tool.
- When changing GGUF quantization, check the server startup log for
  `Lfm2*` / weight `LOAD REPORT` blocks with `UNEXPECTED`/`MISSING` keys —
  that pattern means weights failed to load and the model is running with
  random weights. This has happened before with mismatched model classes.
- If a `pip install`/`uv sync` pulls in an unexpectedly new major version
  of `mcp` or `transformers`, check changelogs for API-breaking rewrites
  before assuming your code is at fault.

## File touch policy

- `mcp_server.py`: safe to add new tools (follow the existing `Tool(...)`
  + `elif name == "...":` pattern in `call_tool()`). Do not change the
  stdio transport or the `list_tools`/`call_tool` decorator style without
  reading gotcha #1 above first.
- `client_example.py`: reference/test client. Changes here should stay
  in sync with whatever the server actually returns — it is also the
  regression-test driver, so don't let it silently start masking failures
  (e.g. don't broaden `except` blocks to hide real errors).
- `openai_api_server_local_llm.py`: see hard constraints above before any
  edit to `LocalLLMManager` or `chat_completions()`.
- `pyproject.toml`: pin changes need the regression set re-run, no
  exceptions, even for "unrelated" dependency bumps.

## What NOT to do without being asked

- Do not add streaming support by assuming `stream=True` already works —
  it is currently a no-op accepted-but-ignored field.
- Do not add authentication, rate limiting, or expose the server beyond
  `localhost` — this is a local/offline dev setup, and `calculate`'s
  `eval()`-based evaluation is not hardened for network exposure.
- Do not swap the model to a different LFM2 variant (e.g. back to
  LFM2.5-230M or a different Encoder model) without flagging that
  `build_raw_prompt()`'s literal special-token format is specific to
  `LFM2-1.2B-Tool`'s documented training format and may not transfer.