"""
OpenAI-compatible API server with MCP tool calling using local LiquidAI model
"""
import ast
import asyncio
import json
import uuid
import re
import time
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionStreamResponse,
    Llama,
)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ============================================================================
# PYDANTIC MODELS - OpenAI API Compatibility
# ============================================================================

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.1
    repeat_penalty: Optional[float] = 1.05


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    """MCP Client wrapper for tool management"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = []
        self.stdio_context = None
    
    async def connect(self, server_script_path: str = "mcp_server.py"):
        """Connect to MCP server"""
        import os
        
        server_params = StdioServerParameters(
            command="python",
            args=[os.path.abspath(server_script_path)],
            env=None
        )
        
        # Store context manager
        self.stdio_context = stdio_client(server_params)
        
        # Enter the context
        read_stream, write_stream = await self.stdio_context.__aenter__()
        
        # Create and initialize session
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
        # List available tools
        try:
            tools_result = await self.session.list_tools()
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools_result.tools
            ]
        except Exception as e:
            print(f"Error listing tools: {e}")
            self.tools = []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool through MCP"""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        result = await self.session.call_tool(name, arguments)
        
        # Extract text content from result
        if result.content:
            return result.content[0].text
        return json.dumps({"error": "No content returned"})
    
    async def close(self):
        """Close MCP connection"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error closing MCP connection: {e}")


# ============================================================================
# LOCAL LLM MODEL MANAGER
# ============================================================================

class LocalLLMManager:
    """Manager for local LiquidAI LFM model"""
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_path: Optional[str] = None
    
    def load_model(
        self,
        model_path: str = "LiquidAI/LFM2.5-350M-Q4_K_M.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,  # Use GPU if available
        verbose: bool = False
    ):
        """Load the local LiquidAI model"""
        try:
            print(f"Loading model from: {model_path}")
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                # No chat_format — we build the prompt manually below and call
                # create_completion(), not create_chat_completion(), so llama-cpp's
                # own template handling is bypassed entirely.
            )
            
            self.model_path = model_path
            print(f"✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def create_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Exact format LFM2.5-350M was trained on."""
        tool_funcs = [t["function"] for t in tools]
        return f"List of tools: <|tool_list_start|>{json.dumps(tool_funcs)}<|tool_list_end|>"

    def build_raw_prompt(self, messages: List[Message], tools: List[Dict[str, Any]]) -> str:
        """
        Build the literal LFM2 chat format instead of trusting an
        auto-detected Jinja template — this is the format shown verbatim
        in the LFM2.5-350M model card.
        """
        parts = ["<|im_start|>system\n"]
        parts.append(self.create_system_prompt(tools))
        parts.append("<|im_end|>\n")

        for msg in messages:
            if msg.role == "user":
                parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>\n")

            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Reconstruct the Pythonic call(s) exactly as the model
                    # would have emitted them, wrapped in the special tokens.
                    calls = []
                    for tc in msg.tool_calls:
                        args = json.loads(tc["function"]["arguments"])
                        arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                        calls.append(f'{tc["function"]["name"]}({arg_str})')
                    call_block = f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"
                    trailing_text = msg.content or ""
                    parts.append(f"<|im_start|>assistant\n{call_block}{trailing_text}<|im_end|>\n")
                else:
                    parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>\n")

            elif msg.role == "tool":
                # content is already the raw JSON string from MCP — wrap it
                # in the special tokens the doc specifies, don't re-encode it.
                parts.append(f"<|im_start|>tool\n<|tool_response_start|>{msg.content}<|tool_response_end|><|im_end|>\n")

        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    
    def parse_tool_calls_from_content(
        self,
        content: str,
        tools: List[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], str]:
        tool_calls = []

        tools_by_name = {}
        if tools:
            for t in tools:
                func = t.get("function", {})
                name = func.get("name")
                if name:
                    tools_by_name[name] = func.get("parameters", {}).get("properties", {})

        def extract_arg_value(node: ast.expr, expected_type: str = None):
            # Fast path: the model quoted/typed the value correctly, so it's
            # a plain literal (str/int/float/bool/None) — use it as-is.
            if isinstance(node, ast.Constant):
                return node.value
            # Fallback: anything else (a bare identifier the model forgot to
            # quote, e.g. `location=Paris`, or an arithmetic expression like
            # `expression=4 * 3`) isn't a Python literal, so ast.literal_eval
            # would raise and we'd lose the whole call. Reconstruct the
            # original source text instead — for the string-typed params
            # every current tool uses, that's exactly the intended value.
            try:
                raw = ast.unparse(node)
            except Exception:
                return None
            if expected_type == "integer":
                try:
                    return int(ast.literal_eval(node))
                except (ValueError, TypeError, SyntaxError):
                    return None
            if expected_type == "number":
                try:
                    return float(ast.literal_eval(node))
                except (ValueError, TypeError, SyntaxError):
                    return None
            if expected_type == "boolean":
                return raw.strip().lower() in ("true", "1")
            return raw

        def is_valid_call(name: str, arguments: dict) -> bool:
            if not tools_by_name:
                return True
            if name not in tools_by_name:
                return False
            schema = tools_by_name[name]
            for arg_name, arg_value in arguments.items():
                enum = schema.get(arg_name, {}).get("enum")
                if enum and arg_value not in enum:
                    return False
            return True

        # Wrapper tokens are optional — llama-cpp-python strips special
        # tokens from detokenized text by default, so they may or may not
        # actually appear in `content`. Match the bracket list either way.
        wrapper_pattern = r'(?:<\|tool_call_start\|>)?\s*\[(.*?)\]\s*(?:<\|tool_call_end\|>)?'
        match = re.search(wrapper_pattern, content, re.DOTALL)

        remaining_text = content
        if match and re.search(r'\w+\s*\(', match.group(1)):  # confirm it looks like call syntax, not stray brackets
            calls_str = match.group(1)

            # Parse as real Python syntax instead of regex. The old regex
            # pair (`\w+\((.*?)\)` + `\w+="([^"]*)"`) broke on two common
            # compound-query cases: (1) string args containing literal
            # parens, e.g. calculate(expression="(12+3)*2") next to another
            # call in the same bracket — the non-greedy `.*?` in the outer
            # pattern truncated at the first `)`, splitting one call into
            # two garbage fragments; (2) any non-string arg (int/float/bool)
            # was silently dropped because the arg regex required quotes,
            # which then failed `is_valid_call()` for tools with numeric
            # params. ast.parse gives us real Python semantics for both.
            try:
                parsed = ast.parse(f"[{calls_str}]", mode="eval")
                call_nodes = [n for n in parsed.body.elts if isinstance(n, ast.Call)]
            except (SyntaxError, ValueError):
                call_nodes = []

            if call_nodes:
                remaining_text = content[:match.start()] + content[match.end():]

                for node in call_nodes:
                    if not isinstance(node.func, ast.Name):
                        continue
                    func_name = node.func.id
                    schema = tools_by_name.get(func_name, {})

                    arguments = {}
                    valid_args = True
                    for kw in node.keywords:
                        if kw.arg is None:  # **kwargs — not a valid tool-call shape
                            valid_args = False
                            break
                        expected_type = schema.get(kw.arg, {}).get("type")
                        value = extract_arg_value(kw.value, expected_type)
                        if value is None and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                            valid_args = False
                            break
                        arguments[kw.arg] = value
                    if not valid_args:
                        continue

                    if is_valid_call(func_name, arguments):
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": func_name, "arguments": json.dumps(arguments)}
                        })

        return tool_calls, remaining_text.strip()
   
    def strip_thinking_tags(self, content: str) -> str:
        """Remove <think>...</think> tags and any stray LFM tool-call tokens
        that leaked into plain-text output."""
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        content = content.replace("<|tool_call_start|>", "").replace("<|tool_call_end|>", "")
        return content.strip()
    
    async def generate_response(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Model not loaded")

        generation_id = uuid.uuid4().hex[:8]
        generation_start = time.perf_counter()

        # ---------------------------------------------------------
        # Prompt construction
        # ---------------------------------------------------------
        prompt_start = time.perf_counter()

        prompt = self.build_raw_prompt(messages, tools)

        prompt_build_seconds = time.perf_counter() - prompt_start

        prompt_chars = len(prompt)

        # ---------------------------------------------------------
        # LLM inference
        # ---------------------------------------------------------
        inference_start = time.perf_counter()

        output = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            stop=["<|im_end|>"],
        )

        inference_seconds = time.perf_counter() - inference_start

        # ---------------------------------------------------------
        # Token usage provided by llama-cpp-python
        # ---------------------------------------------------------
        usage = output.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        completion_tokens_per_second = None

        if completion_tokens and inference_seconds > 0:
            completion_tokens_per_second = (
                completion_tokens / inference_seconds
            )
        
        total_tokens = usage.get("total_tokens")

        # ---------------------------------------------------------
        # Model output
        # ---------------------------------------------------------
        raw_text = output["choices"][0]["text"]

        # ---------------------------------------------------------
        # Tool-call parsing
        # ---------------------------------------------------------
        parsing_start = time.perf_counter()

        tool_calls, plain_text = self.parse_tool_calls_from_content(
            raw_text,
            tools
        )

        parsing_seconds = time.perf_counter() - parsing_start

        # ---------------------------------------------------------
        # Total generation pipeline
        # ---------------------------------------------------------
        total_seconds = time.perf_counter() - generation_start

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------
        print(
            f"[LLM:{generation_id}] "
            f"prompt_build={prompt_build_seconds:.4f}s "
            f"inference={inference_seconds:.4f}s "
            f"parsing={parsing_seconds:.4f}s "
            f"total={total_seconds:.4f}s "
            f"prompt_chars={prompt_chars} "
            f"prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} "
            f"completion_tok_s={completion_tokens_per_second:.2f} "
            if completion_tokens_per_second is not None
            else ""
            f"total_tokens={total_tokens} "
            f"tool_calls={len(tool_calls)}"
        )

        if tool_calls:
            assistant_message = {
                "role": "assistant",
                "content": plain_text or None,
                "tool_calls": tool_calls
            }
            finish_reason = "tool_calls"
        else:
            assistant_message = {
                "role": "assistant",
                "content": raw_text.strip()
            }
            finish_reason = "stop"

        return {
            "message": assistant_message,
            "finish_reason": finish_reason,
            "metrics": {
                "generation_id": generation_id,
                "prompt_build_seconds": prompt_build_seconds,
                "inference_seconds": inference_seconds,
                "parsing_seconds": parsing_seconds,
                "total_seconds": total_seconds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tokens_per_second": completion_tokens_per_second,
                "total_tokens": total_tokens,
                "tool_calls": len(tool_calls),
            }
        }
    

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Global instances
mcp_client = MCPClient()
llm_manager = LocalLLMManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP and local model on startup, close on shutdown"""
    # --- Startup ---
    try:
        print("Connecting to MCP server...")
        await mcp_client.connect()
        print(f"✓ Connected to MCP server")
        print(f"✓ Available tools: {len(mcp_client.tools)}")
        for tool in mcp_client.tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    except Exception as e:
        print(f"✗ Failed to connect to MCP server: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Adjust the path to your downloaded model
        model_path = "models/LFM2.5-350M-Q4_K_M.gguf"  # Update this path
        llm_manager.load_model(model_path, n_ctx=2048, verbose=True)
    except Exception as e:
        print(f"✗ Failed to load local model: {e}")
        import traceback
        traceback.print_exc()

    yield

    # --- Shutdown ---
    await mcp_client.close()


app = FastAPI(title="OpenAI-Compatible API with Local LLM and MCP Tools", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with tool calling
    """
    try:
        tools = request.tools or mcp_client.tools

        # Generate response from local model
        result = await llm_manager.generate_response(
            messages=request.messages,
            tools=tools if tools else [],
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        assistant_message = result["message"]
        finish_reason = result["finish_reason"]
        usage = result["metrics"]
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": finish_reason
                }
            ],
            usage={
                "prompt_tokens": usage["prompt_tokens"] or 0,
                "completion_tokens": usage["completion_tokens"] or 0,
                "total_tokens": usage["total_tokens"] or 0
            }
        )
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    model_id = llm_manager.model_path or "unknown"
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "LiquidAI",
                "permission": [],
                "root": model_id,
                "parent": None
            }
        ]
    }


@app.get("/v1/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "object": "list",
        "data": mcp_client.tools
    }


@app.post("/v1/tools/execute")
async def execute_tool(request: Dict[str, Any]):
    """
    Execute a tool through MCP
    
    Request body:
    {
        "name": "tool_name",
        "arguments": {"param": "value"}
    }
    """
    try:
        tool_name = request.get("name")
        tool_arguments = request.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        # Execute through MCP
        result = await mcp_client.call_tool(tool_name, tool_arguments)
        
        # Parse JSON result
        try:
            result_dict = json.loads(result)
            return result_dict
        except json.JSONDecodeError:
            return {"result": result}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mcp_connected": mcp_client.session is not None,
        "model_loaded": llm_manager.model is not None,
        "model_path": llm_manager.model_path,
        "tools_count": len(mcp_client.tools)
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenAI-Compatible API with Local LLM",
        "version": "1.0.0",
        "model": llm_manager.model_path or "LiquidAI/LFM2.5-350M-GGUF",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "tools": "/v1/tools",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "openai_api_server_local_llm:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to prevent model reloading
    )