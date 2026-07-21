"""
OpenAI-compatible API server with MCP tool calling using local LiquidAI model
"""
import asyncio
import json
import uuid
import re
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

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
    temperature: Optional[float] = 0.05
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
        model_path: str = "LiquidAI/LFM2.5-230M-GGUF",
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
                chat_format="chatml"  # Use ChatML format for tool calling
            )
            
            self.model_path = model_path
            print(f"✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def create_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Create system prompt with tool descriptions"""
        system_prompt = """You are an AI assistant that can help users by calling available tools.
When a user asks a question, determine if a tool should be called to help answer.
If a tool is needed, respond with a tool call using the following JSON format:
{"name": "tool_name", "arguments": {"param": "value"}}

Available tools:
"""
        for tool in tools:
            func = tool["function"]
            system_prompt += f"\n- {func['name']}: {func['description']}"
            if func.get('parameters'):
                params = func['parameters'].get('properties', {})
                system_prompt += f"\n  Parameters: {', '.join(params.keys())}"
        
        system_prompt += "\n\nIf no tool is needed, answer the user directly. Be concise and helpful."
        return system_prompt
    
    def parse_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from model response content.
        Looks for JSON objects with 'name' and 'arguments' fields.
        """
        tool_calls = []
        
        # Pattern to match JSON-like tool calls
        pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                tool_call_json = match.group(0)
                tool_call_dict = json.loads(tool_call_json)
                
                if "name" in tool_call_dict and "arguments" in tool_call_dict:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_call_dict["name"],
                            "arguments": json.dumps(tool_call_dict["arguments"])
                        }
                    })
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    def strip_thinking_tags(self, content: str) -> str:
        """Remove <think>...</think> tags from content"""
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    
    async def generate_response(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        temperature: float = 0.05,
        max_tokens: int = 2048,
        top_k: int = 50,
        top_p: float = 0.1,
        repeat_penalty: float = 1.05,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response from local model"""
        
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Convert messages to llama-cpp format
        llama_messages: List[ChatCompletionRequestMessage] = []
        
        # Add system message with tool descriptions
        system_prompt = self.create_system_prompt(tools)
        llama_messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation messages
        for msg in messages:
            if msg.role == "tool":
                # Add tool results as user messages
                llama_messages.append({
                    "role": "user",
                    "content": f"Tool result from {msg.name}: {msg.content}"
                })
            else:
                llama_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })
        
        # Convert tools to llama-cpp format
        llama_tools = [
            {
                "type": "function",
                "function": tool["function"]
            }
            for tool in tools
        ]
        
        try:
            # Generate response
            response = self.model.create_chat_completion(
                messages=llama_messages,
                tools=llama_tools if llama_tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=stream
            )
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_non_streaming_response(response)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    def _handle_non_streaming_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Handle non-streaming response"""
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content", "")
        
        # Strip thinking tags
        content = self.strip_thinking_tags(content)
        
        # Parse tool calls from content
        tool_calls = self.parse_tool_calls_from_content(content)
        
        # Build response
        assistant_message = {
            "role": "assistant",
            "content": content if not tool_calls else None
        }
        
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        
        return {
            "message": assistant_message,
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }
    
    def _handle_streaming_response(self, response: Iterator[CreateChatCompletionStreamResponse]) -> Dict[str, Any]:
        """Handle streaming response"""
        content_chunks = []
        
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                content_chunks.append(delta["content"])
        
        content = "".join(content_chunks)
        
        # Strip thinking tags
        content = self.strip_thinking_tags(content)
        
        # Parse tool calls
        tool_calls = self.parse_tool_calls_from_content(content)
        
        assistant_message = {
            "role": "assistant",
            "content": content if not tool_calls else None
        }
        
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        
        return {
            "message": assistant_message,
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="OpenAI-Compatible API with Local LLM and MCP Tools")

# Global instances
mcp_client = MCPClient()
llm_manager = LocalLLMManager()


@app.on_event("startup")
async def startup_event():
    """Initialize MCP and local model on startup"""
    # Connect to MCP server
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
    
    # Load local model
    try:
        # Adjust the path to your downloaded model
        model_path = "models/LFM2.5-230M-Q8_0.gguf"  # Update this path
        llm_manager.load_model(model_path, n_ctx=8192, verbose=False)
    except Exception as e:
        print(f"✗ Failed to load local model: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    await mcp_client.close()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with tool calling
    """
    try:
        messages = request.messages
        tools = request.tools or mcp_client.tools
        
        # Check if last message is a tool result - need to continue conversation
        last_message = messages[-1] if messages else None
        needs_tool_execution = False
        
        # Generate response from local model
        result = await llm_manager.generate_response(
            messages=messages,
            tools=tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            repeat_penalty=request.repeat_penalty,
            stream=request.stream
        )
        
        assistant_message = result["message"]
        finish_reason = result["finish_reason"]
        
        # If model wants to call tools, execute them
        if assistant_message.get("tool_calls"):
            tool_calls = assistant_message["tool_calls"]
            
            # Execute first tool call (you can extend this to handle multiple)
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                try:
                    # Call tool through MCP
                    tool_result = await mcp_client.call_tool(function_name, function_args)
                    
                    # For now, include result in content
                    # In a proper implementation, client would send tool result back
                    if not assistant_message.get("content"):
                        assistant_message["content"] = ""
                    assistant_message["content"] += f"\n\nTool Result: {tool_result}"
                    
                except Exception as e:
                    assistant_message["content"] = f"Error calling tool {function_name}: {str(e)}"
                    finish_reason = "stop"
        
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
                "prompt_tokens": 100,  # Approximate
                "completion_tokens": 50,  # Approximate
                "total_tokens": 150
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
    model_id = "LFM2.5-230M" if llm_manager.model_path else "unknown"
    
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
        "model": "LiquidAI/LFM2.5-230M-GGUF",
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