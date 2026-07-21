"""
Example client with proper tool calling loop
Follows OpenAI's tool calling pattern
"""
import requests
import json
import time
from typing import List, Dict, Any


class OpenAIClient:
    """Client for OpenAI-compatible API with tool calling support"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def chat_completion(
        self,
        messages: list,
        model: str = "LFM2.5-230M",
        tools: list = None,
        tool_choice: str = "auto",
        temperature: float = 0.05,
        max_tokens: int = 2048
    ):
        """Send chat completion request"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def execute_tool_call(self, tool_name: str, tool_arguments: dict) -> dict:
        """
        Execute a tool call through the MCP server endpoint
        
        Note: This calls the server which then calls MCP.
        In production, you might want a direct MCP client here.
        """
        # For now, we'll use a helper endpoint to execute tools
        # You could also implement a direct MCP client here
        url = f"{self.base_url}/v1/tools/execute"
        
        payload = {
            "name": tool_name,
            "arguments": tool_arguments
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_tools(self):
        """List available tools"""
        url = f"{self.base_url}/v1/tools"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check API health"""
        url = f"{self.base_url}/health"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class ToolCallingAgent:
    """Agent that handles the complete tool calling loop"""
    
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.max_iterations = 5
    
    def run(
        self,
        user_message: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling loop
        
        Returns:
            Final response and conversation history
        """
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🧑 User: {user_message}")
            print(f"{'='*80}\n")
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get response from LLM
            response = self.client.chat_completion(messages)
            assistant_message = response["choices"][0]["message"]
            finish_reason = response["choices"][0]["finish_reason"]
            
            if verbose:
                print(f"🤖 Assistant Response:")
                print(f"   Finish Reason: {finish_reason}")
            
            # Add assistant message to history
            messages.append(assistant_message)
            
            # Check if tool calls are needed
            tool_calls = assistant_message.get("tool_calls")
            
            if not tool_calls:
                # No tool calls - this is the final answer
                if verbose:
                    print(f"💬 Final Answer: {assistant_message.get('content', '')}\n")
                
                return {
                    "status": "completed",
                    "final_answer": assistant_message.get("content", ""),
                    "messages": messages,
                    "iterations": iteration + 1
                }
            
            # Execute tool calls
            if verbose:
                print(f"🔧 Tool Calls Detected: {len(tool_calls)}")
            
            for tool_call in tool_calls:
                tool_call_id = tool_call["id"]
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                if verbose:
                    print(f"\n   📞 Calling Tool: {function_name}")
                    print(f"   📝 Arguments: {json.dumps(function_args, indent=6)}")
                
                # Execute the tool
                tool_result = self.client.execute_tool_call(function_name, function_args)
                
                if verbose:
                    print(f"   ✅ Result: {json.dumps(tool_result, indent=6)}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps(tool_result)
                })
            
            if verbose:
                print(f"\n{'─'*80}\n")
        
        # Max iterations reached
        if verbose:
            print(f"⚠️  Max iterations ({self.max_iterations}) reached\n")
        
        return {
            "status": "max_iterations_reached",
            "final_answer": messages[-1].get("content", ""),
            "messages": messages,
            "iterations": self.max_iterations
        }


def print_conversation_history(messages: List[Dict[str, Any]]):
    """Pretty print conversation history"""
    print(f"\n{'='*80}")
    print("📜 Conversation History")
    print(f"{'='*80}\n")
    
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            print(f"{i}. 🧑 User:")
            print(f"   {content}\n")
        elif role == "assistant":
            print(f"{i}. 🤖 Assistant:")
            if msg.get("tool_calls"):
                print(f"   [Requesting tool calls]")
                for tc in msg["tool_calls"]:
                    print(f"   - {tc['function']['name']}")
            else:
                print(f"   {content}")
            print()
        elif role == "tool":
            print(f"{i}. 🔧 Tool Result ({msg.get('name', 'unknown')}):")
            print(f"   {content[:100]}..." if len(content) > 100 else f"   {content}")
            print()


# ============================================================================
# EXAMPLES
# ============================================================================

def example_weather_with_tools():
    """Example: Weather query with automatic tool execution"""
    print("\n" + "="*80)
    print("Example 1: Weather Query with Tool Calling Loop")
    print("="*80)
    
    client = OpenAIClient()
    agent = ToolCallingAgent(client)
    
    result = agent.run(
        "What's the weather like in London?",
        verbose=True
    )
    
    print_conversation_history(result["messages"])
    print(f"\n✅ Completed in {result['iterations']} iterations")


def example_calculation_with_tools():
    """Example: Math calculation with tool calling"""
    print("\n" + "="*80)
    print("Example 2: Calculation with Tool Calling Loop")
    print("="*80)
    
    client = OpenAIClient()
    agent = ToolCallingAgent(client)
    
    result = agent.run(
        "Please calculate 1234 * 5678 for me",
        verbose=True
    )
    
    print_conversation_history(result["messages"])
    print(f"\n✅ Completed in {result['iterations']} iterations")


def example_multi_tool_query():
    """Example: Query requiring multiple tools"""
    print("\n" + "="*80)
    print("Example 3: Multi-Tool Query")
    print("="*80)
    
    client = OpenAIClient()
    agent = ToolCallingAgent(client)
    
    result = agent.run(
        "What's the weather in Paris and what time is it there?",
        verbose=True
    )
    
    print_conversation_history(result["messages"])
    print(f"\n✅ Completed in {result['iterations']} iterations")


def example_no_tools_needed():
    """Example: Simple query without tools"""
    print("\n" + "="*80)
    print("Example 4: Simple Query (No Tools)")
    print("="*80)
    
    client = OpenAIClient()
    agent = ToolCallingAgent(client)
    
    result = agent.run(
        "Hello! Please introduce yourself.",
        verbose=True
    )
    
    print(f"\n✅ Completed in {result['iterations']} iterations")


def wait_for_server(client: OpenAIClient, max_retries: int = 10, delay: int = 2):
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    for i in range(max_retries):
        try:
            health = client.health_check()
            if health.get("model_loaded"):
                print("✓ Server is ready!")
                return True
            else:
                print(f"  Attempt {i+1}/{max_retries}: Model not loaded yet...")
        except Exception:
            print(f"  Attempt {i+1}/{max_retries}: Server not responding...")
        
        if i < max_retries - 1:
            time.sleep(delay)
    
    print("✗ Server did not become ready in time")
    return False


if __name__ == "__main__":
    print("="*80)
    print("OpenAI-Compatible API - Tool Calling Examples")
    print("="*80)
    
    try:
        client = OpenAIClient()
        
        # Wait for server
        if not wait_for_server(client):
            print("\n✗ Please start the server: python openai_api_server_local.py")
            exit(1)
        
        # Show available tools
        tools = client.list_tools()
        print(f"\n✓ Available Tools: {len(tools['data'])}")
        for tool in tools["data"]:
            print(f"  - {tool['function']['name']}")
        
        # Run examples
        example_no_tools_needed()
        example_weather_with_tools()
        example_calculation_with_tools()
        example_multi_tool_query()
        
        print("\n" + "="*80)
        print("✅ All examples completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to server")
        print("  Start server: python openai_api_server_local.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()