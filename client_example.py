"""
Example client to test the OpenAI-compatible API with local LLM and MCP tools
"""
import requests
import json
import time


class OpenAIClient:
    """Simple client for OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
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
    
    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/v1/models"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
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
        except Exception as e:
            print(f"  Attempt {i+1}/{max_retries}: Server not responding...")
        
        if i < max_retries - 1:
            time.sleep(delay)
    
    print("✗ Server did not become ready in time")
    return False


def example_weather_query():
    """Example: Query weather using tools"""
    print("\n" + "=" * 80)
    print("Example 1: Weather Query with Local LLM")
    print("=" * 80)
    
    client = OpenAIClient()
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in London?"
        }
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("\nGenerating response from local model...")
    
    response = client.chat_completion(messages)
    
    assistant_message = response["choices"][0]["message"]
    print(f"\nAssistant: {assistant_message.get('content', '')}")
    
    if assistant_message.get("tool_calls"):
        print("\n✓ Tool calls detected:")
        for tool_call in assistant_message["tool_calls"]:
            print(f"  - Tool: {tool_call['function']['name']}")
            print(f"  - Arguments: {tool_call['function']['arguments']}")


def example_calculation():
    """Example: Perform calculation using tools"""
    print("\n" + "=" * 80)
    print("Example 2: Calculation with Local LLM")
    print("=" * 80)
    
    client = OpenAIClient()
    
    messages = [
        {
            "role": "user",
            "content": "Please calculate 156 + 234 for me"
        }
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("\nGenerating response from local model...")
    
    response = client.chat_completion(messages)
    
    assistant_message = response["choices"][0]["message"]
    print(f"\nAssistant: {assistant_message.get('content', '')}")
    
    if assistant_message.get("tool_calls"):
        print("\n✓ Tool calls detected:")
        for tool_call in assistant_message["tool_calls"]:
            print(f"  - Tool: {tool_call['function']['name']}")
            print(f"  - Arguments: {tool_call['function']['arguments']}")


def example_time_query():
    """Example: Get current time"""
    print("\n" + "=" * 80)
    print("Example 3: Time Query with Local LLM")
    print("=" * 80)
    
    client = OpenAIClient()
    
    messages = [
        {
            "role": "user",
            "content": "What time is it in UTC?"
        }
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("\nGenerating response from local model...")
    
    response = client.chat_completion(messages)
    
    assistant_message = response["choices"][0]["message"]
    print(f"\nAssistant: {assistant_message.get('content', '')}")
    
    if assistant_message.get("tool_calls"):
        print("\n✓ Tool calls detected:")
        for tool_call in assistant_message["tool_calls"]:
            print(f"  - Tool: {tool_call['function']['name']}")
            print(f"  - Arguments: {tool_call['function']['arguments']}")


def example_general_query():
    """Example: General query without tools"""
    print("\n" + "=" * 80)
    print("Example 4: General Query (No Tools)")
    print("=" * 80)
    
    client = OpenAIClient()
    
    messages = [
        {
            "role": "user",
            "content": "Hello! Can you introduce yourself?"
        }
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("\nGenerating response from local model...")
    
    response = client.chat_completion(messages)
    
    assistant_message = response["choices"][0]["message"]
    print(f"\nAssistant: {assistant_message.get('content', '')}")


def example_system_info():
    """Example: Display system information"""
    print("\n" + "=" * 80)
    print("System Information")
    print("=" * 80)
    
    client = OpenAIClient()
    
    # Health check
    health = client.health_check()
    print(f"\n✓ API Health:")
    print(f"  - Status: {health['status']}")
    print(f"  - MCP Connected: {health['mcp_connected']}")
    print(f"  - Model Loaded: {health['model_loaded']}")
    print(f"  - Model Path: {health.get('model_path', 'N/A')}")
    print(f"  - Tools Available: {health['tools_count']}")
    
    # List models
    models = client.list_models()
    print(f"\n✓ Available Models:")
    for model in models["data"]:
        print(f"  - {model['id']} (owned by {model['owned_by']})")
    
    # List tools
    tools = client.list_tools()
    print(f"\n✓ Available Tools:")
    for tool in tools["data"]:
        func = tool["function"]
        print(f"  - {func['name']}: {func['description']}")


if __name__ == "__main__":
    print("=" * 80)
    print("OpenAI-Compatible API with Local LLM - Client Examples")
    print("=" * 80)
    
    try:
        client = OpenAIClient()
        
        # Wait for server to be ready
        if not wait_for_server(client):
            print("\n✗ Server is not ready. Please start the server first:")
            print("  python openai_api_server_local_llm.py")
            exit(1)
        
        # Run examples
        example_system_info()
        example_general_query()
        example_weather_query()
        example_calculation()
        example_time_query()
        
        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server")
        print("  Make sure the server is running: python openai_api_server_local_llm.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()