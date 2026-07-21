"""
Simple MCP Server that exposes basic tools
"""
import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ListToolsResult
)

# Initialize MCP server
app = Server("simple-tools-server")

# Define available tools
TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a location",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'London, UK'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    ),
    Tool(
        name="calculate",
        description="Perform basic mathematical calculations",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    ),
    Tool(
        name="get_time",
        description="Get the current time for a timezone",
        inputSchema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g., 'America/New_York'",
                    "default": "UTC"
                }
            }
        }
    )
]


@app.list_tools()
async def list_tools() -> ListToolsResult:
    """List all available tools"""
    return ListToolsResult(tools=TOOLS)


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """
    Execute tool based on name and arguments
    
    Args:
        name: Tool name to execute
        arguments: Dictionary of arguments for the tool
        
    Returns:
        CallToolResult with execution results
    """
    try:
        if name == "get_weather":
            location = arguments.get("location")
            unit = arguments.get("unit", "celsius")
            
            # Simulate weather API call
            result = {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "unit": unit,
                "condition": "Partly cloudy",
                "humidity": 65
            }
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )
                ]
            )
        
        elif name == "calculate":
            expression = arguments.get("expression")
            
            # Safe evaluation (in production, use a proper math parser)
            try:
                # Limit to basic operations
                allowed_chars = set("0123456789+-*/() .")
                if not all(c in allowed_chars for c in expression):
                    raise ValueError("Invalid characters in expression")
                
                result = eval(expression, {"__builtins__": {}}, {})
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "expression": expression,
                                "result": result
                            }, indent=2)
                        )
                    ]
                )
            except Exception as e:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": f"Calculation error: {str(e)}"
                            })
                        )
                    ],
                    isError=True
                )
        
        elif name == "get_time":
            from datetime import datetime
            import pytz
            
            timezone = arguments.get("timezone", "UTC")
            
            try:
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                
                result = {
                    "timezone": timezone,
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "iso_format": current_time.isoformat()
                }
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(result, indent=2)
                        )
                    ]
                )
            except Exception as e:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": f"Timezone error: {str(e)}"
                            })
                        )
                    ],
                    isError=True
                )
        
        else:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Unknown tool: {name}"
                        })
                    )
                ],
                isError=True
            )
            
    except Exception as e:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Tool execution error: {str(e)}"
                    })
                )
            ],
            isError=True
        )


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())