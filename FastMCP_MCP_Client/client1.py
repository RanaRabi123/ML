import os 
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS = { 
    "math": {
        "transport": "stdio",
        "command": "C:\\Users\\User\\.local\\bin\\uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:\\intern preparation\\FastMCP_MCP_Client\\main.py"
       ]
    },
    "expense": {
    "transport": "streamable_http",
    "url": "https://test-server-very-silver-koala.fastmcp.app/mcp",
    "headers": {
        "Authorization": f"Bearer {os.getenv('MCP_API_KEY')}"
        }
    },
    # "manim-server": {
    #     "transport": "stdio",
    #     "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
    #     "args": [
    #     "/Users/nitish/desktop/manim-mcp-server/src/manim_server.py"
    #   ],
    #     "env": {
    #     "MANIM_EXECUTABLE": "/Library/Frameworks/Python.framework/Versions/3.11/bin/manim"
    #   }
    # }
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()


    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys())

    llm = ChatGroq(model="openai/gpt-oss-20b")
    llm_with_tools = llm.bind_tools(tools)

    prompt = "can you add 123 and 234"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())