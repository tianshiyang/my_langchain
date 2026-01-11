#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/10 19:05
@Author  : tianshiyang
@File    : use_mcp.py
"""
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from provider import chatGptLLM



async def get_tools():
    client = MultiServerMCPClient(
        {
            # "weather": {
            #     "transport": "http",
            #     "url": "http://127.0.0.1:8000/mcp",
            #     "headers": {
            #         "Authorization": "Bearer YOUR_TOKEN",
            #         "X-Custom-Header": "custom-value"
            #     }
            # },
            # "math": {
            #     "transport": "stdio",
            #     "command": "/Users/icourt1/Desktop/my_langchain/.venv/bin/python",
            #     "args": ["/Users/icourt1/Desktop/my_langchain/src/langchain_mcp/mcp_tool_math.py"]
            # },
            "weather": {
                "transport": "http",
                "url": "http://101.201.173.163:8000/mcp"
            },
            # "amap-maps": {
            #   "transport": "http",
            #   "url": "https://mcp.api-inference.modelscope.net/49529e12ee7e4e/mcp"
            # },
            # "12306-mcp": {
            #     "transport": "http",
            #     "url": "https://mcp.api-inference.modelscope.net/d375afa19c744c/mcp"
            # }
        }
    )
    tools = await client.get_tools()
    for tool in tools:
        print(tool.name)
    agent = create_agent(
        chatGptLLM,
        tools=tools
    )
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "北京的天气怎么样"}]}
    )
    print(response["messages"][-1].pretty_print())
    print("*" * 50)
    for message in response["messages"]:
        print(message.pretty_print())

if __name__ == "__main__":
    import asyncio
    asyncio.run(get_tools())