#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 22:52
@Author  : tianshiyang
@File    : messages.py
"""
from provider.llms import chatGptLLM
from langchain.tools import tool
@tool
def get_weather(location: str) -> str:
    """
    通过地理位置获取天气信息
    Args:
        location:地理位置

    Returns:天气信息

    """
    return f"{location}的天气是晴天"

response = chatGptLLM.bind_tools([get_weather]).invoke("北京的天气")
for tool_call in response.tool_calls:
    """获取工具调用信息"""
    print(tool_call)
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")

"""获取token用量"""
print(response.usage_metadata)

"""流式输出"""

chunks = chatGptLLM.bind_tools([get_weather]).stream("北京的天气")
for chunk in chunks:
    print(chunk)
