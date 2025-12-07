#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 16:03
@Author  : tianshiyang
@File    : quick_start.py
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from provider import chatGptLLM, qwenLLM


# def get_weather(city: str) -> str:
#     """获取传入城市名称的天气"""
#     print('city', city)
#     return f"{city}总是晴天"
#
# agent = create_agent(
#     model=chatGptLLM,
#     tools=[get_weather],
#     system_prompt="你是一个乐于助人的AI助手"
# )
#
# result = agent.invoke(HumanMessage("北京的天气怎么样"))
# print(result)

# =========================

# 第一步定义系统提示词

SYSTEM_PROMPT = """
你是一位天气预报专家，说话总是双关语。

您可以访问两个工具：

- get_weather_for_location：使用它来获取特定位置的天气
— get_user_location：使用它来获取用户的位置

如果用户向你询问天气，确保你知道它的位置。如果您可以从问题中知道它们的意思，那么可以使用get_user_location工具找到它们的位置。
"""

# 第二步：创建工具

@tool
def get_weather_for_location(city: str) -> str:
    """ 根据城市获取天气信息
    Args:
        city: 城市名换

    Returns: 当前城市的天气情况

    """
    return f"{city}总是晴天"

class Context(BaseModel):
    """用户运行时的上下文信息"""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID检索用户信息"""
    user_id = runtime.context.user_id
    return "上海" if user_id == "1" else "北京"

# 第三步：配置模型
model = qwenLLM

# 第四步：定义响应格式
@dataclass
class ResponseFormat:
    """agent的返回数据格式"""
    # 总是返回一个双语相关的回复
    punny_response: str
    # 天气情况
    weather_conditions: str | None = None

# 第五步：添加内存
checkpointer = InMemorySaver()

# 第六步：创建并运行代理
agent = create_agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

# 'thread_id'是给定对话的唯一标识符
config = RunnableConfig(
    configurable={
        "thread_id": "1"
    }
)

response = agent.invoke(
    HumanMessage("外面天气怎么样"),
    config=config,
    context=Context(user_id="1"),
)

print(response['messages'][-1].content)