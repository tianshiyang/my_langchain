#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 23:55
@Author  : tianshiyang
@File    : tools.py
"""
from typing import Literal, Any

from langchain.agents import create_agent
from langchain_core.messages import RemoveMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from provider import chatGptLLM

#
# # 复杂参数用Pydantic的BaseModel
class WeatherInput(BaseModel):
    location: str = Field(description="城市信息")
    units: Literal['celsius', 'fahrenheit'] = Field('celsius', description="温度单位偏好")
    include_forecast: bool = Field(False, description="包括5天的预测")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = 'celsius', include_forecast: bool = False) -> str:
    """

    Args:
        location: 城市信息
        units: 温度单位
        include_forecast: 是否包括未来5天

    Returns:

    """
    temp = 22 if units == 'celsius' else 72
    result = f"当前{location}的天气信息：{temp}{units}度"
    if include_forecast:
        result = "\n五天后：晴天"
    return result


# ===============================================================================================

# 访问上下文
"""
工具可以通过参数访问运行时信息，ToolRuntime这个参数提供
- State状态 -- 可变数据，通过执行流程（例如：消息、计数器、自定义字段）
- Context上下文 -- 不可变配置，如用户ID，会话详情或者应用特定的配置
- Store存储 -- 长期记忆--跨会话
- Stream Writer流编写器 在工具执行是流式自定义更新
- Config配置 --用于执行的RunnableConfig
- Tool Call ID 工具调用ID， 当前工具调用的ID

-- 使用
参数中加入runtime: ToolRuntime即可，他会自动注入，不会暴露给大语言模型
"""

# 工具可以通过使用ToolRuntime访问当前graph的状态


## 获取当前会话状态
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """总结会话内容"""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
    return f"会话中包含{human_msgs}个人类消息，{ai_msgs}个ai消息，{tool_msgs}个工具消息"

## 访问自定义状态字段
@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """获取用户喜好"""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "未设置")


# ==================================================

# 更新状态 -> 用Command 更新agent的状态或者是graph的状态

@tool
def clear_conversation() -> Command:
    """清除历史记录"""
    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }
    )

@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """更新用户名称"""
    return Command(
        update={
            "user_name": new_name
        }
    )

# ==========================================================================================
# Context用于不可变的配置和上下文数据，如用户id，会话细节或应用特定配置，通过runtime.context访问

USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

class UserContext(BaseModel):
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的账号信息"""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"账号描述：{user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "用户不存在"

model = chatGptLLM
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="你是一个乐于助人的机器人"
)

result = agent.invoke(
    HumanMessage("我目前账号余额是多少"),
    context=UserContext(user_id="user123")
)
print(result['messages'][-1].content, 'result')

# ==========================================================================================
# 记忆（Store）

memory_store = InMemoryStore()
## 获取用户信息
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """
    获取用户信息
    Args:
        user_id: 用户id
        runtime:

    Returns:

    """
    store = runtime.store
    user_info = store.get(("users",), user_id)
    print(user_info.value, 'user_info')
    return str(user_info.value) if user_info else "找不到用户"

@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """
    保存用户信息
    Args:
        user_id: 用户id
        user_info: 更新的用户名称
        runtime:

    Returns:

    """
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "保存用户信息成功"

agent = create_agent(
    model=chatGptLLM,
    tools=[get_user_info, save_user_info],
    store=memory_store,
)

agent.invoke({
    "messages": [HumanMessage("保存这个用户的信息: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev")]
})

agent.invoke({
    "messages": [HumanMessage("通过用户id获取用户信息 userId: 'abc123'")]
})


# ================================================================================================

# Stream Writer
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"

weather_agent = create_agent(
    model=chatGptLLM,
    tools=[get_weather],
)

weather_chunk = weather_agent.stream({
    "messages": [HumanMessage("北京的天气怎么样")]
})

for chunk in weather_chunk:
    print(chunk)