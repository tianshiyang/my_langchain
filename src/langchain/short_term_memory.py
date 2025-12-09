#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/8 21:20
@Author  : tianshiyang
@File    : short_term_memory.py
"""
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware, dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolRuntime
from langchain.tools import tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from provider import chatGptLLM

class CustomAgentState(AgentState):
    user_id: str
    user_name: str
    preference: dict

class CustomContext(BaseModel):
    user_id: str
    user_name: str

user_info = {
    "user_123": {
        "username": "李四",
        "email": "1685821150@qq.com",
        "sex": "男"
    }
}

@tool
def get_user_info(runtime: ToolRuntime[CustomAgentState]) -> str:
    """
    获取用户信息
    Args:
        runtime:

    Returns:

    """
    cur_user_id = runtime.state.get("user_id")
    cur_user_info = user_info[cur_user_id]
    return f"用户名: {cur_user_info['username']}, 邮箱：{cur_user_info['email']}，性别：{cur_user_info['sex']}"

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomAgentState]) -> Command:
    """
    查看和更新用户信息
    Args:
        runtime:

    Returns:
    """
    user_id = runtime.context.user_id
    name = "王五" if user_id == "user_123" else "找不到用户"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage("更新用户信息成功", tool_call_id=runtime.tool_call_id)
        ]
    })

@tool
def greet(runtime: ToolRuntime[CustomContext, CustomAgentState]) -> Command | str:
    """一旦您找到用户的信息，就用它来问候用户"""
    user_name = runtime.state['user_name']
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "请使用update_user_info去更新用户的姓名",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"你好{user_name}"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

DB_URI = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()

    # Trim messages修剪消息
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        裁剪用户消息
        Args:
            state:
            runtime:

        Returns:

        """
        messages = state.get("messages")
        if len(messages) <= 3:
            return None # 不需要做改变
        first_msg = messages[0]
        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    agent = create_agent(
        chatGptLLM,
        tools=[get_user_info],
        middleware=[
            trim_messages,
            SummarizationMiddleware( # 摘要总结中间件
                model=chatGptLLM,
                trigger=("tokens", 4000),
                keep=20
            ),
            dynamic_system_prompt
        ],
        state_schema=CustomAgentState,
        context_schema=CustomContext,
        checkpointer=checkpointer
    )

    """第一条消息"""
    # agent.invoke(
    #     {
    #         "messages": [{"role": "user", "content": "我的名字是张三，我喜欢的主题色是黑色"}],
    #         "user_id": "user_123",
    #         "preferences": {"theme": "dark"}
    #     },
    #     {"configurable": {"thread_id": "1"}})

    """第二条消息"""
    result = agent.invoke(
        {
            "messages": [HumanMessage("我的邮箱是什么呢？")],
            "user_id": "user_123",
            "preferences": {"theme": "dark"}
        },
        config=RunnableConfig(
            configurable={
                "thread_id": "1",
            }
        ),
        context=CustomContext(user_name="John Smith")
    )

    print(result['messages'][-1].content)
