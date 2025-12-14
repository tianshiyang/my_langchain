#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/13 22:57
@Author  : tianshiyang
@File    : runtime.py
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime

from provider import chatGptLLM
from langchain.tools import tool

users = [
    {
        "user_id": "user_001",
        "username": "张三"
    }, {
        "user_id": "user_002",
        "username": "李四"
    }
]

@dataclass
class Context:
    user_id: str

@tool
def get_user_info(runtime: ToolRuntime[Context]):
    """
    获取用户信息，当你需要获取用户的信息的时候，可以调用此方法
    Returns: 用户基本信息，包括username：用户名, user_id: 用户id

    """
    print(runtime.store)
    print(runtime.context)
    print(runtime.state.get('user_like'))
    return list(filter(lambda user: user["user_id"] == runtime.context.user_id, users))[0]

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    # print(f"request.tools: {request.tools}")
    # print(f"request.system_prompt: {request.system_prompt}")
    # print(f"request.tool_choice: {request.tool_choice}")
    user_id = request.runtime.context.user_id
    username = list(filter(lambda user: user["user_id"] == user_id, users))[0]['username']
    system_prompt = f"你是一个乐于助人的AI机器人，你喜欢帮助{username}"
    return system_prompt

def use_context():
    agent = create_agent(
        chatGptLLM,
        tools=[get_user_info],
        middleware=[dynamic_system_prompt],
        context_schema=Context,
    )
    result = agent.invoke({
        "messages": [HumanMessage("我的名字叫什么")],
    }, context=Context(
        user_id="user_001",
    ))

    print(result['messages'][-1].pretty_print())

if __name__ == "__main__":
    use_context()