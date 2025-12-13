#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 15:00
@Author  : tianshiyang
@File    : human_in_the_loop.py
"""
from typing import Literal

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from provider import qwenLLM, chatGptLLM

all_users = {
    "user_123": {
        "name": "zhangsan",
        "email": "1685821150@qq.com",
        "phone": "15125228521"
    },
    "user_456": {
        "name": "lisi",
        "email": "15633777777@qq.com",
        "phone": "13023489232"
    }
}
class CustomAgentState(AgentState):
    user_id: str

@tool
def get_user_email_by_name(username):
    """
    通过用户名称，获取用户的邮件地址
    Args:
        username: 用户名称

    Returns: 返回用户的信息或None，None代表没有找到对应用户, 用户的信息为一个字典，其中name为用户名称，email为用户邮箱，phone为用户手机号
    """
    current_user = None
    for user_id, user_info in all_users.items():
        if user_info["name"] == username:
            current_user = user_info
    return current_user

class SendMessageArgs(BaseModel):
    email: str = Field(description="要发送用户的邮箱")
    message: str = Field(description="发送的内容")

class UserIntention(BaseModel):
    decisions: Literal['approve', 'reject', 'edit'] = Field(description="用户的决定, 可选approve同意/reject拒绝/edit编辑")
    edited_action: SendMessageArgs

chain = chatGptLLM.with_structured_output(UserIntention)

@tool(args_schema=SendMessageArgs)
def send_message(email: str, message: str, runtime: ToolRuntime[CustomAgentState]):
    """
    给{email}发送邮件，内容为{message}
    Args:
        email: 邮件地址
        message: 邮件内容

    Returns: None
    """
    print(email, message)
    user_id = runtime.state.get("user_id")
    user_email = all_users[user_id]["email"]
    return f"{user_email}已经给{email}, 信息为{message}"


DB_URI = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"
def human_in_the_loop():
    config = RunnableConfig(
                configurable={
                    "thread_id": "user_123"
                }
            )
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            chatGptLLM,
            tools=[get_user_email_by_name, send_message],
            state_schema=CustomAgentState,
            middleware=[
                TodoListMiddleware(),
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "send_message": InterruptOnConfig(
                            allowed_decisions=["approve", 'reject', 'edit']
                        )
                    },
                    description_prefix="请确认如下用户信息："
                )
            ],
            checkpointer=checkpointer,
        )
        result = agent.invoke(
            CustomAgentState(
                messages=[HumanMessage("请给用户名为lisi的用户发送一封邮件，内容为恭喜你通过考试")],
                user_id="user_123",
            ),
            config=config
        )
        print(result['todos'])
        if result["__interrupt__"]:
            user_result = input(result["__interrupt__"][0].value['action_requests'][0]['description'])

            user_input = chain.invoke(user_result)
            print(user_input.edited_action)
            last_result = agent.invoke(Command(
                resume={
                    "decisions": [{
                        "type": user_input.decisions,
                        "edited_action": {
                            # "name": result["__interrupt__"][0].value['action_requests'][0]['name'],
                            # "args": {"email": user_input.edited_action.email, "message": user_input.edited_action.message}
                            "name": "send_message",
                            "args": {"message": "111", "email": "hh@qq.com"}
                        }
                    }]
                }
            ), config=config)
            print(last_result["messages"])
        # for chunk in result:
        #     print(chunk)
        #
        # print(result['__interrupt__'])

        # for chunk in result:
        #     print(chunk)
        #     if isinstance(chunk, tuple) and len(chunk) == 2:
        #         message_chunk, metadata = chunk
        #         chunk_count += 1
        #         print("=" * 70)
        #         print(f"第{chunk_count}个chunk")
        #         print("=" * 70)
        #         msg_type = message_chunk.__class__.__name__
        #         if msg_type == "AIMessageChunk":
        #             # 检查是否是工具调用
        #             if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
        #                 print(f"  → AI 决定调用工具: {message_chunk.tool_calls[0]['name']}")
        #                 print(f"  → 工具参数: {message_chunk.tool_calls[0]['args']}")
        #             elif hasattr(message_chunk, 'content') and message_chunk.content:
        #                 # 实时打印内容（打字机效果）
        #                 print(message_chunk.content, end="", flush=True)
        #         elif msg_type == "ToolMessage":
        #             print(f"\n  → 工具执行结果: {message_chunk.content}")
        #             print(f"  → 工具名称: {message_chunk.name}")

if __name__ == "__main__":
    human_in_the_loop()
