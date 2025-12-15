#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/14 18:33
@Author  : tianshiyang
@File    : context_engineering.py
"""
from dataclasses import dataclass
from typing import Callable, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field

from provider import qwenLLM

config = RunnableConfig(
    configurable={
        "thread_id": "chat_conversation_001",
    }
)

@dataclass
class Context:
    user_id: str

namespace = ("writing_style", )

@dynamic_prompt
def use_dynamic_system_prompt(request: ModelRequest):
    return "你是谁"

db_uri = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"

def use_state_context():
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        # ==================================1.系统提示词======================================
        agent = create_agent(
            qwenLLM,
            middleware=[use_dynamic_system_prompt],
            system_prompt="你是一个乐于助人的机器人",
            checkpointer=checkpointer,
        )
        result = agent.invoke({
            "messages": [HumanMessage("你是那个模型")],
        }, config=config)
        print(result['messages'][-1].pretty_print())

@wrap_model_call
def inject_writing_style(request: ModelRequest, handler: Callable[[ModelRequest], ModelRequest]):
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    writing_style = store.get(namespace, user_id)

    if writing_style:
        style = writing_style.value
        style_context = f"""你的创作风格：
            - 语气: {style.get('tone', 'professional')}
            - 问候语: "{style.get('greeting', 'Hi')}"
            - 签名: "{style.get('sign_off', 'Best')}"
        """
        messages = [
            *request.messages,
            HumanMessage(style_context)
        ]
        request = request.override(messages=messages)
    print(f"request.messages: {request.messages}")
    return handler(request)

class WritingInfo(BaseModel):
    tone: Literal["admin", "user", "guest"] = Field(description="用户喜欢的写作风格")
    greeting: str = Field(description="问候语")
    sign_off: str = Field(description="用户的签名")

class UserInfo(BaseModel):
    user_id: str = Field(description="用户id")
    username: str = Field(description="用户名")
    email: str = Field(description="用户邮箱")

all_users: list[UserInfo] = [
    UserInfo(
        user_id="user_id_001",
        username="张三",
        email="zhangsan@qq.com"
    ),
    UserInfo(
        user_id="user_id_002",
        username="李四",
        email="lisi@qq.com"
    ),
]

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> UserInfo:
    """
    获取用户信息
    Args:

    Returns: 返回用户的信息，包含

    """
    user_id = runtime.context.user_id
    return list(filter(lambda user: user.user_id == user_id, all_users))[0]

@tool
def extract_and_save_user_writing(query: str, runtime: ToolRuntime[Context]):
    """
    当你需要保存或更新用户的写作风格信息的时候，可以调用此工具
    Args:
        query: 用户的输入

    Returns:

    """
    chain = qwenLLM.with_structured_output(WritingInfo)
    result = chain.invoke(query)
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(namespace, user_id, {
        "tone": result.tone,
        "greeting": result.greeting,
        "sign_off": result.sign_off,
    })
    return "保存用户写作喜好成功！"


def use_store_context():
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        with PostgresStore.from_conn_string(db_uri) as store:
            checkpointer.setup()
            store.setup()

            agent = create_agent(
                qwenLLM,
                tools=[get_user_info, extract_and_save_user_writing],
                middleware=[inject_writing_style],
                checkpointer=checkpointer,
                store=store,
                context_schema=Context
            )
            # agent.invoke({
            #     "messages": [HumanMessage("请保存我喜欢的写作风格：语气是幽默，签名需要获取我当前的名字，问候语为你好")]
            # },
            #     context=Context(user_id="user_id_001"),
            #     config=config)
            result = agent.invoke(
                {
                    "messages": [HumanMessage("请写一个关于程序员的冷笑话")]
                },
                context=Context(user_id="user_id_001"),
                config=config
            )
            # print(agent.get_state(config=config))
            print(result['messages'][-1].pretty_print())

if __name__ == '__main__':
    # use_state_context()
    use_store_context()
