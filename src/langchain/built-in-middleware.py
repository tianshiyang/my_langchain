#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/10 23:17
@Author  : tianshiyang
@File    : built-in-middleware.py
"""
from typing import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, PIIMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from provider import chatGptLLM, qwenLLM


class RequestToAI(BaseModel):
    messages: list[HumanMessage]


# =============================1.Summarization==========================
def use_summarization():
    # 总结
    config = RunnableConfig(
        configurable={
            "thread_id": "chat_conversation_001",
        }
    )
    db_uri = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        summarization_middleware = SummarizationMiddleware(
            model=chatGptLLM,
            trigger=("token", 4000),
            keep=("messages", 3)
        )
        agent = create_agent(
            chatGptLLM,
            middleware=[summarization_middleware],
            checkpointer=checkpointer,
        )

        agent.invoke(RequestToAI(
            messages=[HumanMessage("我的名字叫张三，年龄29岁")]
        ), config=config)
        agent.invoke(RequestToAI(
            messages=[HumanMessage("给我讲一下LangChain1.0应该怎么学习")]
        ), config=config)
        agent.invoke(RequestToAI(
            messages=[HumanMessage("给我讲一个关于程序员的笑话")]
        ), config=config)
        result = agent.invoke(RequestToAI(
            messages=[HumanMessage("我的名字叫什么")]
        ), config=config)

        for res in result["messages"]:
            print(res.pretty_print())


# =============================2.Human-in-the-loop==========================
def use_human_in_the_loop():
    # 人机交互 参见人机交互
    pass

# =============================3.PII检测==========================
def use_pii():
    agent = create_agent(
        qwenLLM,
        # tools=[customer_service_tool, email_tool],
        middleware=[
            # Redact emails in user input before sending to model
            PIIMiddleware(
                "email",
                strategy="mask",
                apply_to_output=True,
            ),
        ],
    )

    # When user provides PII, it will be handled according to the strategy
    result = agent.invoke({
        "messages": [{"role": "user", "content": "张三：15999999999@qq.com，我叫张三,我的邮箱是什么，"}]
    })
    print(result['messages'][-1].pretty_print())
if __name__ == '__main__':
    # use_summarization()
    use_pii()
