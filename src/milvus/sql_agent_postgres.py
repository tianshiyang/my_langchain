#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/5 14:59
@Author  : tianshiyang
@File    : sql_agent_postgres.py
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from provider import chatGptLLM

SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@127.0.0.1:5432/llmops?client_encoding=utf8"

db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URI)

toolkit = SQLDatabaseToolkit(db=db, llm=chatGptLLM)

tools = toolkit.get_tools()

system_prompt = """
你是一个用于与 SQL 数据库交互的智能体。

给定一个输入问题，请生成一个语法正确的 {dialect} 查询语句，

然后查看查询结果并返回答案。除非用户明确指定了希望获取的示例数量，

否则你的查询结果最多只应返回 {top_k} 条记录。

你可以根据相关列对结果进行排序，以返回数据库中最相关的示例。

永远不要查询某个表的所有列，而只应请求与问题相关的列。

在执行查询前，你必须仔细检查你的查询语句。

如果在执行查询时遇到错误，请重写查询语句并重试。

严禁对数据库执行任何数据操作语言（DML）语句（如 INSERT、UPDATE、DELETE、DROP 等）。

首先，你应当始终先查看数据库中有哪些表，以便了解可以查询的内容。

这一步绝不能跳过。

接下来，你应该查询与问题最相关表的结构（schema）
""".format(
    top_k=5,
    dialect=db.dialect,
)

content = "message这个表里，用户都提问过哪些问题"

with PostgresSaver.from_conn_string("postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8") as checkpointer:
    config = RunnableConfig(
        configurable={
            "thread_id": 'user_001'
        }
    )

    agent = create_agent(
        chatGptLLM,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"sql_db_query": True},
                description_prefix="工具调用需要审核",
            ),
        ]
    )

    for step in agent.stream(
        {"messages": HumanMessage(content=content)},
        stream_mode="values",
        config=config
    ):
        if '__interrupt__' in step:
            interrupt = step["__interrupt__"][0]
            print(interrupt)
        elif "messages" in step:
            print(step["messages"][-1].pretty_print())

    print("*" * 20 + "第二轮" + "*" * 20)
    for step in agent.stream(
        Command(resume={
            "decisions": [{
                "type": "approve"
            }]
        }),
        stream_mode="values",
        config=config
    ):
        if '__interrupt__' in step:
            interrupt = step["__interrupt__"][0]
            print(interrupt)
        elif "messages" in step:
            print(step["messages"][-1].pretty_print())

