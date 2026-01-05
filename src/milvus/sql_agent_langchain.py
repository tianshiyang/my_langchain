#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/5 11:40
@Author  : tianshiyang
@File    : sql_agent_langchain.py
"""
import requests, pathlib
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from provider import chatGptLLM

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


# 配置数据库
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# print(f"语言: {db.dialect}")
# print(f"存在的表: {db.get_usable_table_names()}")
# print(f'简单输出: {db.run("SELECT * FROM Artist LIMIT 5;")}')

# 添加数据库交互工具
toolkit = SQLDatabaseToolkit(db=db, llm=chatGptLLM)

tools = toolkit.get_tools()

# 使用agent
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

接下来，你应该查询与问题最相关表的结构（schema）。
""".format(
    dialect=db.dialect,
    top_k=5,
)

checkpointer = InMemorySaver()

config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    chatGptLLM,
    system_prompt=system_prompt,
    tools=tools,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="sql工具等待审核"
        )
    ],
    checkpointer=checkpointer,
)

question = "哪个流派的曲目平均时长最长？"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    config=config,
):
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass

for step in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass