#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/8 19:07
@Author  : tianshiyang
@File    : custom_agent_langgraph.py
"""
from typing import Literal

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import convert_to_messages, HumanMessage
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from IPython.display import Image, display

from milvus import CONNECTION_ARGS
from provider import chatGptLLM
from utils import embeddings

def get_milvus():
    return Milvus(
        collection_name="custom_agent_langgraph",
        connection_args=CONNECTION_ARGS,
        embedding_function=embeddings,
        enable_dynamic_field=True,
        auto_id=True,
        primary_field="id",
    )


def load_documents():
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sub_doc in docs for item in sub_doc]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    get_milvus().add_documents(doc_splits)

def get_retriever():
    return get_milvus().as_retriever()

# 2.创建检索工具
@tool
def retrieve_blog_posts(query: str) -> str:
    """搜索并返回有关 Lilian Weng 博客文章的信息。"""
    docs = get_retriever().invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts

# 3.生成查询
response_model = chatGptLLM

def generate_query_or_respond(state: MessagesState):
    """调用模型，根据当前状态生成响应。
    给定用户的问题，模型将决定是使用检索工具（retriever tool）进行信息检索，还是直接回答用户。"""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

# 4.对文件进行评分
GRADE_PROMPT = (
    "你是一个评估者，负责判断检索到的文档是否与用户问题相关。\n"
    "以下是检索到的文档：\n\n {context} \n\n"
    "以下是用户的问题：{question} \n"
    "如果文档包含与用户问题相关的关键词或语义内容，则将其评为相关。\n"
    "请给出一个二元评分：'yes'（是）或 'no'（否），以表明该文档是否与问题相关。"
)

class GradeDocument(BaseModel):
    """使用二元评分（“yes”或“no”）对文档进行相关性评估。"""
    binary_score: str = Field(description="相关性评分：若相关则为 'yes'，不相关则为 'no'。")

grader_model = chatGptLLM

def grade_documents(
        state: MessagesState
) -> Literal["generate_answer", "rewrite_question"]:
    """判断检索到的文档是否与问题相关。"""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(context=context, question=question)
    response = grader_model.with_structured_output(GradeDocument).invoke([{"role": "user", "content": prompt}])
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# 重写问题
REWRITE_PROMPT = (
    "请分析输入内容，尝试理解其背后的语义意图或含义。\n"
    "以下是初始问题：\n"
    "\n ------- \n"
    "{question}\n"
    "\n ------- \n"
    "请重新表述为一个更清晰、更准确的问题："
)
def rewrite_question(state: MessagesState):
    """重写用户的原始问题。"""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

# 生成答案
GENERATE_PROMPT = (
    "你是一个问答任务助手。"
    "请使用以下检索到的上下文信息来回答问题。"
    "如果你不知道答案，就直接说明你不知道。"
    "回答最多使用三句话，并保持简洁。\n"
    "问题：{question} \n"
    "上下文：{context}"
)

def generate_answer(state: MessagesState):
    """生成答案"""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# 编排agent
workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
{
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
graph = workflow.compile()

if __name__ == "__main__":
    # load_documents()
    # 测试检索工具
    # result = retriever_tool.invoke({"query": "types of reward hacking"})
    # print(result)
    # 测试生成查询
    # input = {
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "What does Lilian Weng say about types of reward hacking?",
    #         }
    #     ]
    # }
    # generate_query_or_respond(input)["messages"][-1].pretty_print()
    # 测试相关性
    # input = {
    #     "messages": convert_to_messages(
    #         [
    #             {
    #                 "role": "user",
    #                 "content": "What does Lilian Weng say about types of reward hacking?",
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": "",
    #                 "tool_calls": [
    #                     {
    #                         "id": "1",
    #                         "name": "retrieve_blog_posts",
    #                         "args": {"query": "types of reward hacking"},
    #                     }
    #                 ],
    #             },
    #             {
    #                 "role": "tool",
    #                 "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
    #                 "tool_call_id": "1",
    #             },
    #         ]
    #     )
    # }
    # print(grade_documents(input))

    # 测试重写用户问题
    # input = {
    #     "messages": convert_to_messages(
    #         [
    #             {
    #                 "role": "user",
    #                 "content": "What does Lilian Weng say about types of reward hacking?",
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": "",
    #                 "tool_calls": [
    #                     {
    #                         "id": "1",
    #                         "name": "retrieve_blog_posts",
    #                         "args": {"query": "types of reward hacking"},
    #                     }
    #                 ],
    #             },
    #             {"role": "tool", "content": "meow", "tool_call_id": "1"},
    #         ]
    #     )
    # }
    #
    # response = rewrite_question(input)
    # print(response["messages"][-1].content)
    pass