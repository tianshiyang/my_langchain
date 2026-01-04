#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/4 17:11
@Author  : tianshiyang
@File    : rag_agent.py
"""
import bs4
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from milvus import CONNECTION_ARGS
from provider import chatGptLLM
from utils import embeddings

COLLECTION_NAME = "rag_agent"

# 加载文档
def load_documents() -> list[Document]:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_spliter = RecursiveCharacterTextSplitter(chunk_overlap=200, chunk_size=1000)
    chunks = text_spliter.split_documents(docs)
    return chunks

# 获取向量数据库
def get_vector_store() -> Milvus:
    dense_index_param = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
    }
    sparse_index_param = {
        "metric_type": "BM25",
        "index_type": "AUTOINDEX",
    }
    return Milvus(
        index_params=[dense_index_param, sparse_index_param],
        collection_name=COLLECTION_NAME,
        connection_args=CONNECTION_ARGS,
        embedding_function=embeddings,
        vector_field=["dense", "sparse"],
        primary_field="id",
        builtin_function=BM25BuiltInFunction(
            function_name="bm25",
        ),
        enable_dynamic_field=True,
        auto_id=True,
    )

# 插入数据
def insert_documents(chunks: list[Document], vector_store: Milvus):
    print("开始插入数据")
    vector_store.add_documents(chunks)
    print("插入数据成功")


# 构建一个检索上下文的工具
@tool
def retrieve_context(query: str):
    """检索信息以帮助回答查询"""
    retrieved_docs = get_vector_store().similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@dynamic_prompt
def prompt_with_context(reqeust: ModelRequest):
    """Inject context into state messages."""
    last_query = reqeust.messages[-1].text
    retrieved_docs = get_vector_store().similarity_search(last_query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

if __name__ == "__main__":
    chunks = load_documents()
    vector_store = get_vector_store()
    # insert_documents(chunks, vector_store)

    # 方式1
    # tools = [retrieve_context]
    # prompt = (
    #     "You have access to a tool that retrieves context from a blog post. "
    #     "Use the tool to help answer user queries."
    # )
    # agent = create_agent(chatGptLLM, system_prompt=prompt, tools=tools)
    #

    # 方式2.
    agent = create_agent(chatGptLLM, tools=[], middleware=[prompt_with_context])

    query = "What is task decomposition?"
    for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
