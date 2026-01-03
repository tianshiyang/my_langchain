#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/3 21:10
@Author  : tianshiyang
@File    : rag.py
"""
import uuid

import bs4
from document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, ConfigurableField
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus

from milvus import CONNECTION_ARGS
from provider import chatGptLLM
from utils import embeddings


# 1. 准备数据
def prepare_data() -> list[Document]:
    loader = WebBaseLoader(
        web_path=(
            # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )

    loader_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    chunks = text_splitter.split_documents(loader_documents)

    return chunks

def insert_data(chunks: list[Document]):
    def _insert_data(doc: Document):
        doc.metadata['content'] = doc.page_content
        return doc
    Milvus.from_documents(
        documents=list(map(_insert_data, chunks)),
        ids=[str(uuid.uuid4()) for _ in range(len(chunks))],
        embedding=embeddings,
        connection_args=CONNECTION_ARGS,
        collection_name="milvus_rag",
        primary_field="id",
    )

def get_vector_store() -> Milvus:
    return Milvus(
        connection_args=CONNECTION_ARGS,
        collection_name="milvus_rag",
        embedding_function=embeddings,
        primary_field="id",
        text_field="content"
    )

def similarity_search(vector: Milvus):
    result = vector.similarity_search(
        query="What is self-reflection of an AI Agent?",
        k=1
    )
    return result

def format_docs(chunks: list[Document]):
    return "\n\n".join(chunk.page_content for chunk in chunks)

if __name__ == "__main__":
    # documents = prepare_data()
    # insert_data(documents)
    # 获取向量数据库
    vector_store = get_vector_store()
    # search_result = similarity_search(vector_store)

    llm = chatGptLLM
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # 带搜索条件的retriever
    retriever = vector_store.as_retriever().configurable_fields(
        search_kwargs=ConfigurableField(
            id="retriever_search_kwargs",
        )
    ).with_config(
        configurable={
            "retriever_search_kwargs": dict(
                # expr="source == 'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/'",
                expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
                k=1,
            )
        }
    )

    chain = ({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser())


    res = chain.invoke("What is self-reflection of an AI Agent??")
    print(res)
