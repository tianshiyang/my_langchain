#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/22 23:21
@Author  : tianshiyang
@File    : index.py
"""
import os
import uuid

from langchain_milvus import Milvus
import dotenv
from langchain_core.documents import Document

from utils import embeddings

dotenv.load_dotenv()

CONNECTION_ARGS = {
    "uri": os.getenv("MILVUS_URI"),
    "db_name": os.getenv("MILVUS_DB_NAME"),
    "token": os.getenv("MILVUS_TOKEN"),
}

CONNECTION_NAME = "books"

test_documents = [
    Document(
        page_content="这是第一条测试数据",
        metadata={
            "book_id": str(uuid.uuid4()),
            "book_name": "测试数据的book_name",
            "content": "这是第3条测试数据",
            "chunk_id": str(uuid.uuid4()),
        }
    )
]

# 获取Milvus store
vector_store = Milvus(
    connection_args=CONNECTION_ARGS,
    collection_name=CONNECTION_NAME,
    embedding_function=embeddings,
    consistency_level="Strong",  # 设置强一致性，确保删除立即生效
    text_field="content",
    primary_field="id",  # 指定主键字段名，MMR 搜索需要
)

# 往Milvus中插入数据
def insert_documents():
    vector_store.add_documents(
        test_documents,
        ids=[str(uuid.uuid4()) for _ in range(len(test_documents))],
    )

def delete_documents():
    # 1. 通过ID删除
    # result = vector_store.delete(
    #     ids=["95d98e6d-d7b8-4c7a-a7d9-8cc9bf18bda6"]
    # )
    # 2. 通过表达式删除（字符串值需要用引号包裹）
    # vector_store.delete(
    #     expr='content == "这是第3条测试数据"'
    # )
    # 3.使用like语句
    vector_store.delete(
        expr='book_name like "%book_name"'
    )
    # print(result)

# 搜索 1. 直接查询
def similarity_search():
    # 1. 直接查询
    result = vector_store.similarity_search_with_score(
        "第三条",
        k=10,
        expr='book_name == "测试数据的book_name"'
    )
    for doc, score in result:
        print("#"*20)
        print(f"得分: {score}")
        print(doc.page_content)

def retriever_search():
    # mmr 需要 collection 主键名为 'id'，如果不是，使用 similarity 类型
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "expr": 'book_name like "%book_name"'  # 使用 expr，不是 filter
        }
    )
    result = retriever.invoke("第三条")
    for doc in result:
        print("#" * 20)
        print(doc)

if __name__ == "__main__":
    # insert_documents()
    # delete_documents()
    # similarity_search()
    retriever_search()