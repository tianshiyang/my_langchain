#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/3 19:54
@Author  : tianshiyang
@File    : base_use.py
"""
from langchain_core.documents import Document
from langchain_milvus import Milvus

from milvus import CONNECTION_ARGS
from utils import embeddings
import uuid

docs = [
    Document(page_content="i worked at kensho", metadata={"namespace": "harrison"}),
    Document(page_content="i worked at facebook", metadata={"namespace": "ankush"}),
]

vector_store = Milvus(
    connection_args=CONNECTION_ARGS,
    embedding_function=embeddings,
    collection_name="partitioned_collection",
    partition_key_field="namespace",
    text_field="content",
    primary_field="id"
)

def add_document():
    vector_store.add_documents(
        docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

def search_document():
    result = vector_store.as_retriever(
        search_kwargs={"expr": 'namespace == "ankush"'}
    ).invoke("where did i work")
    print(result)

search_document()