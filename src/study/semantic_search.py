#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/2 23:04
@Author  : tianshiyang
@File    : semantic_search.py
"""
import os.path
import uuid

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from milvus import CONNECTION_ARGS
from utils import embeddings

file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "科大讯飞财报.pdf")

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=CONNECTION_ARGS,
    primary_field="id",
    collection_name="financial_report",
    text_field="content",
)

def load_pdf():
    loader = PDFMinerLoader(file_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=50,
        chunk_size=100,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(
        documents,
    )
    return chunks

def insert_document_to_milvus(document: list[Document]):
    def set_content(chunk: Document):
        chunk.metadata['content'] = chunk.page_content
        return chunk

    vector_store.add_documents(
        list(map(set_content, document)),
        ids=[str(uuid.uuid4()) for _ in range(len(document))]
    )

def similarity_search():
    result = vector_store.similarity_search(
        "财务报表"
    )
    for doc in result:
        print(doc)
        print("=====")

def similarity_search_by_vector():
    result = vector_store.similarity_search_by_vector(
        embeddings.embed_query("财务报表"),
        expr='id == "07c71d15-c70b-43ff-b22e-2d34a7ccf35e"'
    )
    for doc in result:
        print("#" * 20)
        print(doc)

def search_by_retriever():
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    docs = retriever.invoke("财务报表", expr='id == "07c71d15-c70b-43ff-b22e-2d34a7ccf35e"')
    for doc in docs:
        print("#" * 30)
        print(doc)

if __name__ == "__main__":
    # documents = load_pdf()
    # insert_document_to_milvus(documents)
    # similarity_search()
    # similarity_search_by_vector()
    search_by_retriever()
