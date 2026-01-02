#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/31 18:20
@Author  : tianshiyang
@File    : edit.py
"""
import os
import sys
import uuid

# 抑制 gRPC 警告
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from milvus import client
from utils import embeddings

COLLECTION_NAME = "books"

def get_system_files():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')

    files = os.listdir(base_path)
    return files, base_path

def get_books_documents():
    files, base_path = get_system_files()
    docs = []
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        cur_docs = UnstructuredMarkdownLoader(file_path).load()
        for cur_doc in cur_docs:
            docs.append(cur_doc)
    return docs

def get_books_chunks():
    docs = get_books_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_insert_data():
    chunks = get_books_chunks()
    data = []
    for i, chunk in enumerate(chunks[:10]):
        content = chunk.page_content
        vector = embeddings.embed_query(content)
        book_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        page_no = 1
        book_name = "llmops项目文档"
        data.append({
            "book_id": book_id,
            "page_no": page_no,
            "book_name": book_name,
            "content": content,
            "vector": vector,
            "chunk_id": chunk_id,
        })
    return data

def inset_to_milvus():
    data = get_insert_data()
    client.insert(collection_name=COLLECTION_NAME, data=data)


def search_milvus():
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=[embeddings.embed_query("llmops")],
        limit=10
    )
    print(result)

if __name__ == "__main__":
    # inset_to_milvus()
    # search_milvus()
    pass
