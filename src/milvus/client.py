#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/31 15:24
@Author  : tianshiyang
@File    : client.py
"""
import os
import warnings

# 抑制警告
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
warnings.filterwarnings("ignore", message=".*AsyncMilvusClient.*")

from langchain_milvus import Milvus

from utils import embeddings

TOKEN = "tianshiyang:tianshiyang"

URI = "http://localhost:19530"

client = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "langchain"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    collection_name="books",
    text_field="content",  # 指定文本字段名
    drop_old=False,  # set to True if seeking to drop the collection with that name if it exists
)

