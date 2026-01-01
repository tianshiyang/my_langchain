#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/31 17:47
@Author  : tianshiyang
@File    : database.py
"""
from milvus import client

client.create_database(
    db_name="langchain"
)