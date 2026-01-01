#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/31 15:24
@Author  : tianshiyang
@File    : client.py
"""
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://192.168.3.112:19530",
    token="tianshiyang:tianshiyang",
    db_name="langchain"
)
