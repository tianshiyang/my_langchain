#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/24 02:14
@Author  : tianshiyang
@File    : embeddings.py
"""
import os

import dotenv
from langchain_community.embeddings import DashScopeEmbeddings

dotenv.load_dotenv()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)