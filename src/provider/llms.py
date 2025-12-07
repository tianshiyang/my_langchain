#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 16:24
@Author  : tianshiyang
@File    : llm.py
"""
import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# chatGpt的大语言模型
chatGptLLm = ChatOpenAI(
    base_url=os.getenv('XIAO_AI_BASE_UR'),
    api_key=os.getenv('XIAO_AI_API_KEY'),
)