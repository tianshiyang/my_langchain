#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 16:24
@Author  : tianshiyang
@File    : llms.py
"""
import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_qwq import ChatQwen

load_dotenv()

# chatGpt的大语言模型
chatGptLLM = ChatOpenAI(
    model="gpt-4o",
    base_url=os.getenv('XIAO_AI_BASE_URL'),
    api_key=os.getenv('XIAO_AI_API_KEY'),
    temperature=0,
    timeout=10,
    max_tokens=1000
)

# 阿里千问模型
qwenLLM = ChatQwen(
    model="qwen3-max",
    base_url=os.getenv('QWEN_BASE_URL'),
    api_key=os.getenv('QWEN_API_KEY'),
)

google_gemini = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
)