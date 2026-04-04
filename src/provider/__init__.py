#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 16:24
@Author  : tianshiyang
@File    : __init__.py.py
"""
from .llms import chatGptLLM, qwenLLM, google_gemini, minimax_llm, get_default_model

__all__ = [
    "chatGptLLM",
    "qwenLLM",
    "google_gemini",
    "minimax_llm",
    "get_default_model",
]