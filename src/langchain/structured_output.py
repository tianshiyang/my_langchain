#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/10 00:13
@Author  : tianshiyang
@File    : structured_output.py
"""
from typing import List

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, EmailStr

from provider import chatGptLLM


class ContactInfo(BaseModel):
    """用户的合同信息"""
    name: str = Field(description="用户名称")
    email: str = Field(description="用户邮箱")
    phone: str = Field(description="用户手机号")

class Request(BaseModel):
    messages: List[HumanMessage]

# ===============1.使用提供商策略=================
def use_provider_strategy():
    agent = create_agent(
        chatGptLLM,
        response_format=ProviderStrategy(ContactInfo)
    )

    request = agent.invoke({
        "messages": [HumanMessage("从以下地址获取联系信息：John Doe， john@example.com, (555) 123-4567")]
    })

    print(request["structured_response"].name)

# ===============2.使用工具策略=================
def use_tool_strategy():
    agent = create_agent(
        chatGptLLM,
        response_format=ToolStrategy(ContactInfo)
    )

    # request = agent.invoke({
    #     "messages": [HumanMessage("从以下地址获取联系信息：John Doe， john@example.com, (555) 123-4567")]
    # })
    request = agent.invoke(Request(messages=[HumanMessage("从以下地址获取联系信息：John Doe， john@example.com, (555) 123-4567")]))

    print(request["structured_response"])

if __name__ == "__main__":
    # use_provider_strategy()
    use_tool_strategy()