#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/7 20:23
@Author  : tianshiyang
@File    : agent.py
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call, wrap_tool_call, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage, HumanMessage
from pydantic import BaseModel

from provider.llms import qwenLLM, chatGptLLM
from langchain.tools import tool

# 定义工具
@tool
def search(query: str) -> str:
    """根据信息进行搜索"""
    return f"搜索：{query}"

@tool
def get_weather(location: str) -> str:
    """获取传入位置的天气信息"""
    return f"{location}的天气是：晴天，36摄氏度"

# Dynamic model 动态模型

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据对话复杂度选择模型
    Args:
        request:
        handler:

    Returns:

    """
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = qwenLLM
    else:
        model = chatGptLLM

    return handler(request.override(model=model))

# 定义工具的错误处理
@wrap_tool_call
def handle_tool_errors(request: ModelRequest, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        handler(request)
    except Exception as e:
        # 返回自定义的错误信息给model
        return ToolMessage(
            content=f"工具调用错误: 请检查你的输入，并且再次尝试.{str(e)}",
            tool_call_id=request.state["messages"][-1].tool_call_id, # 这行代码可能有问题
            # tool_call_id=request.tool_call["id"] # 这个是官网的
        )

class Context(BaseModel):
    user_role: str

# dynamic system prompt 动态系统提示：对于需要根据运行时上下文或agent状态修改系统提示符的高级用例，可以使用middleware中间件
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """通过用户角色生成系统提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = '你是一个乐于助人的机器人'
    if user_role == "user":
        return f"{base_prompt} 提供详细的技术答复"
    elif user_role == "beginner":
        return f"{base_prompt} 简单地解释概念并避免使用行话."
    return base_prompt

agent = create_agent(
    model=qwenLLM, # 默认模型
    tools=[get_weather, search],
    middleware=[dynamic_model_selection, user_role_prompt],
    system_prompt="你是一个乐于助人的AI助手",
    context_schema=Context,
)

result = agent.invoke(HumanMessage("解释下机器学习"), context=Context(user_role="expert"))


# 格式化输出
## ToolStrategy工具策略
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
## ProviderStrategy使用模型提供者的原生结构化输出生成，仅适用于支持原生结构化输出的提供者如OpenAI

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

