#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : react_agent.py

ReAct (Reason + Act) -- 推理与行动模式 (LangChain 1.0+)

核心思想:
    LLM 交替进行"推理"和"行动"：
    Thought → Action → Observation → Thought → ... → Final Answer

来源:
    Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"

LangChain 1.0+ 实现:
    create_agent 本身就是 ReAct 模式！
    它在内部构建了一个 LangGraph 图，自动完成：
    - 模型推理 → 决定是否调用工具 → 执行工具 → 回到模型推理
    - 可通过 middleware 拦截每一步（wrap_model_call, wrap_tool_call）
    - 支持 stream 查看完整的 Thought/Action/Observation 轨迹

与其他模式的区别:
    - vs CoT: CoT 只推理不行动；ReAct 可以调用外部工具获取真实信息
    - vs Plan-and-Resolve: ReAct 是"走一步看一步"，边做边想；P&R 先规划全局再执行
    - vs Self-Ask: Self-Ask 专注于问题分解；ReAct 更通用，工具类型不限

运行方式:
    python -m src.langchain.agent_patterns.react_agent
"""
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from provider import get_default_model


# ==================== 1. 定义工具集 ====================

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，获取事实信息"""
    knowledge_base = {
        "python": "Python 由 Guido van Rossum 于 1991 年发布，最新稳定版本是 3.13。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，1.0 版本引入了 create_agent 和 middleware 体系。",
        "langgraph": "LangGraph 是 LangChain 生态中用于构建有状态多步 Agent 工作流的库，1.0 版本与 LangChain 深度集成。",
        "react": "ReAct 模式由 Yao et al. 2022 年提出，结合推理和行动来解决复杂任务。",
    }
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return f"未找到关于 '{query}' 的信息。请尝试更具体的关键词。"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '2 + 3 * 4'"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_current_date() -> str:
    """获取当前日期"""
    from datetime import date
    return f"当前日期是 {date.today().isoformat()}"


# ==================== 2. 中间件：拦截 ReAct 循环的每一步 ====================

@wrap_model_call
def trace_reasoning(request: ModelRequest, handler) -> ModelResponse:
    """
    拦截模型调用，打印 Agent 的推理轨迹。

    这个中间件让你能"看到" ReAct 循环中模型的每一次思考。
    在生产环境中，可以替换为日志记录或 LangSmith tracing。
    """
    msg_count = len(request.state.get("messages", []))
    print(f"\n  [Thought] 模型正在推理（当前消息数: {msg_count}）...")

    response = handler(request)
    ai_msg = response if isinstance(response, AIMessage) else response.message

    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            print(f"  [Action] 决定调用工具: {tc['name']}({tc['args']})")
    elif ai_msg.content:
        print(f"  [Final Answer] {ai_msg.content[:100]}{'...' if len(ai_msg.content) > 100 else ''}")

    return response


@wrap_tool_call
def trace_observation(request, handler):
    """
    拦截工具调用，打印 Observation（工具返回结果）。

    同时演示了如何在 wrap_tool_call 中做错误处理。
    """
    try:
        result = handler(request)
        content = result.content if hasattr(result, "content") else str(result)
        print(f"  [Observation] {content}")
        return result
    except Exception as e:
        print(f"  [Observation] 工具执行失败: {e}")
        return ToolMessage(
            content=f"工具调用出错: {e}，请检查输入后重试。",
            tool_call_id=request.tool_call["id"],
        )


# ==================== 3. 构建 ReAct Agent ====================

def build_react_agent():
    """
    使用 create_agent 构建标准 ReAct Agent。

    create_agent 内部就是 ReAct 循环：
    model_node → (有 tool_calls?) → tools_node → model_node → ... → END

    通过 middleware 可以拦截循环的每一步。
    """
    return create_agent(
        get_default_model(),
        tools=[search_knowledge, calculator, get_current_date],
        system_prompt=(
            "你是一个乐于助人的 AI 助手。\n"
            "遇到需要查询信息的问题，请使用 search_knowledge 工具。\n"
            "遇到需要计算的问题，请使用 calculator 工具。\n"
            "遇到需要日期的问题，请使用 get_current_date 工具。\n"
            "你可以连续调用多个工具来完成复杂任务。"
        ),
        middleware=[trace_reasoning, trace_observation],
    )


# ==================== 4. 运行并展示轨迹 ====================

def run_react_demo():
    """运行 ReAct Agent 并打印完整的 Thought/Action/Observation 轨迹"""
    agent = build_react_agent()

    test_cases = [
        "LangChain 是什么？它和 LangGraph 有什么关系？",
        "Python 是哪年发布的？距今多少年了？请帮我算一下，今天是几号？",
        "帮我计算 (100 - 37) * 2 + 15 的结果",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {question}")
        print('=' * 60)

        result = agent.invoke({"messages": [HumanMessage(question)]})

        print(f"\n[完整对话轨迹]")
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"  用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  AI→工具: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"  AI回答: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"  工具返回: {msg.content}")


# ==================== 5. 流式 ReAct（打字机效果）====================

def run_react_streaming():
    """
    流式展示 ReAct 循环。

    使用 stream_mode="messages" 可以逐 token 看到 Agent 的推理过程。
    """
    agent = build_react_agent()
    question = "LangChain 是什么？帮我算一下 2024 - 1991 等于多少"

    print(f"\n{'=' * 60}")
    print(f"流式 ReAct: {question}")
    print('=' * 60)

    for chunk in agent.stream(
        {"messages": [HumanMessage(question)]},
        stream_mode="messages",
    ):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            msg_chunk, metadata = chunk
            msg_type = msg_chunk.__class__.__name__

            if msg_type == "AIMessageChunk":
                if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                    print(f"\n[调用工具] {msg_chunk.tool_calls[0]['name']}", end="")
                elif msg_chunk.content:
                    print(msg_chunk.content, end="", flush=True)
            elif msg_type == "ToolMessage":
                print(f"\n[工具返回] {msg_chunk.content}")

    print()


if __name__ == "__main__":
    run_react_demo()
    run_react_streaming()
