#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : self_ask.py

Self-Ask -- 自问自答推理模式 (LangChain 1.0+)

核心思想:
    面对复杂的多跳问题，Agent 自动将其分解为子问题，
    通过搜索工具逐个回答子问题（Intermediate Answer），
    最终综合所有中间答案得出最终结果。

来源:
    Press et al. (2022) "Measuring and Narrowing the Compositionality Gap in Language Models"

LangChain 1.0+ 实现:
    使用 create_agent + @dynamic_prompt 中间件注入 Self-Ask 指令。
    Agent 的 ReAct 循环天然支持多步工具调用，
    通过 prompt 引导模型按 "子问题→搜索→中间答案→下一个子问题" 的模式运行。

    关键技巧：通过 system_prompt 教会模型"先分解问题再逐步搜索"的行为模式。

与其他模式的区别:
    - vs ReAct: ReAct 的工具调用是通用的；Self-Ask 专注于"问题分解 + 搜索验证"
    - vs CoT: CoT 靠 LLM 内部知识推理；Self-Ask 每一步都通过外部搜索验证
    - vs Plan-and-Resolve: P&R 一次性规划所有步骤；Self-Ask 是渐进式分解

运行方式:
    python -m src.langchain.agent_patterns.self_ask
"""
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from provider import get_default_model


# ==================== 1. 搜索工具 ====================

@tool
def search(query: str) -> str:
    """
    搜索事实信息。用于回答具体的事实性子问题。
    请传入简洁明确的搜索关键词。
    """
    facts = {
        "百年孤独 作者": "《百年孤独》的作者是加布里埃尔·加西亚·马尔克斯（Gabriel García Márquez）。",
        "马尔克斯 出生": "加西亚·马尔克斯于 1927 年 3 月 6 日出生于哥伦比亚的阿拉卡塔卡（Aracataca）。",
        "阿拉卡塔卡": "阿拉卡塔卡是哥伦比亚马格达莱纳省的一个城市。",
        "特斯拉 创始人": "特斯拉（Tesla）于 2003 年由马丁·艾伯哈德和马克·塔彭宁创立。埃隆·马斯克于 2004 年加入并成为最大股东。",
        "马斯克 出生": "埃隆·马斯克于 1971 年 6 月 28 日出生于南非的比勒陀利亚。",
        "比勒陀利亚": "比勒陀利亚是南非的行政首都，位于豪登省。",
        "图灵奖 2023": "2023 年图灵奖授予了 Avi Wigderson，以表彰他在计算复杂性理论方面的贡献。",
        "Avi Wigderson": "Avi Wigderson 是以色列裔美国数学家和计算机科学家，任职于普林斯顿高等研究院。",
        "Wigderson 国籍": "Avi Wigderson 出生于以色列海法，后移居美国。",
    }
    query_lower = query.lower()
    for key, value in facts.items():
        if all(k in query_lower for k in key.split()):
            return value
    return f"搜索 '{query}' 未找到相关结果。请尝试不同的关键词。"


# ==================== 2. Self-Ask 中间件 ====================

SELF_ASK_SYSTEM_PROMPT = (
    "你是一个善于分解复杂问题的推理专家。\n\n"
    "当遇到需要多步推理的复杂问题时，请遵循以下策略：\n\n"
    "1. **判断是否需要分解**：如果问题涉及多个实体或多跳关系，需要先分解。\n"
    "2. **提出子问题**：将复杂问题拆分为简单的事实性子问题。\n"
    "3. **逐步搜索**：用 search 工具依次回答每个子问题。\n"
    "4. **综合答案**：当所有子问题都有了答案，综合得出最终结论。\n\n"
    "示例推理过程：\n"
    "问题：《百年孤独》作者出生在哪个国家？\n"
    "→ 子问题1：《百年孤独》的作者是谁？→ search(\"百年孤独 作者\") → 马尔克斯\n"
    "→ 子问题2：马尔克斯出生在哪里？→ search(\"马尔克斯 出生\") → 哥伦比亚\n"
    "→ 最终答案：哥伦比亚\n\n"
    "重要：每一步都必须通过 search 工具验证，不要凭记忆回答事实性问题。"
)

step_counter = {"count": 0}


@wrap_model_call
def trace_self_ask_steps(request: ModelRequest, handler) -> ModelResponse:
    """
    追踪 Self-Ask 的每一步推理，打印子问题和中间答案。
    """
    messages = request.state.get("messages", [])

    tool_call_count = sum(
        1 for m in messages
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls
    )

    if tool_call_count > 0:
        step_counter["count"] = tool_call_count
        print(f"  [第 {tool_call_count} 轮搜索完成，继续推理...]")

    response = handler(request)
    ai_msg = response if isinstance(response, AIMessage) else response.message

    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            print(f"  [子问题] → search(\"{tc['args'].get('query', '')}\")")
    elif ai_msg.content:
        print(f"  [最终答案] {ai_msg.content[:150]}...")

    return response


# ==================== 3. 构建 Self-Ask Agent ====================

def build_self_ask_agent():
    """
    使用 create_agent 构建 Self-Ask Agent。

    核心思路：ReAct 循环本身就支持多步工具调用，
    通过精心设计的 system_prompt 引导模型按 Self-Ask 模式运行：
    "分解问题 → 逐步搜索 → 综合答案"
    """
    return create_agent(
        get_default_model(),
        tools=[search],
        system_prompt=SELF_ASK_SYSTEM_PROMPT,
        middleware=[trace_self_ask_steps],
    )


# ==================== 4. 运行演示 ====================

def run_self_ask_demo():
    """运行 Self-Ask 演示"""
    agent = build_self_ask_agent()

    test_cases = [
        "《百年孤独》的作者出生于哪个国家的哪个城市？",
        "特斯拉公司的最大股东出生在哪个国家？",
        "2023 年图灵奖获得者是哪个国家的人？",
    ]

    for i, question in enumerate(test_cases, 1):
        step_counter["count"] = 0
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
                        print(f"  AI→搜索: search(\"{tc['args'].get('query', '')}\")")
                elif msg.content:
                    print(f"  AI总结: {msg.content}")
            else:
                print(f"  搜索结果: {msg.content}")

        final_answer = result["messages"][-1].content
        print(f"\n>>> 最终答案: {final_answer}")
        print(f">>> 总共搜索了 {step_counter['count']} 次")


if __name__ == "__main__":
    run_self_ask_demo()
