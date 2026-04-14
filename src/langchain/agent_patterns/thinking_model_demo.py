#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : thinking_model_demo.py

思考模型（Reasoning Model）内容提取演示

本文件演示如何从 MiniMax-M2.7 这类自带思考能力的模型中：
1. 分离"思考过程"和"最终回答"
2. 结构化输出（JSON）
3. 流式输出思考内容和回答内容

MiniMax M2.7 的两种思考内容获取方式:
  方式 A（默认）: 思考内容嵌入在 content 中，用 <think>...</think> 标签包裹
  方式 B（推荐）: 设置 reasoning_split=True，思考内容分离到 reasoning_details 字段

LangChain 集成:
  ChatOpenAI 会将 reasoning 内容存入 AIMessage.additional_kwargs["reasoning_content"]
  也可以直接用 OpenAI SDK 获取原始 reasoning_details 字段

运行方式:
    python -m src.langchain.agent_patterns.thinking_model_demo
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


def get_thinking_model(reasoning_split: bool = True) -> ChatOpenAI:
    """
    获取 MiniMax-M2.7 模型实例。

    Args:
        reasoning_split: 是否启用思考内容分离。
            True  → 思考内容在 reasoning_details 字段（推荐）
            False → 思考内容在 content 中用 <think> 标签包裹
    """
    kwargs = {}
    if reasoning_split:
        kwargs["extra_body"] = {"reasoning_split": True}

    return ChatOpenAI(
        model="MiniMax-M2.7",
        base_url="https://api.minimax.chat/v1",
        api_key=os.getenv("MINIMAX_API_KEY"),
        timeout=120,
        max_tokens=4000,
        model_kwargs=kwargs,
    )


# ==================== 1. 基础提取：思考 vs 回答 ====================

def parse_think_tags(content: str) -> tuple[str, str]:
    """
    从 content 中解析 <think>...</think> 标签，分离思考和回答。
    适用于 reasoning_split=False 的情况。
    """
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return thinking, answer


def demo_basic_extraction():
    """
    演示 1: 基础提取 — 分离思考内容和最终回答

    两种方式对比:
    - 方式 A: 不设 reasoning_split，从 content 的 <think> 标签中解析
    - 方式 B: 设 reasoning_split=True，从 additional_kwargs 获取
    """
    question = "一个水池有两个进水管和一个出水管。进水管 A 每小时注入 3 吨水，进水管 B 每小时注入 5 吨水，出水管每小时排出 2 吨水。水池容量 48 吨，从空池开始，多久能注满？"

    print("=" * 60)
    print("演示 1: 基础提取 — 两种方式对比")
    print("=" * 60)

    # 方式 A: 从 <think> 标签提取
    print("\n--- 方式 A: 从 <think> 标签解析 ---")
    model_a = get_thinking_model(reasoning_split=False)
    response_a = model_a.invoke([HumanMessage(question)])

    thinking_a, answer_a = parse_think_tags(response_a.content)
    print(f"\n[思考过程]\n{thinking_a}")
    print(f"\n[最终回答]\n{answer_a}")

    # 方式 B: 从 reasoning_details 字段获取（推荐）
    print("\n\n--- 方式 B: reasoning_split=True（推荐）---")
    model_b = get_thinking_model(reasoning_split=True)
    response_b = model_b.invoke([HumanMessage(question)])

    # LangChain 将 reasoning_content 放在 additional_kwargs 中
    reasoning = response_b.additional_kwargs.get("reasoning_content", "")
    answer = response_b.content

    print(f"\n[思考过程] (来自 additional_kwargs['reasoning_content'])")
    print(reasoning if reasoning else "（未获取到，可能需要检查 langchain-openai 版本）")
    print(f"\n[最终回答] (来自 content)")
    print(answer)

    # 展示完整的 additional_kwargs 结构
    print(f"\n[additional_kwargs 完整结构]")
    for key, value in response_b.additional_kwargs.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")


# ==================== 2. 结构化输出（JSON） ====================

class MathSolution(BaseModel):
    """数学题的结构化解答"""
    problem_summary: str = Field(description="题目简述")
    key_conditions: list[str] = Field(description="关键条件列表")
    solution_steps: list[str] = Field(description="解题步骤")
    final_answer: str = Field(description="最终答案")
    unit: str = Field(description="答案的单位")


class TravelPlan(BaseModel):
    """旅行计划的结构化输出"""
    destination: str = Field(description="目的地")
    duration: str = Field(description="建议天数")
    highlights: list[str] = Field(description="必去景点")
    budget_estimate: str = Field(description="预算估计")
    best_season: str = Field(description="最佳旅行季节")
    tips: list[str] = Field(description="注意事项")


def demo_structured_output():
    """
    演示 2: 思考模型 + 结构化输出（JSON）

    即使是思考模型，也可以通过 with_structured_output 输出结构化 JSON。
    模型会先内部思考，然后按照 schema 格式化输出。
    """
    print("\n\n" + "=" * 60)
    print("演示 2: 思考模型 + 结构化输出")
    print("=" * 60)

    model = get_thinking_model(reasoning_split=True)

    # 示例 1: 数学题结构化解答
    print("\n--- 数学题结构化解答 ---")
    structured_model = model.with_structured_output(MathSolution)
    result = structured_model.invoke([
        HumanMessage("甲乙两人同时从 A 地出发去 B 地，甲骑车时速 15 公里，乙步行时速 5 公里。甲到达 B 地后立即返回，途中与乙相遇。A、B 两地相距 30 公里，问相遇时乙走了多少公里？")
    ])

    print(f"题目简述: {result.problem_summary}")
    print(f"关键条件:")
    for cond in result.key_conditions:
        print(f"  - {cond}")
    print(f"解题步骤:")
    for i, step in enumerate(result.solution_steps, 1):
        print(f"  {i}. {step}")
    print(f"最终答案: {result.final_answer} {result.unit}")

    # 示例 2: 旅行计划结构化输出
    print("\n--- 旅行计划结构化输出 ---")
    structured_model_2 = model.with_structured_output(TravelPlan)
    result_2 = structured_model_2.invoke([
        HumanMessage("帮我规划一个去日本京都的旅行计划")
    ])

    print(f"目的地: {result_2.destination}")
    print(f"建议天数: {result_2.duration}")
    print(f"预算估计: {result_2.budget_estimate}")
    print(f"最佳季节: {result_2.best_season}")
    print(f"必去景点:")
    for h in result_2.highlights:
        print(f"  - {h}")
    print(f"注意事项:")
    for t in result_2.tips:
        print(f"  - {t}")


# ==================== 3. 流式输出 ====================

def demo_streaming():
    """
    演示 3: 流式输出 — 实时展示思考过程和最终回答

    流式模式下思考内容和回答内容的区分:
    - reasoning_split=True 时: 思考内容在 chunk.additional_kwargs["reasoning_content"] 中
    - reasoning_split=False 时: 所有内容在 chunk.content 中，需要自行解析 <think> 标签
    """
    print("\n\n" + "=" * 60)
    print("演示 3: 流式输出 — 实时展示思考和回答")
    print("=" * 60)

    question = "请分析为什么天空是蓝色的，用物理学原理解释。"
    print(f"\n问题: {question}")

    # 方式 A: reasoning_split=False，从 content 流中识别 <think> 标签
    print("\n--- 方式 A: 流式 + <think> 标签解析 ---")
    model_a = get_thinking_model(reasoning_split=False)

    in_thinking = False
    thinking_buffer = ""
    answer_buffer = ""

    for chunk in model_a.stream([HumanMessage(question)]):
        text = chunk.content
        if not text:
            continue

        if "<think>" in text:
            in_thinking = True
            text = text.replace("<think>", "")
            if not thinking_buffer:
                print("\n[思考中] ", end="", flush=True)

        if "</think>" in text:
            in_thinking = False
            text = text.replace("</think>", "")
            thinking_buffer += text
            print(f"\n\n[开始回答] ", end="", flush=True)
            continue

        if in_thinking:
            thinking_buffer += text
            print(text, end="", flush=True)
        else:
            answer_buffer += text
            print(text, end="", flush=True)

    print(f"\n\n思考长度: {len(thinking_buffer)} 字 | 回答长度: {len(answer_buffer)} 字")

    # 方式 B: reasoning_split=True，从 additional_kwargs 流中获取
    print("\n\n--- 方式 B: 流式 + reasoning_split=True（推荐）---")
    model_b = get_thinking_model(reasoning_split=True)

    thinking_text = ""
    answer_text = ""
    phase = "unknown"

    for chunk in model_b.stream([HumanMessage(question)]):
        reasoning_chunk = chunk.additional_kwargs.get("reasoning_content", "")
        content_chunk = chunk.content or ""

        if reasoning_chunk:
            if phase != "thinking":
                phase = "thinking"
                print("\n[思考中] ", end="", flush=True)
            print(reasoning_chunk, end="", flush=True)
            thinking_text += reasoning_chunk

        if content_chunk:
            if phase != "answering":
                phase = "answering"
                print(f"\n\n[回答] ", end="", flush=True)
            print(content_chunk, end="", flush=True)
            answer_text += content_chunk

    print(f"\n\n思考长度: {len(thinking_text)} 字 | 回答长度: {len(answer_text)} 字")


# ==================== 4. 结合 Agent 使用 ====================

def demo_thinking_with_agent():
    """
    演示 4: 思考模型在 Agent 中的使用

    思考模型作为 Agent 的大脑时，会在每次工具调用决策前先进行内部推理。
    通过 reasoning_split 可以看到模型在决定调用哪个工具时的思考过程。
    """
    from langchain_core.tools import tool
    from langgraph.constants import START, END
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    print("\n\n" + "=" * 60)
    print("演示 4: 思考模型 + Agent（查看决策思考过程）")
    print("=" * 60)

    @tool
    def get_weather(city: str) -> str:
        """获取城市天气信息"""
        data = {"北京": "晴天 25°C", "上海": "多云 22°C", "广州": "小雨 28°C"}
        return data.get(city, f"暂无{city}的天气数据")

    @tool
    def get_population(city: str) -> str:
        """获取城市人口信息"""
        data = {"北京": "约 2170 万", "上海": "约 2490 万", "广州": "约 1880 万"}
        return data.get(city, f"暂无{city}的人口数据")

    tools = [get_weather, get_population]
    model = get_thinking_model(reasoning_split=True).bind_tools(tools)

    def agent_node(state: MessagesState):
        response = model.invoke(state["messages"])
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            print(f"\n  [模型思考] {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    question = "北京今天天气怎么样？人口有多少？"
    print(f"\n问题: {question}")

    result = app.invoke({"messages": [HumanMessage(question)]})

    print("\n[对话轨迹]")
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"  用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  AI→工具: {tc['name']}({tc['args']})")
            elif msg.content:
                print(f"  AI回答: {msg.content}")
        else:
            print(f"  工具返回: {msg.content}")


# ==================== 主入口 ====================

if __name__ == "__main__":
    print("MiniMax-M2.7 思考模型内容提取演示")
    print("=" * 60)

    demo_basic_extraction()
    demo_structured_output()
    demo_streaming()
    demo_thinking_with_agent()
