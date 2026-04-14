#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : chain_of_thought.py

Chain-of-Thought (CoT) -- 思维链推理模式 (LangChain 1.0+)

核心思想:
    通过 prompt 引导 LLM 在给出最终答案前，先输出逐步推理过程。
    这是最基础的推理增强手段，不涉及工具调用，纯粹通过 prompt engineering 实现。

三种实现方式:
    1. Zero-shot CoT: 仅在 prompt 中加入 "请一步步思考" 即可触发
    2. Few-shot CoT: 在 prompt 中给出包含推理过程的示例
    3. Agent + dynamic_prompt: 用 create_agent 配合动态 prompt 中间件，根据问题类型切换 CoT 策略

LangChain 1.0+ 要点:
    - 纯推理场景可直接用 LCEL (model + prompt)
    - 如果想在 Agent 中嵌入 CoT，可以通过 @dynamic_prompt 中间件动态注入思维链指令

运行方式:
    python -m src.langchain.agent_patterns.chain_of_thought
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from provider import get_default_model


# ==================== 1. 无 CoT 的直接回答（LCEL）====================

def direct_answer(question: str) -> str:
    """不使用 CoT，让模型直接给出答案"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手。请直接给出最终答案，不需要解释过程。"),
        ("human", "{question}")
    ])
    chain = prompt | get_default_model()
    return chain.invoke({"question": question}).content


# ==================== 2. Zero-shot CoT（LCEL）====================

def zero_shot_cot(question: str) -> str:
    """
    Zero-shot CoT：仅通过一句 "请一步步思考" 就能显著提升推理质量。
    来源: Kojima et al. (2022) "Large Language Models are Zero-Shot Reasoners"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个逻辑推理专家。\n"
         "请按以下格式回答：\n"
         "【思考过程】逐步分析问题...\n"
         "【最终答案】给出结论"),
        ("human", "{question}\n\n请一步步思考。")
    ])
    chain = prompt | get_default_model()
    return chain.invoke({"question": question}).content


# ==================== 3. Few-shot CoT（LCEL）====================

FEW_SHOT_EXAMPLES = """
示例 1:
问题: 一个商店有 23 个苹果，卖掉了 17 个，又进货了 6 个，现在有多少个？
【思考过程】
第一步：初始数量是 23 个苹果。
第二步：卖掉 17 个后剩余 23 - 17 = 6 个。
第三步：进货 6 个后总共 6 + 6 = 12 个。
【最终答案】12 个苹果。

示例 2:
问题: 小明比小红大 3 岁，小红比小刚小 2 岁，小刚今年 10 岁，小明今年多大？
【思考过程】
第一步：小刚今年 10 岁。
第二步：小红比小刚小 2 岁，所以小红 = 10 - 2 = 8 岁。
第三步：小明比小红大 3 岁，所以小明 = 8 + 3 = 11 岁。
【最终答案】小明今年 11 岁。
""".strip()


def few_shot_cot(question: str) -> str:
    """
    Few-shot CoT：通过提供包含推理过程的示例，引导模型生成更结构化的推理链。
    来源: Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个逻辑推理专家。请参考下面的示例格式来解题。\n\n"
         f"{FEW_SHOT_EXAMPLES}"),
        ("human", "问题: {question}\n请按照示例格式，先展示思考过程，再给出最终答案。")
    ])
    chain = prompt | get_default_model()
    return chain.invoke({"question": question}).content


# ==================== 4. Agent + dynamic_prompt（LangChain 1.0+ 方式）====================

MATH_COT_PROMPT = (
    "你是一个数学推理专家。\n"
    "对于每个问题，你必须：\n"
    "1. 识别已知条件\n"
    "2. 列出解题步骤\n"
    "3. 逐步计算\n"
    "4. 验证答案\n"
    "请使用【思考过程】和【最终答案】格式。"
)

LOGIC_COT_PROMPT = (
    "你是一个逻辑推理专家。\n"
    "对于每个问题，你必须：\n"
    "1. 分析题目约束条件\n"
    "2. 列举可能的情况\n"
    "3. 用排除法或推导法逐步缩小范围\n"
    "4. 得出结论\n"
    "请使用【思考过程】和【最终答案】格式。"
)

DEFAULT_COT_PROMPT = (
    "你是一个善于分析的助手。请一步步思考问题，展示完整推理过程，最后给出答案。"
)


@dynamic_prompt
def cot_strategy_prompt(request: ModelRequest) -> str:
    """
    动态 CoT 中间件：根据问题类型自动选择合适的思维链策略。

    这是 LangChain 1.0+ 推荐的方式 —— 通过 @dynamic_prompt 中间件
    在运行时动态修改 system prompt，而不是硬编码在 agent 创建时。
    """
    if not request.state.get("messages"):
        return DEFAULT_COT_PROMPT

    last_msg = request.state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    math_keywords = ["计算", "多少", "等于", "加", "减", "乘", "除", "速度", "距离", "时间", "百分比"]
    logic_keywords = ["如果", "假设", "推理", "判断", "哪个", "为什么", "证明", "开关", "灯"]

    if any(kw in content for kw in math_keywords):
        print("  [CoT策略] 检测到数学题 → 使用数学 CoT")
        return MATH_COT_PROMPT
    elif any(kw in content for kw in logic_keywords):
        print("  [CoT策略] 检测到逻辑题 → 使用逻辑 CoT")
        return LOGIC_COT_PROMPT
    else:
        print("  [CoT策略] 通用问题 → 使用默认 CoT")
        return DEFAULT_COT_PROMPT


def agent_cot(question: str) -> str:
    """
    使用 create_agent + @dynamic_prompt 实现动态 CoT。

    优势：Agent 可以根据问题类型自动切换最合适的 CoT prompt，
    而且可以扩展工具（如计算器）辅助推理。
    """
    agent = create_agent(
        get_default_model(),
        tools=[],
        middleware=[cot_strategy_prompt],
    )
    result = agent.invoke({"messages": [HumanMessage(question)]})
    return result["messages"][-1].content


# ==================== 演示入口 ====================

def run_comparison():
    """对比四种方式在同一问题上的表现"""
    test_questions = [
        "一辆火车从 A 站出发，时速 60 公里，2 小时后另一辆火车从 B 站出发追赶，时速 90 公里。"
        "A、B 两站距离 30 公里。第二辆火车需要多长时间追上第一辆？",

        "有 3 个开关控制 3 盏灯，你在门外看不到灯。"
        "你可以随意拨动开关，但只能进门看一次。怎样确定每个开关控制哪盏灯？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"测试问题 {i}: {question}")
        print('=' * 60)

        print("\n--- 1. 直接回答（无 CoT）---")
        print(direct_answer(question))

        print("\n--- 2. Zero-shot CoT ---")
        print(zero_shot_cot(question))

        print("\n--- 3. Few-shot CoT ---")
        print(few_shot_cot(question))

        print("\n--- 4. Agent + dynamic_prompt CoT（LangChain 1.0+）---")
        print(agent_cot(question))


if __name__ == "__main__":
    run_comparison()
