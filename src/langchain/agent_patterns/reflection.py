#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : reflection.py

Reflection -- 反思迭代模式 (LangChain 1.0+)

核心思想:
    通过 Generator（生成者）和 Reflector（反思者）两个角色交替工作，持续改进输出：
    1. Generator 根据任务要求生成初始内容
    2. Reflector 以"严格审稿人"的视角批评内容，给出改进建议
    3. Generator 根据反馈修改内容
    4. 重复 2-3 步，直到质量达标或达到最大迭代次数

来源:
    Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"

LangChain 1.0+ 实现:
    方式 1: 用两个 create_agent 分别扮演 Generator 和 Reflector（本文件演示）
    方式 2: 用单个 create_agent + @wrap_model_call 中间件在模型调用前注入反思逻辑

    方式 1 更直观，两个 Agent 职责清晰；
    方式 2 更紧凑，适合简单的反思场景。

    本文件两种方式都会演示。

与其他模式的区别:
    - vs CoT: CoT 一次生成；Reflection 多轮迭代改进
    - vs ToT: ToT 并行探索多条路径；Reflection 单路径反复打磨
    - vs ReAct: ReAct 用工具获取外部信息；Reflection 用"内省"来提升质量

运行方式:
    python -m src.langchain.agent_patterns.reflection
"""
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from provider import get_default_model


# ==================== 1. 结构化输出定义 ====================

class ReflectionFeedback(BaseModel):
    """Reflector 的反馈"""
    is_satisfactory: bool = Field(description="当前内容是否达到要求")
    strengths: list[str] = Field(description="内容的优点")
    weaknesses: list[str] = Field(description="需要改进的地方")
    suggestions: list[str] = Field(description="具体的改进建议")
    overall_score: int = Field(description="整体评分 1-10，8 分以上视为达标")


# ==================== 方式 1: 双 Agent 协作 ====================

@tool
def search_technical_info(query: str) -> str:
    """搜索技术资料，用于增强文章内容"""
    info = {
        "stategraph": "StateGraph 是 LangGraph 的核心类，通过 add_node/add_edge 构建有状态工作流。",
        "rag": "RAG 通过在生成前检索相关文档，解决了 LLM 知识过时和幻觉问题。",
        "agent": "LangChain 1.0+ 的 create_agent 内部是 ReAct 循环，通过 middleware 实现扩展。",
        "middleware": "LangChain middleware 提供 wrap_model_call、wrap_tool_call、dynamic_prompt 等钩子。",
    }
    for key, value in info.items():
        if key in query.lower():
            return value
    return f"关于 '{query}' 的技术信息：这是一个重要的技术概念。"


def build_generator_agent():
    """Generator Agent: 负责生成和修改文章"""
    return create_agent(
        get_default_model(),
        tools=[search_technical_info],
        system_prompt=(
            "你是一个专业的技术写作者。\n"
            "你可以使用搜索工具获取技术资料来增强文章。\n"
            "写作要求：\n"
            "- 包含引言、正文（2-3 个要点）、总结\n"
            "- 用通俗易懂的语言解释技术概念\n"
            "- 适当举例说明\n"
            "- 字数 300-500 字"
        ),
    )


def reflect_on_content(topic: str, draft: str) -> ReflectionFeedback:
    """
    Reflector: 用 with_structured_output 评审文章。

    评审不需要工具调用，直接输出结构化反馈即可。
    """
    model = get_default_model()
    reflector = model.with_structured_output(ReflectionFeedback)

    return reflector.invoke([
        {"role": "system", "content": (
            "你是一位经验丰富的技术文章审稿人。\n"
            "请从以下维度严格评估：\n"
            "1. 结构完整性 - 是否有清晰的引言/正文/结论\n"
            "2. 内容准确性 - 技术概念是否正确\n"
            "3. 可读性 - 是否通俗易懂，有恰当的例子\n"
            "4. 深度 - 是否有足够的分析\n"
            "5. 实用性 - 读者能否获得可操作的知识\n\n"
            "8 分以上视为达标。请客观评价。"
        )},
        {"role": "user", "content": f"文章主题: {topic}\n\n文章内容:\n{draft}"}
    ])


def run_reflection_dual_agent(topic: str, max_iterations: int = 3) -> str:
    """
    方式 1: 双 Agent Reflection

    Generator Agent 生成 → Reflector 评审 → Generator Agent 修改 → ...
    """
    generator = build_generator_agent()

    # 首次生成
    print(f"\n[Generator] 生成初始草稿...")
    result = generator.invoke({
        "messages": [HumanMessage(f"请写一篇关于以下主题的技术文章：{topic}")]
    })
    draft = result["messages"][-1].content
    print(f"  [草稿预览] {draft[:120]}...")

    for iteration in range(1, max_iterations + 1):
        # 反思评审
        print(f"\n[Reflector] 第 {iteration} 轮评审...")
        feedback = reflect_on_content(topic, draft)
        print(f"  评分: {feedback.overall_score}/10")
        print(f"  优点: {', '.join(feedback.strengths[:2])}")
        print(f"  不足: {', '.join(feedback.weaknesses[:2])}")

        if feedback.is_satisfactory or feedback.overall_score >= 8:
            print(f"  >>> 文章达标，结束迭代。")
            break

        # 根据反馈修改
        print(f"\n[Generator] 根据反馈修改...")
        revision_prompt = (
            f"请修改以下文章：\n\n{draft}\n\n"
            f"审稿人反馈：\n"
            f"优点: {'; '.join(feedback.strengths)}\n"
            f"不足: {'; '.join(feedback.weaknesses)}\n"
            f"改进建议: {'; '.join(feedback.suggestions)}\n\n"
            f"请保留优点，针对不足进行改进。可以搜索技术资料增强内容。"
        )
        result = generator.invoke({
            "messages": [HumanMessage(revision_prompt)]
        })
        draft = result["messages"][-1].content
        print(f"  [修改后预览] {draft[:120]}...")
    else:
        print(f"\n  达到最大迭代次数 ({max_iterations})，结束。")

    return draft


# ==================== 方式 2: 单 Agent + middleware ====================

class ReflectionState:
    """跨迭代共享的反思状态"""
    def __init__(self):
        self.iteration = 0
        self.last_feedback: ReflectionFeedback | None = None

reflection_state = ReflectionState()


@wrap_model_call
def auto_reflect_middleware(request: ModelRequest, handler) -> ModelResponse:
    """
    方式 2 的核心中间件：在模型调用前自动注入反思指令。

    工作原理：
    1. 第一次调用：正常生成
    2. 后续调用：如果上一轮有反馈，自动将反馈注入到消息中
    """
    reflection_state.iteration += 1

    if reflection_state.last_feedback and not reflection_state.last_feedback.is_satisfactory:
        fb = reflection_state.last_feedback
        reflection_note = (
            f"\n\n[自动反思] 上一轮评审结果（{fb.overall_score}/10）:\n"
            f"不足: {'; '.join(fb.weaknesses)}\n"
            f"建议: {'; '.join(fb.suggestions)}\n"
            f"请在回答时改进这些方面。"
        )
        messages = list(request.state["messages"])
        if messages and hasattr(messages[-1], "content"):
            from langchain_core.messages import HumanMessage as HM
            last = messages[-1]
            messages[-1] = HM(content=last.content + reflection_note)
            request = request.override(state={**request.state, "messages": messages})
            print(f"  [Middleware] 已注入第 {reflection_state.iteration} 轮反思指令")

    return handler(request)


def run_reflection_middleware(topic: str, max_iterations: int = 3) -> str:
    """
    方式 2: 单 Agent + @wrap_model_call 反思中间件

    更紧凑的实现，反思逻辑封装在中间件中。
    """
    reflection_state.iteration = 0
    reflection_state.last_feedback = None

    agent = create_agent(
        get_default_model(),
        tools=[search_technical_info],
        system_prompt=(
            "你是一个技术写作专家。写文章要求：结构完整、通俗易懂、有例子、300-500 字。"
        ),
        middleware=[auto_reflect_middleware],
    )

    draft = ""
    prompt = f"请写一篇关于以下主题的技术文章：{topic}"

    for iteration in range(max_iterations):
        print(f"\n[迭代 {iteration + 1}] 生成/修改中...")
        result = agent.invoke({"messages": [HumanMessage(prompt)]})
        draft = result["messages"][-1].content
        print(f"  [预览] {draft[:120]}...")

        feedback = reflect_on_content(topic, draft)
        reflection_state.last_feedback = feedback
        print(f"  [评审] {feedback.overall_score}/10 | 达标: {feedback.is_satisfactory}")

        if feedback.is_satisfactory or feedback.overall_score >= 8:
            print(f"  >>> 达标，结束。")
            break

        prompt = f"请改进这篇文章：\n{draft}"

    return draft


# ==================== 运行演示 ====================

def run_demo():
    """运行两种 Reflection 方式的对比演示"""
    topic = "LangGraph 中的 StateGraph 是什么？如何用它构建 Agent 工作流？"

    print(f"{'=' * 60}")
    print(f"主题: {topic}")

    print(f"\n{'=' * 60}")
    print("方式 1: 双 Agent 协作（Generator + Reflector）")
    print('=' * 60)
    result_1 = run_reflection_dual_agent(topic)
    print(f"\n[方式 1 最终文章]\n{result_1}")

    print(f"\n\n{'=' * 60}")
    print("方式 2: 单 Agent + middleware")
    print('=' * 60)
    result_2 = run_reflection_middleware(topic)
    print(f"\n[方式 2 最终文章]\n{result_2}")


if __name__ == "__main__":
    run_demo()
