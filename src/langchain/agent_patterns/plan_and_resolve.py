#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : plan_and_resolve.py

Plan-and-Resolve -- 先规划后执行模式 (LangChain 1.0+)

核心思想:
    将复杂任务拆分为两个阶段：
    1. Planning（规划）: LLM 分析任务，生成完整的执行步骤列表
    2. Execution（执行）: 按步骤逐个执行，每步可调用工具
    3. Replanning（可选重规划）: 执行过程中发现计划需要调整时，重新规划

来源:
    Wang et al. (2023) "Plan-and-Solve Prompting"

LangChain 1.0+ 实现:
    方式 1（本文件演示）: 用两个 create_agent 协作 —— planner_agent 负责规划，
    executor_agent 负责执行每一步。通过外部 Python 循环编排两者。

    方式 2: 用 StateGraph 显式编排 plan/execute/replan 节点（适合更复杂的场景）。

    方式 1 更简洁，充分利用了 create_agent 的 ReAct 能力；
    方式 2 更灵活，可以精确控制每一步的数据流。

与其他模式的区别:
    - vs ReAct: ReAct 每步都由 LLM 决策，是"边走边看"；P&R 先有全局视图
    - vs Self-Ask: Self-Ask 按需分解问题；P&R 一开始就规划完整路径
    - 优势: 更高效（子任务可用小模型），更可控（计划可审查）

运行方式:
    python -m src.langchain.agent_patterns.plan_and_resolve
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from provider import get_default_model


# ==================== 1. 执行工具集 ====================

@tool
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    results = {
        "langchain": "LangChain 是一个用于 LLM 应用开发的框架，1.0+ 版本引入了 create_agent 和 middleware。",
        "langgraph": "LangGraph 是基于 LangChain 的有状态工作流编排库，使用图结构管理 Agent 流程。",
        "区别": "LangChain 提供基础构建块（模型、工具、prompt），LangGraph 在此之上提供显式的状态管理和循环控制。",
        "agent 模式": "常见 Agent 模式：ReAct（推理+行动）、Plan-and-Execute（先规划后执行）、Self-Ask（自问自答）、Reflection（反思迭代）。",
        "react 模式": "ReAct 让 Agent 交替进行推理和工具调用，适合探索性任务。",
        "reflection 模式": "Reflection 让 Agent 生成内容后自我反思改进，适合需要质量把关的生成任务。",
    }
    for key, value in results.items():
        if any(k in query.lower() for k in key.split()):
            return value
    return f"搜索 '{query}' 返回：这是一个通用搜索结果。"


@tool
def summarize_text(text: str) -> str:
    """对一段文本进行摘要"""
    if len(text) > 100:
        return text[:100] + "...（已摘要）"
    return f"摘要：{text}"


@tool
def compare_items(description: str) -> str:
    """对比两个事物，输入格式：'A vs B' 或 'A 和 B 的区别'"""
    return f"对比分析 [{description}]：两者各有优劣，需要根据具体场景选择。"


# ==================== 2. Planner：规划 Agent ====================

class PlanStep(BaseModel):
    """计划中的单个步骤"""
    step_id: int = Field(description="步骤编号")
    description: str = Field(description="这一步要做什么")
    tool_to_use: str = Field(description="要使用的工具: search_web / summarize_text / compare_items / none")
    tool_input: str = Field(description="工具的输入参数")


class TaskPlan(BaseModel):
    """完整的执行计划"""
    goal: str = Field(description="任务目标简述")
    steps: list[PlanStep] = Field(description="2~5 个执行步骤")


class ReplanDecision(BaseModel):
    """重规划决策"""
    is_complete: bool = Field(description="任务是否已完成")
    final_answer: str = Field(default="", description="如果已完成，给出最终答案")
    adjusted_steps: list[PlanStep] = Field(default_factory=list, description="如果需要调整，给出新的步骤")
    reason: str = Field(description="决策理由")


def generate_plan(task: str) -> TaskPlan:
    """
    用 LLM 生成执行计划。

    这里直接用 model.with_structured_output 而不是 create_agent，
    因为规划阶段不需要工具调用，只需要结构化输出。
    """
    model = get_default_model()
    planner = model.with_structured_output(TaskPlan)

    plan = planner.invoke([
        {"role": "system", "content": (
            "你是一个任务规划专家。\n"
            "给定一个任务，将它分解为 2~5 个具体的执行步骤。\n"
            "每个步骤应该明确、可执行，并指定使用的工具。\n\n"
            "可用工具:\n"
            "- search_web: 搜索网络获取信息\n"
            "- summarize_text: 对文本进行摘要\n"
            "- compare_items: 对比两个事物\n"
            "- none: 不需要工具，纯推理步骤"
        )},
        {"role": "user", "content": f"任务: {task}"}
    ])

    return plan


def replan(task: str, completed_results: list[str], remaining_steps: list[PlanStep]) -> ReplanDecision:
    """
    审查进度，决定继续/调整/完成。
    """
    model = get_default_model()
    replanner = model.with_structured_output(ReplanDecision)

    completed_text = "\n".join(f"  - {r}" for r in completed_results)
    remaining_text = "\n".join(f"  - 步骤{s.step_id}: {s.description}" for s in remaining_steps)

    return replanner.invoke([
        {"role": "system", "content": (
            "你是一个任务审查专家。\n"
            "请审视已完成的步骤和结果，判断：\n"
            "1. 信息是否已足够回答原始任务？\n"
            "2. 如果已足够，给出最终答案\n"
            "3. 如果需要更多信息，是否需要调整剩余计划？"
        )},
        {"role": "user", "content": (
            f"原始任务: {task}\n\n"
            f"已完成的结果:\n{completed_text}\n\n"
            f"剩余步骤:\n{remaining_text if remaining_text.strip() else '（无）'}"
        )}
    ])


# ==================== 3. Executor：执行 Agent ====================

def build_executor_agent():
    """
    使用 create_agent 构建执行 Agent。

    执行 Agent 负责按照计划中的单个步骤执行任务，
    它有 ReAct 能力，可以在执行单步时灵活使用工具。
    """
    return create_agent(
        get_default_model(),
        tools=[search_web, summarize_text, compare_items],
        system_prompt=(
            "你是一个任务执行专家。\n"
            "你会收到一个具体的执行步骤，请使用合适的工具完成它。\n"
            "完成后，用 1-2 句话总结这一步获取到的关键信息。"
        ),
    )


# ==================== 4. Plan-and-Resolve 编排 ====================

def run_plan_and_resolve(task: str) -> str:
    """
    Plan-and-Resolve 的完整流程：

    1. Planner 生成计划
    2. 逐步执行
    3. 每步执行后 Replanner 审查
    4. 得到最终答案
    """
    # Phase 1: 规划
    print("\n[Phase 1: Planning]")
    plan = generate_plan(task)
    print(f"  目标: {plan.goal}")
    for step in plan.steps:
        print(f"  步骤 {step.step_id}: {step.description} (工具: {step.tool_to_use})")

    # Phase 2: 逐步执行
    executor = build_executor_agent()
    completed_results = []
    remaining_steps = list(plan.steps)

    while remaining_steps:
        current_step = remaining_steps.pop(0)

        print(f"\n[Phase 2: Executing Step {current_step.step_id}]")
        print(f"  描述: {current_step.description}")

        step_instruction = (
            f"请执行以下步骤：{current_step.description}\n"
            f"建议使用工具: {current_step.tool_to_use}\n"
            f"工具输入: {current_step.tool_input}"
        )

        result = executor.invoke({"messages": [HumanMessage(step_instruction)]})
        step_result = result["messages"][-1].content
        completed_results.append(f"步骤{current_step.step_id}: {step_result}")
        print(f"  结果: {step_result[:120]}...")

        # Phase 3: 重规划
        print(f"\n[Phase 3: Replanning]")
        decision = replan(task, completed_results, remaining_steps)
        print(f"  完成? {decision.is_complete} | 理由: {decision.reason}")

        if decision.is_complete:
            print(f"\n[完成] {decision.final_answer}")
            return decision.final_answer

        if decision.adjusted_steps:
            print(f"  计划已调整，新的步骤:")
            remaining_steps = decision.adjusted_steps
            for s in remaining_steps:
                print(f"    步骤 {s.step_id}: {s.description}")

    # 如果所有步骤都执行完了但 replanner 没有给出最终答案
    final_decision = replan(task, completed_results, [])
    return final_decision.final_answer or "所有步骤已完成，但未能生成最终答案。"


# ==================== 5. 运行演示 ====================

def run_demo():
    """运行 Plan-and-Resolve 演示"""
    test_cases = [
        "帮我调研 LangChain 和 LangGraph 的区别，写一个简要的对比总结",
        "调研主流 Agent 设计模式，列出各自的适用场景",
    ]

    for i, task in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"任务 {i}: {task}")
        print('=' * 60)

        answer = run_plan_and_resolve(task)

        print(f"\n{'=' * 40}")
        print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    run_demo()
