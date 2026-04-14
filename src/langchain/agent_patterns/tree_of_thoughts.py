#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/7
@Author  : tianshiyang
@File    : tree_of_thoughts.py

Tree of Thoughts (ToT) -- 思维树推理模式 (LangChain 1.0+)

核心思想:
    面对一个问题，不是只走一条推理路径（如 CoT），而是：
    1. Generate（生成）: 并行产出 N 个候选思路/方案
    2. Evaluate（评估）: 对每个候选方案从多个维度打分
    3. Select（选择）: 挑选最优方案
    4. Synthesize（综合）: 吸收其他方案优点，优化最终输出

来源:
    Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"

LangChain 1.0+ 实现:
    ToT 需要并行生成和评估，这不是单个 create_agent 的 ReAct 循环能直接完成的。
    最佳实践是结合：
    - model.with_structured_output() 做结构化的生成和评估
    - create_agent 做最终的综合优化（可调用工具搜索额外信息）
    - 用 Python 循环编排 generate → evaluate → synthesize 流程

与其他模式的区别:
    - vs CoT: CoT 是一条链（深度优先），ToT 是多条链并行（广度优先）
    - vs Reflection: Reflection 是单路径反复打磨，ToT 是多路竞争择优
    - vs Plan-and-Resolve: P&R 按步骤执行，ToT 在每一步探索多种可能

运行方式:
    python -m src.langchain.agent_patterns.tree_of_thoughts
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from provider import get_default_model


# ==================== 1. 结构化输出定义 ====================

class ThoughtCandidate(BaseModel):
    """单个候选思路"""
    title: str = Field(description="方案标题")
    approach: str = Field(description="具体方案描述")
    reasoning: str = Field(description="为什么采用这种方案的推理过程")
    potential_issues: str = Field(description="可能的问题或风险")


class CandidateList(BaseModel):
    """多个候选思路"""
    candidates: list[ThoughtCandidate] = Field(description="3 个不同的候选方案")


class DimensionScore(BaseModel):
    """单个维度的评分"""
    dimension: str = Field(description="评估维度名称")
    score: float = Field(description="评分 (1-10)")
    comment: str = Field(description="评分说明")


class CandidateEvaluation(BaseModel):
    """对单个候选方案的综合评估"""
    candidate_title: str = Field(description="被评估方案的标题")
    scores: list[DimensionScore] = Field(description="各维度评分")
    total_score: float = Field(description="综合得分")
    summary: str = Field(description="评估总结")


class EvaluationResult(BaseModel):
    """所有候选方案的评估结果"""
    evaluations: list[CandidateEvaluation] = Field(description="每个候选方案的评估")
    best_index: int = Field(description="最佳方案的索引（从 0 开始）")
    justification: str = Field(description="选择最佳方案的理由")


# ==================== 2. ToT 三阶段实现 ====================

def generate_candidates(problem: str) -> CandidateList:
    """
    阶段 1: Generate -- 生成多个候选方案

    使用 with_structured_output 确保输出格式统一。
    """
    model = get_default_model()
    generator = model.with_structured_output(CandidateList)

    return generator.invoke([
        {"role": "system", "content": (
            "你是一个创意思维专家。\n"
            "给定一个问题，请从完全不同的角度提出 3 个候选解决方案。\n"
            "每个方案应该有独特的切入点，避免雷同。\n"
            "对每个方案，说明推理过程和可能的风险。"
        )},
        {"role": "user", "content": f"问题: {problem}"}
    ])


def evaluate_candidates(problem: str, candidates: CandidateList) -> EvaluationResult:
    """
    阶段 2: Evaluate -- 对每个候选方案多维度打分

    评估维度：可行性、创新性、完整性、效率
    """
    model = get_default_model()
    evaluator = model.with_structured_output(EvaluationResult)

    candidates_text = ""
    for i, c in enumerate(candidates.candidates):
        candidates_text += (
            f"\n方案 {i + 1}: {c.title}\n"
            f"  思路: {c.approach}\n"
            f"  推理: {c.reasoning}\n"
            f"  风险: {c.potential_issues}\n"
        )

    return evaluator.invoke([
        {"role": "system", "content": (
            "你是一个严格的方案评估专家。\n"
            "请从以下维度对每个方案进行 1-10 分的评估：\n"
            "1. 可行性 - 方案是否切实可行\n"
            "2. 创新性 - 方案是否有新颖之处\n"
            "3. 完整性 - 方案是否考虑周全\n"
            "4. 效率 - 实施成本和时间\n\n"
            "请客观公正，不要给出全高分或全低分。\n"
            "最终选出综合最优的方案。"
        )},
        {"role": "user", "content": f"问题: {problem}\n\n候选方案:{candidates_text}"}
    ])


# ==================== 3. Synthesize Agent（LangChain 1.0+）====================

@tool
def search_best_practices(topic: str) -> str:
    """搜索行业最佳实践来优化方案"""
    practices = {
        "学习路线": "高效学习路线的最佳实践：设定明确目标、循序渐进、项目驱动、定期复盘。",
        "产品策略": "AI 产品策略最佳实践：找准痛点、MVP 验证、快速迭代、用户反馈驱动。",
        "技术选型": "技术选型最佳实践：考虑团队能力、社区生态、长期维护性、性能需求。",
    }
    for key, value in practices.items():
        if any(k in topic for k in key.split()):
            return value
    return f"关于 '{topic}' 的最佳实践：需要结合具体场景综合判断。"


def synthesize_solution(
    problem: str,
    candidates: CandidateList,
    evaluation: EvaluationResult,
) -> str:
    """
    阶段 3: Synthesize -- 使用 create_agent 综合优化最终方案

    这里用 create_agent 而不是简单的 model.invoke，
    因为 Agent 可以调用工具搜索额外信息来增强最终方案。
    """
    best_idx = evaluation.best_index
    best_candidate = candidates.candidates[best_idx]
    best_eval = evaluation.evaluations[best_idx]

    other_highlights = []
    for i, (c, e) in enumerate(zip(candidates.candidates, evaluation.evaluations)):
        if i != best_idx:
            top_dim = max(e.scores, key=lambda s: s.score)
            other_highlights.append(f"「{c.title}」在 {top_dim.dimension} 方面表现突出: {top_dim.comment}")

    synthesis_prompt = (
        f"你需要优化一个方案。\n\n"
        f"原始问题: {problem}\n\n"
        f"选定的最佳方案: {best_candidate.title}\n"
        f"方案内容: {best_candidate.approach}\n"
        f"评估反馈: {best_eval.summary}\n\n"
        f"其他方案的亮点（请吸收）:\n" +
        "\n".join(f"  - {h}" for h in other_highlights) +
        "\n\n请搜索相关最佳实践，然后给出优化后的最终方案。"
        "输出格式：\n"
        "【选定方案】方案名\n"
        "【优化后方案】详细描述\n"
        "【核心优势】列出 3-5 个\n"
        "【实施建议】具体建议"
    )

    agent = create_agent(
        get_default_model(),
        tools=[search_best_practices],
        system_prompt="你是一个方案优化专家。请结合搜索到的最佳实践，优化给定方案。",
    )

    result = agent.invoke({"messages": [HumanMessage(synthesis_prompt)]})
    return result["messages"][-1].content


# ==================== 4. 完整 ToT 流程 ====================

def run_tree_of_thoughts(problem: str) -> str:
    """
    Tree of Thoughts 完整流程：Generate → Evaluate → Synthesize
    """
    # 阶段 1: Generate
    print("\n[阶段 1: Generate] 生成候选方案...")
    candidates = generate_candidates(problem)
    for i, c in enumerate(candidates.candidates):
        print(f"  方案 {i + 1}: {c.title}")
        print(f"    思路: {c.approach[:80]}...")

    # 阶段 2: Evaluate
    print("\n[阶段 2: Evaluate] 评估各方案...")
    evaluation = evaluate_candidates(problem, candidates)
    for ev in evaluation.evaluations:
        print(f"  {ev.candidate_title}: 综合 {ev.total_score:.1f} 分")
        for s in ev.scores:
            print(f"    {s.dimension}: {s.score}/10")
    print(f"  >>> 最佳: {evaluation.evaluations[evaluation.best_index].candidate_title}")
    print(f"  >>> 理由: {evaluation.justification}")

    # 阶段 3: Synthesize
    print("\n[阶段 3: Synthesize] 综合优化最终方案...")
    final = synthesize_solution(problem, candidates, evaluation)

    return final


# ==================== 5. 运行演示 ====================

def run_demo():
    """运行 Tree of Thoughts 演示"""
    test_cases = [
        "设计一个面向初学者的 Python 学习路线，要求 3 个月内能独立完成小项目",
        "一个 10 人创业团队想要开发一款 AI 编程助手产品，请提出产品策略",
    ]

    for i, problem in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"问题 {i}: {problem}")
        print('=' * 60)

        result = run_tree_of_thoughts(problem)

        print(f"\n{'=' * 40}")
        print(f"最终输出:\n{result}")


if __name__ == "__main__":
    run_demo()
