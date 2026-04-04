#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
一个不依赖外部模型的 Agent Loop 最小示例。

运行方式:
    python src/langchain/agent_loop_demo.py

这个示例刻意把 Agent 的核心闭环拆开来写:
1. Planner: 决定下一步做什么
2. Tool: 执行动作，返回观察结果
3. Scratchpad: 记录思考、动作、观察
4. Loop: 在 "计划 -> 执行 -> 观察" 之间反复迭代
5. Finish: 形成最终答案并退出
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


def search_policy(keyword: str) -> str:
    """查询售后规则。"""
    if "退款" in keyword:
        return (
            "退款规则：未发货订单可直接退款；"
            "已签收订单若在 7 天内且商品完好，可申请退款。"
        )
    return f"没有找到与 {keyword} 相关的规则。"


def lookup_order(order_id: str) -> str:
    """查询订单状态。"""
    order_database = {
        "A1001": "订单 A1001 当前状态：已签收 2 天，商品无异常。",
        "B2002": "订单 B2002 当前状态：仓库打包中，尚未发货。",
    }
    return order_database.get(order_id, f"没有找到订单 {order_id}。")


TOOLS: dict[str, Callable[[str], str]] = {
    "search_policy": search_policy,
    "lookup_order": lookup_order,
}


@dataclass
class AgentAction:
    """Planner 决定的一次动作。"""

    kind: str
    name: str
    tool_input: str
    thought: str
    answer: str = ""


@dataclass
class AgentState:
    """Agent 循环过程中的状态。"""

    user_input: str
    scratchpad: list[str] = field(default_factory=list)
    observations: dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    max_iterations: int = 5


class DemoPlanner:
    """一个规则驱动的 Planner，用来模拟大模型的决策过程。"""

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        match = re.search(r"\b[A-Z]\d{4}\b", text)
        return match.group(0) if match else None

    def plan(self, state: AgentState) -> AgentAction:
        user_input = state.user_input
        order_id = self._extract_order_id(user_input)

        if order_id and "lookup_order" not in state.observations:
            return AgentAction(
                kind="tool",
                name="lookup_order",
                tool_input=order_id,
                thought="我需要先确认订单当前所处的状态，才能判断后续处理方式。",
            )

        if "退款" in user_input and "search_policy" not in state.observations:
            return AgentAction(
                kind="tool",
                name="search_policy",
                tool_input="退款规则",
                thought="用户还问到了退款，我需要再查一下售后规则。",
            )

        return AgentAction(
            kind="finish",
            name="finish",
            tool_input="",
            thought="我已经拿到了订单状态和规则信息，可以给出最终结论。",
            answer=self._build_answer(state),
        )

    @staticmethod
    def _build_answer(state: AgentState) -> str:
        order_info = state.observations.get("lookup_order", "未查询到订单信息。")
        policy_info = state.observations.get("search_policy", "未查询到售后规则。")

        if "已签收 2 天" in order_info and "7 天内" in policy_info:
            conclusion = "根据当前信息，这个订单符合退款条件。"
        elif "尚未发货" in order_info:
            conclusion = "订单尚未发货，通常可以直接发起退款。"
        else:
            conclusion = "信息还不够完整，暂时无法准确判断是否可以退款。"

        return f"{order_info}\n{policy_info}\n结论：{conclusion}"


class AgentLoopDemo:
    """负责驱动整个 Agent Loop。"""

    def __init__(self, planner: DemoPlanner, tools: dict[str, Callable[[str], str]]) -> None:
        self.planner = planner
        self.tools = tools

    def run(self, user_input: str) -> AgentState:
        state = AgentState(user_input=user_input)

        for iteration in range(1, state.max_iterations + 1):
            action = self.planner.plan(state)
            state.scratchpad.append(f"[Round {iteration}] Thought: {action.thought}")

            if action.kind == "finish":
                state.final_answer = action.answer
                state.scratchpad.append(f"[Round {iteration}] Final Answer: {action.answer}")
                break

            tool = self.tools[action.name]
            state.scratchpad.append(
                f"[Round {iteration}] Action: {action.name}({action.tool_input})"
            )
            observation = tool(action.tool_input)
            state.observations[action.name] = observation
            state.scratchpad.append(f"[Round {iteration}] Observation: {observation}")
        else:
            state.final_answer = "超过最大迭代次数，Agent 被强制停止。"
            state.scratchpad.append(state.final_answer)

        return state


def run_demo() -> None:
    user_question = "帮我看看订单 A1001 现在是什么状态，如果我要退款，是否符合规则？"
    agent = AgentLoopDemo(planner=DemoPlanner(), tools=TOOLS)
    result = agent.run(user_question)

    print("=" * 20 + " 用户问题 " + "=" * 20)
    print(user_question)
    print()

    print("=" * 20 + " Agent Loop 轨迹 " + "=" * 20)
    for item in result.scratchpad:
        print(item)
    print()

    print("=" * 20 + " 最终回答 " + "=" * 20)
    print(result.final_answer)


if __name__ == "__main__":
    run_demo()
