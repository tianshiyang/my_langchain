#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 进阶版：
classify -> plan -> dispatch -> specialists/refund_subgraph -> synthesize -> review -> notify
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, Send, interrupt

from .advanced_shared import (
    DEFAULT_NOTIFICATION_RECIPIENT,
    build_support_draft,
    create_notification_agent,
    review_support_draft,
    run_logistics_specialist_fallback,
    run_notification_specialist,
    run_product_specialist_fallback,
)
from .models import (
    MAX_REVIEW_ROUNDS,
    MAX_SUPERVISOR_ROUNDS,
    ReviewResult,
    SubTask,
    TaskResult,
    all_results_terminal,
    should_stop_agent_loop,
)
from .support_shared import (
    blocked_task_result,
    classify_customer_query,
    escalate_refund_service,
    get_default_model,
    refund_eligibility_service,
    submit_refund_service,
)


class SupportGraphState(TypedDict):
    """主图状态。"""

    thread_id: str
    user_query: str
    route_plan: dict[str, Any]
    tasks: list[SubTask]
    active_task_ids: list[str]
    pending_task_ids: list[str]
    completed_task_ids: Annotated[list[str], operator.add]
    task_results: Annotated[list[TaskResult], operator.add]
    clarification_question: str
    requires_human_review: bool
    draft_response: str
    review_decision: str
    review_feedback: str
    supervisor_round: int
    review_round: int
    final_response: str
    notification_result: dict[str, Any]
    force_reviewer_revision: bool
    recipient: str
    execution_trace: Annotated[list[dict[str, Any]], operator.add]


class SpecialistTaskState(TypedDict):
    """并行 specialist 的输入。"""

    task: SubTask
    completed_task_ids: Annotated[list[str], operator.add]
    task_results: Annotated[list[TaskResult], operator.add]
    execution_trace: Annotated[list[dict[str, Any]], operator.add]


class RefundWorkflowState(TypedDict):
    """退款子图状态。"""

    task: SubTask
    order_id: str
    reason: str
    eligibility: bool
    audit_required: bool
    manual_approval_required: bool
    approval_decision: str
    completed_task_ids: Annotated[list[str], operator.add]
    task_results: Annotated[list[TaskResult], operator.add]
    execution_trace: Annotated[list[dict[str, Any]], operator.add]


def build_refund_subgraph():
    """构建显式退款子图。"""

    def collect_inputs(state: RefundWorkflowState):
        task = state["task"]
        if task.missing_slots:
            return {
                "task_results": [blocked_task_result(task)],
                "completed_task_ids": [task.task_id],
                "execution_trace": [
                    {
                        "stage": "refund_collect_inputs",
                        "task_id": task.task_id,
                        "status": "blocked_waiting_user",
                    }
                ],
            }
        return {
            "order_id": str(task.slots["order_id"]),
            "reason": str(task.slots["refund_reason"]),
            "execution_trace": [
                {
                    "stage": "refund_collect_inputs",
                    "task_id": task.task_id,
                    "status": "ready",
                }
            ],
        }

    def route_after_collect(state: RefundWorkflowState) -> Literal["evaluate_eligibility", END]:
        if state.get("completed_task_ids"):
            return END
        return "evaluate_eligibility"

    def evaluate_eligibility(state: RefundWorkflowState):
        policy = refund_eligibility_service(state["order_id"])
        return {
            "eligibility": bool(policy["eligible"]),
            "audit_required": bool(policy["audit_required"]),
            "manual_approval_required": bool(policy.get("manual_approval_required")),
            "execution_trace": [
                {
                    "stage": "refund_policy_checked",
                    "task_id": state["task"].task_id,
                    "policy": policy,
                }
            ],
        }

    def choose_path(
        state: RefundWorkflowState,
    ) -> Literal["submit_refund", "escalate_refund", "approval_gate", "decline_refund"]:
        if not state["eligibility"]:
            return "decline_refund"
        if state["audit_required"]:
            return "escalate_refund"
        if state["manual_approval_required"]:
            return "approval_gate"
        return "submit_refund"

    def approval_gate(state: RefundWorkflowState):
        decision = interrupt(
            {
                "kind": "refund_approval",
                "task_id": state["task"].task_id,
                "order_id": state["order_id"],
                "reason": state["reason"],
                "message": "高金额退款需要人工批准后才能自动放行。",
            }
        )
        decision_text = ""
        if isinstance(decision, dict):
            decision_text = str(decision.get("decision", "approve"))
        elif isinstance(decision, str):
            decision_text = decision
        return {
            "approval_decision": decision_text or "approve",
            "execution_trace": [
                {
                    "stage": "refund_human_gate",
                    "task_id": state["task"].task_id,
                    "decision": decision_text or "approve",
                }
            ],
        }

    def route_after_approval(state: RefundWorkflowState) -> Literal["submit_refund", "manual_reject"]:
        if state.get("approval_decision", "approve") == "approve":
            return "submit_refund"
        return "manual_reject"

    def submit_refund(state: RefundWorkflowState):
        payload = submit_refund_service(state["order_id"], state["reason"])
        return {
            "task_results": [
                TaskResult(
                    task_id=state["task"].task_id,
                    intent=state["task"].intent,
                    status="done",
                    answer=(
                        f"订单 {state['order_id']} 已提交退款申请，"
                        f"退款单号为 {payload['refund_id']}。"
                    ),
                    raw_data=payload,
                )
            ],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "refund_submit",
                    "task_id": state["task"].task_id,
                    "status": "done",
                }
            ],
        }

    def escalate_refund(state: RefundWorkflowState):
        payload = escalate_refund_service(state["order_id"], state["reason"])
        return {
            "task_results": [
                TaskResult(
                    task_id=state["task"].task_id,
                    intent=state["task"].intent,
                    status="escalated_to_human",
                    answer=(
                        f"订单 {state['order_id']} 需要人工审核，"
                        f"已创建工单 {payload['ticket_id']}。"
                    ),
                    human_required=True,
                    raw_data=payload,
                )
            ],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "refund_escalate",
                    "task_id": state["task"].task_id,
                    "status": "escalated_to_human",
                }
            ],
        }

    def decline_refund(state: RefundWorkflowState):
        policy = refund_eligibility_service(state["order_id"])
        return {
            "task_results": [
                TaskResult(
                    task_id=state["task"].task_id,
                    intent=state["task"].intent,
                    status="done",
                    answer=policy["reason"],
                    raw_data=policy,
                )
            ],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "refund_decline",
                    "task_id": state["task"].task_id,
                    "status": "done",
                }
            ],
        }

    def manual_reject(state: RefundWorkflowState):
        return {
            "task_results": [
                TaskResult(
                    task_id=state["task"].task_id,
                    intent=state["task"].intent,
                    status="escalated_to_human",
                    answer=(
                        f"订单 {state['order_id']} 的高金额退款未获得值班主管自动放行批准，"
                        "已转人工售后继续处理。"
                    ),
                    human_required=True,
                    raw_data={"approval_decision": state.get("approval_decision", "reject")},
                )
            ],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "refund_manual_reject",
                    "task_id": state["task"].task_id,
                    "status": "escalated_to_human",
                }
            ],
        }

    workflow = StateGraph(RefundWorkflowState)
    workflow.add_node("collect_inputs", collect_inputs)
    workflow.add_node("evaluate_eligibility", evaluate_eligibility)
    workflow.add_node("approval_gate", approval_gate)
    workflow.add_node("submit_refund", submit_refund)
    workflow.add_node("escalate_refund", escalate_refund)
    workflow.add_node("decline_refund", decline_refund)
    workflow.add_node("manual_reject", manual_reject)
    workflow.add_edge(START, "collect_inputs")
    workflow.add_conditional_edges("collect_inputs", route_after_collect)
    workflow.add_conditional_edges("evaluate_eligibility", choose_path)
    workflow.add_conditional_edges("approval_gate", route_after_approval)
    workflow.add_edge("submit_refund", END)
    workflow.add_edge("escalate_refund", END)
    workflow.add_edge("decline_refund", END)
    workflow.add_edge("manual_reject", END)
    return workflow.compile(checkpointer=InMemorySaver())


def build_langgraph_agent_loop_multi_agent(model=None):
    """构建 LangGraph 进阶版主图。"""
    llm = model or get_default_model()
    notification_agent = create_notification_agent(llm)
    refund_subgraph = build_refund_subgraph()

    def classify_node(state: SupportGraphState):
        route_plan = classify_customer_query(llm, state["user_query"])
        return {
            "route_plan": route_plan.model_dump(mode="json"),
            "tasks": route_plan.sub_tasks,
            "clarification_question": route_plan.clarification_question,
            "execution_trace": [
                {
                    "stage": "route",
                    "intents": route_plan.intents,
                    "need_clarification": route_plan.need_clarification,
                }
            ],
        }

    def plan_tasks_node(state: SupportGraphState):
        blocked_results = [blocked_task_result(task) for task in state["tasks"] if task.missing_slots]
        runnable_tasks = [task for task in state["tasks"] if not task.missing_slots]
        return {
            "active_task_ids": [task.task_id for task in runnable_tasks],
            "pending_task_ids": [task.task_id for task in runnable_tasks],
            "completed_task_ids": [result.task_id for result in blocked_results],
            "task_results": blocked_results,
            "supervisor_round": state.get("supervisor_round", 0) + 1,
            "execution_trace": [
                {
                    "stage": "plan_tasks",
                    "active_task_ids": [task.task_id for task in runnable_tasks],
                    "blocked_task_ids": [result.task_id for result in blocked_results],
                }
            ],
        }

    def route_after_planning(state: SupportGraphState) -> Literal["dispatch_tasks", "synthesize"]:
        if state["active_task_ids"]:
            return "dispatch_tasks"
        return "synthesize"

    def dispatch_tasks(_: SupportGraphState):
        return {}

    def fan_out_tasks(state: SupportGraphState):
        sends = []
        runnable_tasks = [task for task in state["tasks"] if not task.missing_slots]
        for task in runnable_tasks:
            if task.intent == "物流查询":
                sends.append(Send("logistics_task", {"task": task}))
            elif task.intent == "商品咨询":
                sends.append(Send("product_task", {"task": task}))
            elif task.intent == "退款":
                sends.append(Send("refund_task", {"task": task}))
        return sends

    def logistics_task_node(state: SpecialistTaskState):
        result = run_logistics_specialist_fallback(state["task"])
        return {
            "task_results": [result],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "logistics_task",
                    "task_id": state["task"].task_id,
                    "status": result.status,
                }
            ],
        }

    def product_task_node(state: SpecialistTaskState):
        result = run_product_specialist_fallback(state["task"])
        return {
            "task_results": [result],
            "completed_task_ids": [state["task"].task_id],
            "execution_trace": [
                {
                    "stage": "product_task",
                    "task_id": state["task"].task_id,
                    "status": result.status,
                }
            ],
        }

    def synthesize_node(state: SupportGraphState):
        pending = [task_id for task_id in state["active_task_ids"] if task_id not in state["completed_task_ids"]]
        ordered_results = []
        task_result_map = {item.task_id: item for item in state["task_results"]}
        for task in state["tasks"]:
            if task.task_id in task_result_map:
                ordered_results.append(task_result_map[task.task_id])

        draft = build_support_draft(
            llm,
            state["user_query"],
            ordered_results,
            clarification_question=state.get("clarification_question", ""),
            review_feedback=state.get("review_feedback", ""),
            force_incomplete=state.get("force_reviewer_revision", False) and state.get("review_round", 0) == 0,
        )
        return {
            "pending_task_ids": pending,
            "requires_human_review": any(item.human_required for item in ordered_results),
            "draft_response": draft,
            "supervisor_round": state.get("supervisor_round", 0) + 1,
            "execution_trace": [
                {
                    "stage": "synthesize",
                    "pending_task_ids": pending,
                    "task_count": len(ordered_results),
                }
            ],
        }

    def review_node(state: SupportGraphState):
        ordered_results = []
        task_result_map = {item.task_id: item for item in state["task_results"]}
        for task in state["tasks"]:
            if task.task_id in task_result_map:
                ordered_results.append(task_result_map[task.task_id])

        review_result = review_support_draft(
            llm,
            state["user_query"],
            ordered_results,
            state["draft_response"],
        )
        return {
            "review_decision": review_result.decision,
            "review_feedback": review_result.feedback,
            "review_round": state.get("review_round", 0) + 1,
            "execution_trace": [
                {
                    "stage": "review",
                    "decision": review_result.decision,
                    "feedback": review_result.feedback,
                }
            ],
        }

    def route_after_review(state: SupportGraphState) -> Literal["notify", "synthesize"]:
        ordered_results = []
        task_result_map = {item.task_id: item for item in state["task_results"]}
        for task in state["tasks"]:
            if task.task_id in task_result_map:
                ordered_results.append(task_result_map[task.task_id])

        review_result = ReviewResult(
            decision=state["review_decision"],  # type: ignore[arg-type]
            feedback=state.get("review_feedback", ""),
        )
        if should_stop_agent_loop(
            ordered_results,
            review_result=review_result,
            supervisor_round=state.get("supervisor_round", 0),
            review_round=state.get("review_round", 0),
        ) and not state.get("pending_task_ids"):
            return "notify"
        if state.get("review_round", 0) >= MAX_REVIEW_ROUNDS:
            return "notify"
        return "synthesize"

    def notify_node(state: SupportGraphState):
        content = state["draft_response"]
        if state.get("review_decision") != "approved" and state.get("review_feedback"):
            content = f"{content}\n\n审核补充：{state['review_feedback']}"

        local_trace: list[dict[str, Any]] = []
        notification_result = run_notification_specialist(
            notification_agent,
            content=content,
            recipient=state.get("recipient", DEFAULT_NOTIFICATION_RECIPIENT),
            thread_id=f"{state['thread_id']}:notification",
            execution_trace=local_trace,
        )
        return {
            "final_response": content,
            "notification_result": notification_result,
            "execution_trace": local_trace
            + [
                {
                    "stage": "notify",
                    "recipient": state.get("recipient", DEFAULT_NOTIFICATION_RECIPIENT),
                    "status": notification_result["status"],
                }
            ],
        }

    workflow = StateGraph(SupportGraphState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("plan_tasks", plan_tasks_node)
    workflow.add_node("dispatch_tasks", dispatch_tasks)
    workflow.add_node("logistics_task", logistics_task_node)
    workflow.add_node("product_task", product_task_node)
    workflow.add_node("refund_task", refund_subgraph)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("review", review_node)
    workflow.add_node("notify", notify_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "plan_tasks")
    workflow.add_conditional_edges("plan_tasks", route_after_planning)
    workflow.add_conditional_edges("dispatch_tasks", fan_out_tasks)
    workflow.add_edge("logistics_task", "synthesize")
    workflow.add_edge("product_task", "synthesize")
    workflow.add_edge("refund_task", "synthesize")
    workflow.add_edge("synthesize", "review")
    workflow.add_conditional_edges("review", route_after_review)
    workflow.add_edge("notify", END)

    return workflow.compile(checkpointer=InMemorySaver())


def default_graph_human_decision(interrupt_value: dict[str, Any]) -> dict[str, str]:
    """LangGraph refund 审批的默认人工决策。"""
    if interrupt_value.get("kind") == "refund_approval":
        return {"decision": "approve"}
    return {"decision": "approve"}


def run_langgraph_demo(
    graph,
    payload: dict[str, Any],
    thread_id: str,
    decision_resolver=None,
):
    """自动处理 graph interrupt，便于 demo 一次跑完。"""
    resolver = decision_resolver or default_graph_human_decision
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(payload, config=config)
    while "__interrupt__" in result:
        interrupt_ = result["__interrupt__"][0]
        decision = resolver(interrupt_.value)
        result = graph.invoke(Command(resume=decision), config=config)
    return result


if __name__ == "__main__":
    graph = build_langgraph_agent_loop_multi_agent()
    scenarios = [
        (
            "帮我查订单 A1001 的物流；如果订单 A1004 可以退款就帮我申请退款，原因是买错了；"
            "最后把处理结果发给我确认"
        ),
        "订单 A1003 我想退款，请帮我处理",
        "帮我查订单 A1001 的物流，再把 A1003 退掉，最后把处理结果发给我确认",
    ]

    for index, query in enumerate(scenarios, start=1):
        print("\n" + "=" * 100)
        print(f"[LangGraph Advanced Demo {index}] {query}")
        output = run_langgraph_demo(
            graph,
            {
                "thread_id": f"lg-advanced-{index}",
                "user_query": query,
                "route_plan": {},
                "tasks": [],
                "active_task_ids": [],
                "pending_task_ids": [],
                "completed_task_ids": [],
                "task_results": [],
                "clarification_question": "",
                "requires_human_review": False,
                "draft_response": "",
                "review_decision": "",
                "review_feedback": "",
                "supervisor_round": 0,
                "review_round": 0,
                "final_response": "",
                "notification_result": {},
                "force_reviewer_revision": index == 1,
                "recipient": DEFAULT_NOTIFICATION_RECIPIENT,
                "execution_trace": [],
            },
            thread_id=f"lg-advanced-{index}",
        )
        print(output["route_plan"])
        print("-" * 100)
        print(output["final_response"])
        print("-" * 100)
        print(output["notification_result"])
