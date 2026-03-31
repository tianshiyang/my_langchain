#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 1.x 版本：
显式状态图 + 并行 specialist + refund subflow
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from .models import CustomerIntentResult, SubTask, TaskResult
from .support_shared import (
    blocked_task_result,
    classify_customer_query,
    create_logistics_agent,
    create_product_agent,
    get_default_model,
    refund_eligibility_service,
    run_logistics_specialist,
    run_product_specialist,
    submit_refund_service,
    escalate_refund_service,
    synthesize_support_response,
)


class SupportGraphState(TypedDict):
    """主图状态。"""

    user_query: str
    detected_intents: list[str]
    tasks: list[SubTask]
    active_task_ids: list[str]
    task_results: Annotated[list[TaskResult], operator.add]
    missing_slots: list[str]
    need_clarification: bool
    clarification_question: str
    requires_human_review: bool
    final_response: str


class SpecialistTaskState(TypedDict):
    """并行 specialist 的输入。"""

    task: SubTask


class RefundWorkflowState(TypedDict):
    """退款子流程状态。"""

    task: SubTask
    order_id: str
    reason: str
    eligibility: bool
    audit_required: bool
    task_result: TaskResult


def build_refund_subgraph():
    """构建退款子流程。"""

    def collect_inputs(state: RefundWorkflowState):
        task = state["task"]
        if task.missing_slots:
            return {"task_result": blocked_task_result(task)}
        return {
            "order_id": str(task.slots["order_id"]),
            "reason": str(task.slots["refund_reason"]),
        }

    def route_after_collect(state: RefundWorkflowState) -> Literal["evaluate_eligibility", END]:
        if state.get("task_result"):
            return END
        return "evaluate_eligibility"

    def evaluate_eligibility(state: RefundWorkflowState):
        policy = refund_eligibility_service(state["order_id"])
        return {
            "eligibility": bool(policy["eligible"]),
            "audit_required": bool(policy["audit_required"]),
            "task_result": TaskResult(
                task_id=state["task"].task_id,
                intent=state["task"].intent,
                status="done",
                answer=policy["reason"],
                human_required=False,
                raw_data=policy,
            ),
        }

    def choose_path(
        state: RefundWorkflowState,
    ) -> Literal["submit_refund", "escalate_refund", "decline_refund"]:
        if not state["eligibility"]:
            return "decline_refund"
        if state["audit_required"]:
            return "escalate_refund"
        return "submit_refund"

    def submit_refund(state: RefundWorkflowState):
        payload = submit_refund_service(state["order_id"], state["reason"])
        return {
            "task_result": TaskResult(
                task_id=state["task"].task_id,
                intent=state["task"].intent,
                status="done",
                answer=(
                    f"订单 {state['order_id']} 已提交退款申请，"
                    f"退款单号为 {payload['refund_id']}。"
                ),
                human_required=False,
                raw_data=payload,
            )
        }

    def escalate_refund(state: RefundWorkflowState):
        payload = escalate_refund_service(state["order_id"], state["reason"])
        return {
            "task_result": TaskResult(
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
        }

    def decline_refund(state: RefundWorkflowState):
        policy = refund_eligibility_service(state["order_id"])
        return {
            "task_result": TaskResult(
                task_id=state["task"].task_id,
                intent=state["task"].intent,
                status="done",
                answer=policy["reason"],
                human_required=False,
                raw_data=policy,
            )
        }

    workflow = StateGraph(RefundWorkflowState)
    workflow.add_node("collect_inputs", collect_inputs)
    workflow.add_node("evaluate_eligibility", evaluate_eligibility)
    workflow.add_node("submit_refund", submit_refund)
    workflow.add_node("escalate_refund", escalate_refund)
    workflow.add_node("decline_refund", decline_refund)
    workflow.add_edge(START, "collect_inputs")
    workflow.add_conditional_edges("collect_inputs", route_after_collect)
    workflow.add_conditional_edges("evaluate_eligibility", choose_path)
    workflow.add_edge("submit_refund", END)
    workflow.add_edge("escalate_refund", END)
    workflow.add_edge("decline_refund", END)
    return workflow.compile()


def build_customer_support_graph(model=None):
    """构建客服主图。"""
    llm = model or get_default_model()
    logistics_agent = create_logistics_agent(llm)
    product_agent = create_product_agent(llm)
    refund_subgraph = build_refund_subgraph()

    def classify_node(state: SupportGraphState):
        route_plan: CustomerIntentResult = classify_customer_query(llm, state["user_query"])
        return {
            "detected_intents": route_plan.intents,
            "tasks": route_plan.sub_tasks,
            "missing_slots": route_plan.missing_slots,
            "need_clarification": route_plan.need_clarification,
            "clarification_question": route_plan.clarification_question,
        }

    def plan_tasks_node(state: SupportGraphState):
        runnable_tasks = [task for task in state["tasks"] if not task.missing_slots]
        blocked_results = [blocked_task_result(task) for task in state["tasks"] if task.missing_slots]
        return {
            "active_task_ids": [task.task_id for task in runnable_tasks],
            "task_results": blocked_results,
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
        result = run_logistics_specialist(logistics_agent, state["task"])
        return {"task_results": [result]}

    def product_task_node(state: SpecialistTaskState):
        result = run_product_specialist(product_agent, state["task"])
        return {"task_results": [result]}

    def refund_task_node(state: SpecialistTaskState):
        subgraph_result = refund_subgraph.invoke({"task": state["task"]})
        return {"task_results": [subgraph_result["task_result"]]}

    def synthesize_node(state: SupportGraphState):
        requires_human_review = any(item.human_required for item in state["task_results"])
        final_response = synthesize_support_response(
            llm,
            state["user_query"],
            state["task_results"],
            clarification_question=state.get("clarification_question", ""),
        )
        return {
            "requires_human_review": requires_human_review,
            "final_response": final_response,
        }

    workflow = StateGraph(SupportGraphState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("plan_tasks", plan_tasks_node)
    workflow.add_node("dispatch_tasks", dispatch_tasks)
    workflow.add_node("logistics_task", logistics_task_node)
    workflow.add_node("product_task", product_task_node)
    workflow.add_node("refund_task", refund_task_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "plan_tasks")
    workflow.add_conditional_edges("plan_tasks", route_after_planning)
    workflow.add_conditional_edges("dispatch_tasks", fan_out_tasks)
    workflow.add_edge("logistics_task", "synthesize")
    workflow.add_edge("product_task", "synthesize")
    workflow.add_edge("refund_task", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_customer_support_graph()
    scenarios = [
        "帮我查一下订单 A1001 的物流到哪了",
        "无线耳机 Pro 有货吗，续航怎么样？",
        "订单 A1003 我想退款，原因是买错了",
        "帮我查订单 A1001 的物流，另外我想把 A1003 退掉，原因是买错了",
    ]

    for index, query in enumerate(scenarios, start=1):
        print("\n" + "=" * 100)
        print(f"[LangGraph Demo {index}] {query}")
        output = graph.invoke(
            {
                "user_query": query,
                "detected_intents": [],
                "tasks": [],
                "active_task_ids": [],
                "task_results": [],
                "missing_slots": [],
                "need_clarification": False,
                "clarification_question": "",
                "requires_human_review": False,
                "final_response": "",
            }
        )
        print(output["final_response"])
        print("-" * 100)
        for task_result in output["task_results"]:
            print(task_result.model_dump_json(ensure_ascii=False))
