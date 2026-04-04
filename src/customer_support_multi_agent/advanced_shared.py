#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
进阶版客服多智能体共享逻辑：
- reviewer 契约
- notification specialist
- 退款 agent loop + 人机审批
- LangChain / LangGraph 共用的 draft / review / HITL 工具函数
"""
from __future__ import annotations

import json
from typing import Any, Callable, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from .models import ReviewResult, SubTask, TaskResult
from .support_shared import (
    blocked_task_result,
    build_clarification_question,
    check_refund_eligibility,
    extract_last_message_text,
    get_default_model,
    lookup_order_service,
    query_logistics_service,
    refund_eligibility_service,
    search_product_service,
    submit_refund_service,
    escalate_refund_service,
)


DEFAULT_NOTIFICATION_RECIPIENT = "站内信-当前用户"
NOTIFICATION_OUTBOX: list[dict[str, str]] = []


class AdvancedRefundAgentState(AgentState):
    """退款子 agent 的状态。"""

    current_step: NotRequired[str]
    order_id: NotRequired[str]
    reason: NotRequired[str]
    refund_status: NotRequired[str]
    requires_human_review: NotRequired[bool]


class SendConfirmationArgs(BaseModel):
    """通知工具参数。"""

    recipient: str = Field(description="接收通知的目标，例如站内信或邮箱")
    content: str = Field(description="要发送给用户的确认内容")


def default_human_decision(action_name: str, action_request: dict[str, Any]) -> dict[str, Any]:
    """demo 使用的默认人工审批策略。"""
    if action_name == "send_customer_confirmation":
        edited_action = action_request.get("args", {}).copy()
        edited_action["content"] = f"[人工已审阅] {edited_action.get('content', '')}".strip()
        return {"type": "edit", "args": edited_action}
    return {"type": "approve"}


def invoke_with_auto_human_resolution(
    agent,
    payload: dict[str, Any] | Command,
    config: dict[str, Any] | None = None,
    decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
):
    """自动处理 HumanInTheLoopMiddleware 的 interrupt，便于 demo 直接跑通。"""
    resolver = decision_resolver or default_human_decision
    result = agent.invoke(payload, config=config)

    while result.get("__interrupt__"):
        interrupt_ = result["__interrupt__"][0]
        action_requests = interrupt_.value.get("action_requests", [])
        decisions = []
        for request in action_requests:
            decision = resolver(request["name"], request)
            execution_trace and execution_trace.append(
                {
                    "stage": "human_review",
                    "tool": request["name"],
                    "requested_args": request.get("args", {}),
                    "decision": decision["type"],
                }
            )
            if decision["type"] == "edit":
                edited_action = request.copy()
                edited_action["args"] = decision.get("args", request.get("args", {}))
                decisions.append({"type": "edit", "edited_action": edited_action})
            elif decision["type"] == "reject":
                decisions.append({"type": "reject"})
            else:
                decisions.append({"type": "approve"})

        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    return result


@tool
def approve_high_value_refund_release(
    order_id: str,
    reason: str,
    runtime: ToolRuntime[AdvancedRefundAgentState],
) -> Command:
    """高金额退款发放前的人审确认。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"人工已批准高金额退款放行：order_id={order_id}, reason={reason}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "finalize_refund",
            "requires_human_review": False,
        }
    )


@tool
def record_refund_order_id(order_id: str, runtime: ToolRuntime[AdvancedRefundAgentState]) -> Command:
    """记录退款订单号。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"已记录订单号：{order_id}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "order_id": order_id,
            "current_step": "collect_reason",
        }
    )


@tool
def record_refund_reason(reason: str, runtime: ToolRuntime[AdvancedRefundAgentState]) -> Command:
    """记录退款原因。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"已记录退款原因：{reason}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "reason": reason,
            "current_step": "resolve_refund",
        }
    )


@tool
def submit_refund_request(order_id: str, reason: str, runtime: ToolRuntime[AdvancedRefundAgentState]) -> Command:
    """提交自动退款。"""
    payload = submit_refund_service(order_id, reason)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "refund_status": "submitted",
            "requires_human_review": False,
            "current_step": "completed",
        }
    )


@tool
def escalate_refund_case(order_id: str, reason: str, runtime: ToolRuntime[AdvancedRefundAgentState]) -> Command:
    """升级到人工审核。"""
    payload = escalate_refund_service(order_id, reason)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "refund_status": "human_review",
            "requires_human_review": True,
            "current_step": "completed",
        }
    )


@tool
def decline_refund_case(order_id: str, reason: str, runtime: ToolRuntime[AdvancedRefundAgentState]) -> Command:
    """拒绝当前退款。"""
    payload = {
        "order_id": order_id,
        "reason": reason,
        "status": "declined",
    }
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "refund_status": "declined",
            "requires_human_review": False,
            "current_step": "completed",
        }
    )


REFUND_STAGE_CONFIG = {
    "collect_order": {
        "prompt": """
你是退款客服专员。
当前阶段：收集订单号。
如果用户输入里已经出现订单号，就调用 record_refund_order_id。
如果还没有订单号，直接向用户追问订单号。
""",
        "tools": [record_refund_order_id],
        "requires": [],
    },
    "collect_reason": {
        "prompt": """
你是退款客服专员。
当前阶段：收集退款原因。
当前订单号：{order_id}
如果已经有退款原因，调用 record_refund_reason。
如果还没有，请追问退款原因。
""",
        "tools": [record_refund_reason],
        "requires": ["order_id"],
    },
    "resolve_refund": {
        "prompt": """
你是退款客服专员。
当前阶段：决策退款路径。
订单号：{order_id}
退款原因：{reason}
先调用 check_refund_eligibility。
如果 eligible=false，调用 decline_refund_case。
如果 eligible=true 且 audit_required=true，调用 escalate_refund_case。
如果 eligible=true 且 manual_approval_required=true，调用 approve_high_value_refund_release。
如果 eligible=true 且不需要额外审批，调用 submit_refund_request。
""",
        "tools": [
            check_refund_eligibility,
            approve_high_value_refund_release,
            submit_refund_request,
            escalate_refund_case,
            decline_refund_case,
        ],
        "requires": ["order_id", "reason"],
    },
    "finalize_refund": {
        "prompt": """
你是退款客服专员。
当前阶段：人工审批已通过，请立即调用 submit_refund_request 完成退款。
订单号：{order_id}
退款原因：{reason}
""",
        "tools": [submit_refund_request],
        "requires": ["order_id", "reason"],
    },
    "completed": {
        "prompt": """
你是退款客服专员。
退款流程已经结束，请用一句到两句给用户总结当前结果。
""",
        "tools": [],
        "requires": [],
    },
}


@wrap_model_call
def apply_advanced_refund_stage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
):
    """根据退款阶段动态切换 prompt 和工具。"""
    current_step = request.state.get("current_step", "collect_order")
    stage_config = REFUND_STAGE_CONFIG[current_step]

    for field_name in stage_config["requires"]:
        if request.state.get(field_name) is None:
            raise ValueError(f"{field_name} must be set before reaching {current_step}")

    request = request.override(
        system_prompt=stage_config["prompt"].format(**request.state),
        tools=stage_config["tools"],
    )
    return handler(request)


@tool(args_schema=SendConfirmationArgs)
def send_customer_confirmation(recipient: str, content: str) -> str:
    """发送处理结果确认消息给用户。"""
    payload = {
        "recipient": recipient,
        "content": content,
        "status": "sent",
    }
    NOTIFICATION_OUTBOX.append(payload)
    return json.dumps(payload, ensure_ascii=False)


def create_advanced_refund_agent(model=None):
    """创建进阶版退款 specialist。"""
    return create_agent(
        model or get_default_model(),
        tools=[
            record_refund_order_id,
            record_refund_reason,
            check_refund_eligibility,
            approve_high_value_refund_release,
            submit_refund_request,
            escalate_refund_case,
            decline_refund_case,
        ],
        state_schema=AdvancedRefundAgentState,
        middleware=[
            apply_advanced_refund_stage,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "approve_high_value_refund_release": InterruptOnConfig(
                        allowed_decisions=["approve", "reject"]
                    )
                },
                description_prefix="高金额退款需要人工批准",
            ),
        ],
        checkpointer=InMemorySaver(),
        name="advanced_refund_specialist",
    )


def create_notification_agent(model=None):
    """创建结果通知 agent。"""
    return create_agent(
        model or get_default_model(),
        tools=[send_customer_confirmation],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_customer_confirmation": InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject"]
                    )
                },
                description_prefix="发送处理结果前请人工确认通知内容",
            )
        ],
        system_prompt=(
            "你是通知专员。"
            "收到处理结果后，请整理成一段自然、清晰的客户确认消息，"
            "然后调用 send_customer_confirmation。"
        ),
        checkpointer=InMemorySaver(),
        name="notification_specialist",
    )


def build_support_draft(
    model,
    user_query: str,
    task_results: list[TaskResult],
    clarification_question: str = "",
    review_feedback: str = "",
    force_incomplete: bool = False,
) -> str:
    """汇总多个 task_result，得到给用户看的草稿。"""
    effective_results = task_results
    if force_incomplete and len(task_results) > 1:
        effective_results = task_results[:-1]

    payload = "\n\n".join(
        f"[{item.intent}|{item.status}] {item.answer}"
        for item in effective_results
    )

    prompt = (
        "你是客服总控，请把多个专家结果整合成一段给用户看的自然回复。"
        "要求：先回答已经完成的事项，再列出仍需用户补充的信息；"
        "如果有人工审核，明确说明后续动作。"
        f"\n原始用户问题：{user_query}"
        f"\n专家结果：\n{payload}"
    )
    if clarification_question:
        prompt += f"\n仍需补充：{clarification_question}"
    if review_feedback:
        prompt += f"\nreviewer 反馈：{review_feedback}"

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        if isinstance(response.content, str) and response.content:
            return response.content
    except Exception:
        pass

    lines = [item.answer for item in effective_results if item.answer]
    if clarification_question:
        lines.append(clarification_question)
    if review_feedback:
        lines.append(f"审核修正提示：{review_feedback}")
    return "\n".join(lines)


def review_support_draft(
    model,
    user_query: str,
    task_results: list[TaskResult],
    draft_response: str,
) -> ReviewResult:
    """reviewer 判断草稿是否能结束 loop。"""
    prompt = (
        "你是客服回复审核员。"
        "你需要判断当前草稿是否覆盖了所有 task_results，"
        "是否明确提到了待补信息或人工审核。"
        "如果合格，decision=approved；如果遗漏信息，decision=revise。"
        f"\n原始问题：{user_query}"
        f"\n当前草稿：{draft_response}"
        f"\n任务结果：{json.dumps([item.model_dump(mode='json') for item in task_results], ensure_ascii=False)}"
    )

    try:
        structured_model = model.with_structured_output(ReviewResult)
        return structured_model.invoke(
            [
                SystemMessage(content="请严格输出 ReviewResult。"),
                HumanMessage(content=prompt),
            ]
        )
    except Exception:
        for item in task_results:
            if item.answer and item.answer not in draft_response:
                return ReviewResult(
                    decision="revise",
                    feedback=f"草稿遗漏了 {item.intent} 任务结果，请补上：{item.answer}",
                )
            if item.status == "blocked_waiting_user":
                question = item.follow_up_questions[0] if item.follow_up_questions else item.answer
                if question and question not in draft_response:
                    return ReviewResult(
                        decision="revise",
                        feedback=f"草稿没有明确追问用户：{question}",
                    )
            if item.human_required and "人工" not in draft_response:
                return ReviewResult(
                    decision="revise",
                    feedback="草稿没有说明当前事项已经进入人工审核。",
                )
        return ReviewResult(decision="approved", feedback="草稿已覆盖全部任务结果。")


def run_notification_specialist(
    agent,
    content: str,
    thread_id: str,
    recipient: str = DEFAULT_NOTIFICATION_RECIPIENT,
    decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行通知 specialist，并自动处理审批。"""
    result = invoke_with_auto_human_resolution(
        agent,
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"请把下面的处理结果整理成确认消息，并发送给用户。\n"
                        f"recipient={recipient}\n"
                        f"content={content}"
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
        decision_resolver=decision_resolver,
        execution_trace=execution_trace,
    )
    return {
        "status": "sent",
        "recipient": recipient,
        "message": extract_last_message_text(result),
    }


def run_refund_specialist_advanced(
    agent,
    task: SubTask,
    thread_id: str,
    decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
) -> TaskResult:
    """执行进阶版退款 specialist。"""
    if task.missing_slots:
        return blocked_task_result(task)

    seed_lines = [task.user_text]
    if task.slots.get("order_id"):
        seed_lines.append(f"订单号：{task.slots['order_id']}")
    if task.slots.get("refund_reason"):
        seed_lines.append(f"退款原因：{task.slots['refund_reason']}")

    try:
        result = invoke_with_auto_human_resolution(
            agent,
            {"messages": [HumanMessage(content="\n".join(seed_lines))]},
            config={"configurable": {"thread_id": thread_id}},
            decision_resolver=decision_resolver,
            execution_trace=execution_trace,
        )
        last_text = extract_last_message_text(result)
        refund_status = result.get("refund_status", "pending")
    except Exception:
        order_id = str(task.slots["order_id"])
        reason = str(task.slots["refund_reason"])
        policy = refund_eligibility_service(order_id)
        if not policy["eligible"]:
            return TaskResult(
                task_id=task.task_id,
                intent=task.intent,
                status="done",
                answer=policy["reason"],
                raw_data=policy,
            )
        if policy.get("audit_required"):
            payload = escalate_refund_service(order_id, reason)
            return TaskResult(
                task_id=task.task_id,
                intent=task.intent,
                status="escalated_to_human",
                answer=f"订单 {order_id} 需要人工审核，已创建工单 {payload['ticket_id']}。",
                human_required=True,
                raw_data=payload,
            )
        payload = submit_refund_service(order_id, reason)
        return TaskResult(
            task_id=task.task_id,
            intent=task.intent,
            status="done",
            answer=f"订单 {order_id} 已提交退款申请，退款单号为 {payload['refund_id']}。",
            raw_data=payload,
        )

    if refund_status == "submitted":
        return TaskResult(
            task_id=task.task_id,
            intent=task.intent,
            status="done",
            answer=last_text,
            raw_data={"refund_status": refund_status},
        )
    if refund_status == "human_review":
        return TaskResult(
            task_id=task.task_id,
            intent=task.intent,
            status="escalated_to_human",
            answer=last_text,
            human_required=True,
            raw_data={"refund_status": refund_status},
        )
    if refund_status == "declined":
        return TaskResult(
            task_id=task.task_id,
            intent=task.intent,
            status="done",
            answer=last_text,
            raw_data={"refund_status": refund_status},
        )

    follow_up = last_text or build_clarification_question(task.missing_slots)
    return TaskResult(
        task_id=task.task_id,
        intent=task.intent,
        status="blocked_waiting_user",
        answer=follow_up,
        follow_up_questions=[follow_up],
        raw_data={"refund_status": refund_status},
    )


def run_logistics_specialist_fallback(task: SubTask) -> TaskResult:
    """用于 LangGraph 的无 agent 物流执行。"""
    if task.missing_slots:
        return blocked_task_result(task)
    payload = query_logistics_service(str(task.slots["order_id"]))
    answer = (
        f"订单 {payload['order_id']} 当前物流状态为“{payload['shipping_status']}”，"
        f"运单号 {payload['tracking_no']}，预计送达时间 {payload['eta']}。"
    )
    return TaskResult(
        task_id=task.task_id,
        intent=task.intent,
        status="done",
        answer=answer,
        raw_data=payload,
    )


def run_product_specialist_fallback(task: SubTask) -> TaskResult:
    """用于 LangGraph 的无 agent 商品查询执行。"""
    if task.missing_slots:
        return blocked_task_result(task)

    product_name = str(task.slots.get("product_name", ""))
    if not product_name and task.slots.get("order_id"):
        order = lookup_order_service(str(task.slots["order_id"]))
        product_name = order["product_name"]

    product = search_product_service(product_name)
    highlights = "、".join(product["highlights"])
    answer = (
        f"{product_name} 当前库存 {product['inventory']} 件，售价 {product['price']} 元。"
        f"主要卖点包括：{highlights}。"
    )
    return TaskResult(
        task_id=task.task_id,
        intent="商品咨询",
        status="done",
        answer=answer,
        raw_data=product,
    )
