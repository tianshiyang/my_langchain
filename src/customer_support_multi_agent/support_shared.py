#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
客服多智能体共享业务逻辑、工具与 agent 工厂。
"""
import json
import os
import re
import uuid
from typing import Any, Callable, Literal, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

from .models import CustomerIntentResult, RefundState, SubTask, TaskResult

# 模拟订单与商品数据库
ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {
        "order_id": "A1001",
        "product_name": "无线耳机 Pro",
        "shipping_status": "运输中",
        "tracking_no": "SF1234567890",
        "eta": "2026-04-02",
        "refund_policy": {
            "eligible": True,
            "audit_required": False,
            "reason": "订单尚未签收，可直接发起拦截退款。",
        },
    },
    "A1002": {
        "order_id": "A1002",
        "product_name": "机械键盘 K2",
        "shipping_status": "已签收",
        "tracking_no": "YT9988776655",
        "eta": "2026-03-10",
        "refund_policy": {
            "eligible": False,
            "audit_required": False,
            "reason": "订单已超过 7 天无理由退款时效，当前不符合自动退款条件。",
        },
    },
    "A1003": {
        "order_id": "A1003",
        "product_name": "显示器 M27",
        "shipping_status": "已签收",
        "tracking_no": "JD5566778899",
        "eta": "2026-03-28",
        "refund_policy": {
            "eligible": True,
            "audit_required": True,
            "reason": "订单金额较高，需要人工审核后才能继续退款。",
        },
    },
    "A1004": {
        "order_id": "A1004",
        "product_name": "显示器 M27",
        "shipping_status": "已签收",
        "tracking_no": "SF9988007766",
        "eta": "2026-03-31",
        "refund_policy": {
            "eligible": True,
            "audit_required": False,
            "manual_approval_required": True,
            "reason": "订单符合退款条件，但因金额较高，退款放行前需要值班主管确认。",
        },
    },
}

PRODUCTS: dict[str, dict[str, Any]] = {
    "无线耳机 Pro": {
        "product_name": "无线耳机 Pro",
        "price": 899,
        "inventory": 23,
        "highlights": [
            "主动降噪",
            "续航 38 小时",
            "支持双设备切换",
        ],
    },
    "机械键盘 K2": {
        "product_name": "机械键盘 K2",
        "price": 469,
        "inventory": 56,
        "highlights": [
            "84 键布局",
            "支持蓝牙与有线双模",
            "Mac / Windows 双系统兼容",
        ],
    },
    "显示器 M27": {
        "product_name": "显示器 M27",
        "price": 2199,
        "inventory": 9,
        "highlights": [
            "27 英寸 4K 面板",
            "Type-C 一线连",
            "支持 90W 反向供电",
        ],
    },
}

LOGISTICS_KEYWORDS = ("物流", "快递", "发货", "配送", "到哪", "运单", "签收")
PRODUCT_KEYWORDS = ("商品", "产品", "规格", "参数", "兼容", "库存", "颜色", "尺寸")
REFUND_KEYWORDS = ("退款", "退货", "退钱", "退掉", "退了", "不想要", "申请退款", "取消订单")
REFUND_REASON_HINTS = (
    "不想要",
    "买错了",
    "质量问题",
    "破损",
    "与描述不符",
    "延迟发货",
)


def get_default_model():
    """当前项目默认模型。"""
    load_dotenv()
    return ChatOpenAI(
        model="MiniMax-M2.7",
        base_url=os.getenv("MINIMAX_BASE_URL"),
        api_key=os.getenv("MINIMAX_API_KEY"),
        temperature=0,
        timeout=20,
        max_tokens=1200,
    )


def extract_order_id(text: str) -> str:
    """提取订单号。"""
    order_ids = extract_order_ids(text)
    return order_ids[0] if order_ids else ""


def extract_order_ids(text: str) -> list[str]:
    """提取文本中的全部订单号。"""
    return re.findall(r"\bA\d{4}\b", text.upper())


def infer_product_name(text: str) -> str:
    """从文本里猜商品名。"""
    for product_name in PRODUCTS:
        if product_name in text:
            return product_name
    return ""


def infer_refund_reason(text: str) -> str:
    """从用户表达中提取退款原因。"""
    for hint in REFUND_REASON_HINTS:
        if hint in text:
            return hint
    return ""


def lookup_order_service(order_id: str) -> dict[str, Any]:
    """查询订单。"""
    order = ORDERS.get(order_id)
    if not order:
        raise ValueError(f"未找到订单 {order_id}")
    return order


def query_logistics_service(order_id: str) -> dict[str, Any]:
    """查询物流。"""
    order = lookup_order_service(order_id)
    return {
        "order_id": order_id,
        "shipping_status": order["shipping_status"],
        "tracking_no": order["tracking_no"],
        "eta": order["eta"],
    }


def search_product_service(product_name: str) -> dict[str, Any]:
    """查询商品信息。"""
    product = PRODUCTS.get(product_name)
    if not product:
        raise ValueError(f"未找到商品 {product_name}")
    return product


def refund_eligibility_service(order_id: str) -> dict[str, Any]:
    """判断退款资格。"""
    order = lookup_order_service(order_id)
    policy = order["refund_policy"]
    return {
        "order_id": order_id,
        "eligible": policy["eligible"],
        "audit_required": policy["audit_required"],
        "manual_approval_required": policy.get("manual_approval_required", False),
        "reason": policy["reason"],
    }


def submit_refund_service(order_id: str, reason: str) -> dict[str, Any]:
    """提交退款。"""
    return {
        "refund_id": f"RF-{uuid.uuid4().hex[:8]}",
        "order_id": order_id,
        "reason": reason,
        "status": "submitted",
    }


def escalate_refund_service(order_id: str, reason: str) -> dict[str, Any]:
    """升级人工审核。"""
    return {
        "ticket_id": f"TK-{uuid.uuid4().hex[:8]}",
        "order_id": order_id,
        "reason": reason,
        "status": "human_review",
    }


@tool
def lookup_order(order_id: str) -> str:
    """根据订单号查询订单详情。"""
    return json.dumps(lookup_order_service(order_id), ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流状态。"""
    return json.dumps(query_logistics_service(order_id), ensure_ascii=False)


@tool
def search_product_info(product_name: str) -> str:
    """根据商品名查询商品资料、价格、库存与卖点。"""
    return json.dumps(search_product_service(product_name), ensure_ascii=False)


@tool
def check_refund_eligibility(order_id: str) -> str:
    """检查订单是否符合退款条件。"""
    return json.dumps(refund_eligibility_service(order_id), ensure_ascii=False)


def build_clarification_question(missing_slots: list[str]) -> str:
    """将缺失槽位转成用户可读追问。"""
    questions = []
    if "order_id" in missing_slots:
        questions.append("请提供订单号")
    if "product_name_or_order_id" in missing_slots:
        questions.append("请告诉我商品名或订单号")
    if "refund_reason" in missing_slots:
        questions.append("请补充退款原因")
    if not questions:
        return "请补充更多信息，我才能继续处理你的问题。"
    return "；".join(dict.fromkeys(questions)) + "。"


def make_subtask(intent: str, user_text: str) -> SubTask:
    """构造任务，并补齐槽位。"""
    order_ids = extract_order_ids(user_text)
    order_id = ""
    if order_ids:
        if intent == "退款":
            order_id = order_ids[-1]
        else:
            order_id = order_ids[0]
    product_name = infer_product_name(user_text)
    refund_reason = infer_refund_reason(user_text)

    slots: dict[str, Any] = {}
    missing_slots: list[str] = []

    if intent == "物流查询":
        if order_id:
            slots["order_id"] = order_id
        else:
            missing_slots.append("order_id")
    elif intent == "商品咨询":
        if product_name:
            slots["product_name"] = product_name
        elif order_id:
            slots["order_id"] = order_id
        else:
            missing_slots.append("product_name_or_order_id")
    elif intent == "退款":
        if order_id:
            slots["order_id"] = order_id
        else:
            missing_slots.append("order_id")

        if refund_reason:
            slots["refund_reason"] = refund_reason
        else:
            missing_slots.append("refund_reason")

    return SubTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        intent=intent,  # type: ignore[arg-type]
        user_text=user_text,
        slots=slots,
        missing_slots=missing_slots,
    )


def keyword_classify_customer_query(user_query: str) -> CustomerIntentResult:
    """无模型兜底路由。"""
    intents: list[str] = []
    for keyword in LOGISTICS_KEYWORDS:
        if keyword in user_query:
            intents.append("物流查询")
            break
    for keyword in PRODUCT_KEYWORDS:
        if keyword in user_query:
            intents.append("商品咨询")
            break
    for keyword in REFUND_KEYWORDS:
        if keyword in user_query:
            intents.append("退款")
            break

    if not intents:
        if "订单" in user_query:
            return CustomerIntentResult(
                intents=[],
                sub_tasks=[],
                missing_slots=[],
                need_clarification=True,
                clarification_question="你是想查物流、咨询商品，还是申请退款？",
                confidence=0.35,
            )
        return CustomerIntentResult(
            intents=[],
            sub_tasks=[],
            missing_slots=[],
            need_clarification=True,
            clarification_question="我先帮你确认一下，你是要查物流、问商品信息，还是处理退款？",
            confidence=0.2,
        )

    sub_tasks = [make_subtask(intent, user_query) for intent in dict.fromkeys(intents)]
    missing_slots = sorted({slot for task in sub_tasks for slot in task.missing_slots})
    need_clarification = bool(missing_slots)
    clarification_question = build_clarification_question(missing_slots) if need_clarification else ""

    return CustomerIntentResult(
        intents=list(dict.fromkeys(intents)),  # type: ignore[list-item]
        sub_tasks=sub_tasks,
        missing_slots=missing_slots,
        need_clarification=need_clarification,
        clarification_question=clarification_question,
        confidence=0.72,
    )


def normalize_intent_result(result: CustomerIntentResult, user_query: str) -> CustomerIntentResult:
    """修正 LLM 分类结果里的槽位与缺失字段。"""
    if not result.intents:
        fallback = keyword_classify_customer_query(user_query)
        return fallback

    normalized_tasks: list[SubTask] = []
    for task in result.sub_tasks or [make_subtask(intent, user_query) for intent in result.intents]:
        slots = task.slots or {}
        missing_slots = list(task.missing_slots)

        if task.intent == "物流查询" and not slots.get("order_id"):
            inferred_orders = extract_order_ids(task.user_text or user_query)
            inferred = inferred_orders[0] if inferred_orders else ""
            if inferred:
                slots["order_id"] = inferred
            elif "order_id" not in missing_slots:
                missing_slots.append("order_id")

        if task.intent == "商品咨询":
            if not slots.get("product_name") and not slots.get("order_id"):
                inferred_product = infer_product_name(task.user_text or user_query)
                inferred_orders = extract_order_ids(task.user_text or user_query)
                inferred_order = inferred_orders[0] if inferred_orders else ""
                if inferred_product:
                    slots["product_name"] = inferred_product
                elif inferred_order:
                    slots["order_id"] = inferred_order
                elif "product_name_or_order_id" not in missing_slots:
                    missing_slots.append("product_name_or_order_id")

        if task.intent == "退款":
            inferred_orders = extract_order_ids(task.user_text or user_query)
            inferred_order = slots.get("order_id") or (inferred_orders[-1] if inferred_orders else "")
            inferred_reason = slots.get("refund_reason") or infer_refund_reason(task.user_text or user_query)
            if inferred_order:
                slots["order_id"] = inferred_order
            elif "order_id" not in missing_slots:
                missing_slots.append("order_id")

            if inferred_reason:
                slots["refund_reason"] = inferred_reason
            elif "refund_reason" not in missing_slots:
                missing_slots.append("refund_reason")

        normalized_tasks.append(
            SubTask(
                task_id=task.task_id or f"task_{uuid.uuid4().hex[:8]}",
                intent=task.intent,
                user_text=task.user_text or user_query,
                slots=slots,
                missing_slots=sorted(set(missing_slots)),
            )
        )

    missing_slots = sorted({slot for task in normalized_tasks for slot in task.missing_slots})
    need_clarification = bool(missing_slots)

    return CustomerIntentResult(
        intents=list(dict.fromkeys(result.intents)),
        sub_tasks=normalized_tasks,
        missing_slots=missing_slots,
        need_clarification=need_clarification,
        clarification_question=result.clarification_question or (
            build_clarification_question(missing_slots) if need_clarification else ""
        ),
        confidence=result.confidence or 0.7,
    )


def classify_customer_query(model, user_query: str) -> CustomerIntentResult:
    """优先走结构化输出，失败时回退到规则路由。"""
    router_prompt = (
        "你是电商客服系统的路由器。"
        "请从用户输入中识别一个或多个客服意图，只能从 物流查询、商品咨询、退款 里选择。"
        "如果信息不足，也要返回已有意图，并在 missing_slots 中指出缺的槽位。"
        "物流查询至少需要 order_id；商品咨询至少需要 product_name 或 order_id；"
        "退款至少需要 order_id 和 refund_reason。"
        "请生成 sub_tasks，每个 task 只对应一个 intent。"
    )
    try:
        structured_model = model.with_structured_output(CustomerIntentResult)
        result = structured_model.invoke(
            [
                SystemMessage(content=router_prompt),
                HumanMessage(content=user_query),
            ]
        )
        return normalize_intent_result(result, user_query)
    except Exception:
        return keyword_classify_customer_query(user_query)


def extract_last_message_text(result: dict[str, Any]) -> str:
    """从 agent 结果中提取最后一条文本消息。"""
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content:
            return content
    return ""


def blocked_task_result(task: SubTask) -> TaskResult:
    """缺槽位时的统一阻塞结果。"""
    question = build_clarification_question(task.missing_slots)
    return TaskResult(
        task_id=task.task_id,
        intent=task.intent,
        status="blocked_waiting_user",
        answer=question,
        follow_up_questions=[question],
        human_required=False,
        raw_data={"missing_slots": task.missing_slots},
    )


LOGISTICS_AGENT_PROMPT = """
你是物流客服专家。
你的职责：
1. 优先根据订单号查询订单和物流信息。
2. 告诉用户当前物流状态、运单号和预计送达时间。
3. 如果没有订单号，不要编造，明确说明还需要订单号。
4. 最终回答要简洁、准确、像客服一样自然。
"""

PRODUCT_AGENT_PROMPT = """
你是商品咨询客服专家。
你的职责：
1. 回答商品规格、库存、兼容性、卖点等问题。
2. 如果用户给的是订单号，可以先查订单对应商品，再回答。
3. 如果既没有商品名也没有订单号，不要编造，请明确说明需要补充信息。
4. 最终回答要面向用户，而不是输出原始 JSON。
"""


class RefundAgentState(AgentState):
    """退款 handoff agent 的状态。"""

    current_step: NotRequired[Literal["collect_order", "collect_reason", "resolve_refund", "completed"]]
    order_id: NotRequired[str]
    reason: NotRequired[str]
    requires_human_review: NotRequired[bool]
    refund_status: NotRequired[Literal["pending", "submitted", "human_review", "declined"]]


@tool
def record_refund_order_id(order_id: str, runtime: ToolRuntime) -> Command:
    """记录退款订单号并进入退款原因收集阶段。"""
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
def record_refund_reason(reason: str, runtime: ToolRuntime) -> Command:
    """记录退款原因并进入退款决策阶段。"""
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
def submit_refund_request(order_id: str, reason: str, runtime: ToolRuntime) -> Command:
    """提交自动退款请求。"""
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
def escalate_refund_case(order_id: str, reason: str, runtime: ToolRuntime) -> Command:
    """将退款案件升级到人工审核。"""
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
def decline_refund_case(order_id: str, reason: str, runtime: ToolRuntime) -> Command:
    """结束退款流程并说明当前不符合退款条件。"""
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
请识别用户输入里是否已经有订单号。
如果有，调用 record_refund_order_id。
如果没有，直接向用户索取订单号。
一次只推进一个最关键字段。
""",
        "tools": [record_refund_order_id],
        "requires": [],
    },
    "collect_reason": {
        "prompt": """
你是退款客服专员。
当前阶段：收集退款原因。
当前订单号：{order_id}
如果用户已经给出退款原因，调用 record_refund_reason。
如果没有，请简洁地追问退款原因。
""",
        "tools": [record_refund_reason],
        "requires": ["order_id"],
    },
    "resolve_refund": {
        "prompt": """
你是退款客服专员。
当前阶段：退款决策。
订单号：{order_id}
退款原因：{reason}
先调用 check_refund_eligibility 检查退款资格。
如果 eligible=true 且 audit_required=false，调用 submit_refund_request。
如果 eligible=true 且 audit_required=true，调用 escalate_refund_case。
如果 eligible=false，调用 decline_refund_case。
最终回答里必须明确告诉用户当前结果。
""",
        "tools": [
            check_refund_eligibility,
            submit_refund_request,
            escalate_refund_case,
            decline_refund_case,
        ],
        "requires": ["order_id", "reason"],
    },
    "completed": {
        "prompt": """
你是退款客服专员。
退款流程已经完成，请用一句到两句总结最终结果并结束本轮对话。
""",
        "tools": [],
        "requires": [],
    },
}


@wrap_model_call
def apply_refund_stage(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
    """根据退款阶段动态切换 prompt 和工具。"""
    current_step = request.state.get("current_step", "collect_order")
    stage_config = REFUND_STAGE_CONFIG[current_step]

    for field_name in stage_config["requires"]:
        if request.state.get(field_name) is None:
            raise ValueError(f"{field_name} must be set before reaching {current_step}")

    system_prompt = stage_config["prompt"].format(**request.state)
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )
    return handler(request)


def create_logistics_agent(model=None):
    """创建物流专家 agent。"""
    return create_agent(
        model or get_default_model(),
        tools=[lookup_order, query_logistics],
        system_prompt=LOGISTICS_AGENT_PROMPT,
        name="logistics_specialist",
    )


def create_product_agent(model=None):
    """创建商品咨询专家 agent。"""
    return create_agent(
        model or get_default_model(),
        tools=[lookup_order, search_product_info],
        system_prompt=PRODUCT_AGENT_PROMPT,
        name="product_specialist",
    )


def create_refund_agent(model=None):
    """创建退款 handoff agent。"""
    return create_agent(
        model or get_default_model(),
        tools=[
            record_refund_order_id,
            record_refund_reason,
            check_refund_eligibility,
            submit_refund_request,
            escalate_refund_case,
            decline_refund_case,
        ],
        state_schema=RefundAgentState,
        middleware=[apply_refund_stage],
        checkpointer=InMemorySaver(),
        name="refund_specialist",
    )


def run_logistics_specialist(agent, task: SubTask) -> TaskResult:
    """执行物流任务。"""
    if task.missing_slots:
        return blocked_task_result(task)

    order_id = str(task.slots["order_id"])
    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"{task.user_text}\n已知订单号：{order_id}"
                    )
                ]
            }
        )
        answer = extract_last_message_text(result)
    except Exception:
        payload = query_logistics_service(order_id)
        answer = (
            f"订单 {order_id} 当前物流状态为“{payload['shipping_status']}”，"
            f"运单号 {payload['tracking_no']}，预计送达时间 {payload['eta']}。"
        )
    return TaskResult(
        task_id=task.task_id,
        intent=task.intent,
        status="done",
        answer=answer,
        raw_data={"order_id": order_id},
    )


def run_product_specialist(agent, task: SubTask) -> TaskResult:
    """执行商品咨询任务。"""
    if task.missing_slots:
        return blocked_task_result(task)

    known = []
    if task.slots.get("product_name"):
        known.append(f"已知商品名：{task.slots['product_name']}")
    if task.slots.get("order_id"):
        known.append(f"已知订单号：{task.slots['order_id']}")

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"{task.user_text}\n" + "\n".join(known)
                    )
                ]
            }
        )
        answer = extract_last_message_text(result)
    except Exception:
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
        intent=task.intent,
        status="done",
        answer=answer,
        raw_data=task.slots,
    )


def run_refund_specialist(agent, task: SubTask, thread_id: str) -> TaskResult:
    """执行退款任务。"""
    if task.missing_slots:
        return blocked_task_result(task)

    seed_lines = [task.user_text]
    if task.slots.get("order_id"):
        seed_lines.append(f"订单号：{task.slots['order_id']}")
    if task.slots.get("refund_reason"):
        seed_lines.append(f"退款原因：{task.slots['refund_reason']}")

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content="\n".join(seed_lines))]},
            {"configurable": {"thread_id": thread_id}},
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
        if policy["audit_required"]:
            payload = escalate_refund_service(order_id, reason)
            return TaskResult(
                task_id=task.task_id,
                intent=task.intent,
                status="escalated_to_human",
                answer=(
                    f"订单 {order_id} 需要人工审核，"
                    f"已创建工单 {payload['ticket_id']}。"
                ),
                human_required=True,
                raw_data=payload,
            )
        payload = submit_refund_service(order_id, reason)
        return TaskResult(
            task_id=task.task_id,
            intent=task.intent,
            status="done",
            answer=(
                f"订单 {order_id} 已提交退款申请，"
                f"退款单号为 {payload['refund_id']}。"
            ),
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


def synthesize_support_response(
    model,
    user_query: str,
    task_results: list[TaskResult],
    clarification_question: str = "",
) -> str:
    """合并多个任务结果。"""
    if not task_results and clarification_question:
        return clarification_question

    payload = "\n\n".join(
        f"[{item.intent}|{item.status}] {item.answer}"
        for item in task_results
    )

    prompt = (
        "你是客服总控，请把多个专家结果整合成一段给用户看的自然回复。"
        "要求：先回答已完成的问题，再提出仍需用户补充的信息。"
        "如果有人工审核，也要明确告诉用户接下来会发生什么。"
        f"\n原始用户问题：{user_query}"
        f"\n专家结果：\n{payload}"
    )
    if clarification_question:
        prompt += f"\n需要补充的问题：{clarification_question}"

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        lines = [item.answer for item in task_results if item.answer]
        if clarification_question:
            lines.append(clarification_question)
        return "\n".join(lines)

if __name__ == "__main__":
    model = get_default_model()
    resp = model.invoke("你好")
    print(resp)
