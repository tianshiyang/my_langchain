#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
客服多智能体共享类型。
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


IntentType = Literal["物流查询", "商品咨询", "退款"]
TaskTerminalStatus = Literal["done", "blocked_waiting_user", "escalated_to_human"]


class SubTask(BaseModel):
    """意图识别后的原子任务。"""

    task_id: str
    intent: IntentType
    user_text: str
    slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list) # 缺哪些关键信息


class CustomerIntentResult(BaseModel):
    """前门路由契约。"""

    intents: list[IntentType] = Field(default_factory=list) # 识别到了哪些意图
    sub_tasks: list[SubTask] = Field(default_factory=list) # 每个原子任务一个 SubTask
    missing_slots: list[str] = Field(default_factory=list) # 缺哪些关键信息
    need_clarification: bool = False # 是否需要追问
    clarification_question: str = "" # 应该怎么追问用户
    confidence: float = 0.0


class TaskResult(BaseModel):
    """专家 agent 的统一输出。"""

    task_id: str
    intent: IntentType
    status: TaskTerminalStatus
    answer: str
    follow_up_questions: list[str] = Field(default_factory=list)
    human_required: bool = False
    raw_data: dict[str, Any] = Field(default_factory=dict)


class RefundState(BaseModel):
    """退款子流程共享状态。"""

    order_id: str = ""
    eligibility: str = "unknown"
    reason: str = ""
    audit_required: bool = False
    refund_status: str = "pending"


class SupportContext(BaseModel):
    """LangChain supervisor 的上下文。"""

    thread_id: str = "default-thread"


def all_results_terminal(task_results: list[TaskResult]) -> bool:
    """是否全部处于可结束状态。"""
    return all(
        task.status in {"done", "blocked_waiting_user", "escalated_to_human"}
        for task in task_results
    )
