#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 进阶版：
router -> supervisor agent loop -> specialists -> reviewer -> notification
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .advanced_shared import (
    DEFAULT_NOTIFICATION_RECIPIENT,
    build_support_draft,
    create_advanced_refund_agent,
    create_notification_agent,
    review_support_draft,
    run_notification_specialist,
    run_refund_specialist_advanced,
)
from .models import (
    MAX_REVIEW_ROUNDS,
    MAX_SUPERVISOR_ROUNDS,
    ReviewResult,
    SubTask,
    SupportContext,
    TaskResult,
    all_results_terminal,
    should_stop_agent_loop,
)
from .support_shared import (
    blocked_task_result,
    classify_customer_query,
    create_logistics_agent,
    create_product_agent,
    get_default_model,
    run_logistics_specialist,
    run_product_specialist,
)


class SpecialistTaskArgs(BaseModel):
    """supervisor 调 specialist 时使用的参数。"""

    task_id: str = Field(description="必须是 route_plan 里的 task_id")


class LangChainAgentLoopMultiAgentApp:
    """LangChain 版进阶教学 demo。"""

    def __init__(self, model=None):
        self.model = model or get_default_model()
        self.logistics_agent = create_logistics_agent(self.model)
        self.product_agent = create_product_agent(self.model)
        self.refund_agent = create_advanced_refund_agent(self.model)
        self.notification_agent = create_notification_agent(self.model)

    def _run_specialist(
        self,
        task: SubTask,
        thread_id: str,
        decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
        execution_trace: list[dict[str, Any]],
    ) -> TaskResult:
        """按 task.intent 调度对应 specialist。"""
        if task.intent == "物流查询":
            return run_logistics_specialist(self.logistics_agent, task)
        if task.intent == "商品咨询":
            return run_product_specialist(self.product_agent, task)
        if task.intent == "退款":
            return run_refund_specialist_advanced(
                self.refund_agent,
                task,
                thread_id=f"{thread_id}:{task.task_id}",
                decision_resolver=decision_resolver,
                execution_trace=execution_trace,
            )
        raise ValueError(f"未知意图：{task.intent}")

    def _build_supervisor_agent(
        self,
        task_lookup: dict[str, SubTask],
        task_results: dict[str, TaskResult],
        execution_trace: list[dict[str, Any]],
        thread_id: str,
        decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    ):
        """构建当前请求专属的 supervisor agent。"""
        app = self

        @tool(args_schema=SpecialistTaskArgs)
        def logistics_specialist(task_id: str) -> str:
            """处理物流查询 task。"""
            task = task_lookup[task_id]
            if task_id in task_results:
                return task_results[task_id].model_dump_json(ensure_ascii=False)

            execution_trace.append(
                {"stage": "supervisor_tool_call", "tool": "logistics_specialist", "task_id": task_id}
            )
            result = app._run_specialist(task, thread_id, decision_resolver, execution_trace)
            task_results[task_id] = result
            execution_trace.append(
                {
                    "stage": "specialist_done",
                    "tool": "logistics_specialist",
                    "task_id": task_id,
                    "status": result.status,
                }
            )
            return result.model_dump_json(ensure_ascii=False)

        @tool(args_schema=SpecialistTaskArgs)
        def product_specialist(task_id: str) -> str:
            """处理商品咨询 task。"""
            task = task_lookup[task_id]
            if task_id in task_results:
                return task_results[task_id].model_dump_json(ensure_ascii=False)

            execution_trace.append(
                {"stage": "supervisor_tool_call", "tool": "product_specialist", "task_id": task_id}
            )
            result = app._run_specialist(task, thread_id, decision_resolver, execution_trace)
            task_results[task_id] = result
            execution_trace.append(
                {
                    "stage": "specialist_done",
                    "tool": "product_specialist",
                    "task_id": task_id,
                    "status": result.status,
                }
            )
            return result.model_dump_json(ensure_ascii=False)

        @tool(args_schema=SpecialistTaskArgs)
        def refund_specialist(task_id: str, runtime: ToolRuntime[SupportContext, Any]) -> str:
            """处理退款 task。"""
            task = task_lookup[task_id]
            if task_id in task_results:
                return task_results[task_id].model_dump_json(ensure_ascii=False)

            refund_thread_id = f"{runtime.context.thread_id}:{task.task_id}"
            execution_trace.append(
                {
                    "stage": "supervisor_tool_call",
                    "tool": "refund_specialist",
                    "task_id": task_id,
                    "refund_thread_id": refund_thread_id,
                }
            )
            result = app._run_specialist(task, runtime.context.thread_id, decision_resolver, execution_trace)
            task_results[task_id] = result
            execution_trace.append(
                {
                    "stage": "specialist_done",
                    "tool": "refund_specialist",
                    "task_id": task_id,
                    "status": result.status,
                    "human_required": result.human_required,
                }
            )
            return result.model_dump_json(ensure_ascii=False)

        system_prompt = """
你是电商客服总控 supervisor。
你会收到 route_plan、pending task 列表、以及已经完成的 task_results。
你的职责：
1. 只处理 pending task。
2. 物流查询调用 logistics_specialist(task_id)。
3. 商品咨询调用 product_specialist(task_id)。
4. 退款调用 refund_specialist(task_id)。
5. 一个 task 最多处理一次，不要重复调用已完成任务。
6. 当所有 pending task 都已经处理完时，停止工具调用并用一句话说明“任务已齐备，可进入汇总”。
"""
        return create_agent(
            self.model,
            tools=[logistics_specialist, product_specialist, refund_specialist],
            system_prompt=system_prompt,
            context_schema=SupportContext,
            name="advanced_customer_support_supervisor",
        )

    def handle_request(
        self,
        user_query: str,
        thread_id: str = "lc-advanced-demo",
        recipient: str = DEFAULT_NOTIFICATION_RECIPIENT,
        force_reviewer_revision: bool = False,
        decision_resolver: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """处理一轮复合售后请求。"""
        route_plan = classify_customer_query(self.model, user_query)
        task_lookup = {task.task_id: task for task in route_plan.sub_tasks}
        task_results: dict[str, TaskResult] = {}
        execution_trace: list[dict[str, Any]] = [
            {
                "stage": "route",
                "intents": route_plan.intents,
                "need_clarification": route_plan.need_clarification,
                "sub_task_ids": [task.task_id for task in route_plan.sub_tasks],
            }
        ]

        for task in route_plan.sub_tasks:
            if task.missing_slots:
                blocked = blocked_task_result(task)
                task_results[task.task_id] = blocked
                execution_trace.append(
                    {
                        "stage": "task_blocked",
                        "task_id": task.task_id,
                        "intent": task.intent,
                        "missing_slots": task.missing_slots,
                    }
                )

        pending_task_ids = [
            task.task_id for task in route_plan.sub_tasks if task.task_id not in task_results
        ]
        supervisor_round = 0
        supervisor_agent = self._build_supervisor_agent(
            task_lookup,
            task_results,
            execution_trace,
            thread_id,
            decision_resolver,
        )

        while pending_task_ids and supervisor_round < MAX_SUPERVISOR_ROUNDS:
            supervisor_round += 1
            before_ids = set(task_results)
            control_message = (
                f"用户问题：{user_query}\n\n"
                f"route_plan=\n{json.dumps(route_plan.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
                f"pending_task_ids={json.dumps(pending_task_ids, ensure_ascii=False)}\n"
                f"已完成结果={json.dumps([item.model_dump(mode='json') for item in task_results.values()], ensure_ascii=False)}\n"
                "请继续推进 pending task。"
            )
            execution_trace.append(
                {
                    "stage": "supervisor_round",
                    "round": supervisor_round,
                    "pending_task_ids": pending_task_ids[:],
                }
            )
            try:
                supervisor_agent.invoke(
                    {"messages": [HumanMessage(content=control_message)]},
                    context=SupportContext(thread_id=thread_id),
                )
            except Exception as exc:
                execution_trace.append(
                    {
                        "stage": "supervisor_error",
                        "round": supervisor_round,
                        "error": str(exc),
                    }
                )

            pending_task_ids = [
                task_id for task_id in pending_task_ids if task_id not in task_results
            ]
            if before_ids == set(task_results) and pending_task_ids:
                fallback_task_id = pending_task_ids[0]
                fallback_task = task_lookup[fallback_task_id]
                result = self._run_specialist(
                    fallback_task,
                    thread_id,
                    decision_resolver,
                    execution_trace,
                )
                task_results[fallback_task_id] = result
                execution_trace.append(
                    {
                        "stage": "supervisor_guard_fallback",
                        "task_id": fallback_task_id,
                        "status": result.status,
                    }
                )
                pending_task_ids = [
                    task_id for task_id in pending_task_ids if task_id not in task_results
                ]

        if pending_task_ids:
            for fallback_task_id in pending_task_ids:
                fallback_task = task_lookup[fallback_task_id]
                result = self._run_specialist(
                    fallback_task,
                    thread_id,
                    decision_resolver,
                    execution_trace,
                )
                task_results[fallback_task_id] = result
                execution_trace.append(
                    {
                        "stage": "supervisor_post_limit_fallback",
                        "task_id": fallback_task_id,
                        "status": result.status,
                    }
                )
            pending_task_ids = []

        ordered_results = [task_results[task.task_id] for task in route_plan.sub_tasks]

        review_feedback = ""
        review_result = ReviewResult(decision="revise", feedback="尚未审核")
        draft_response = ""
        final_response = ""

        for review_round in range(1, MAX_REVIEW_ROUNDS + 1):
            draft_response = build_support_draft(
                self.model,
                user_query,
                ordered_results,
                clarification_question=route_plan.clarification_question,
                review_feedback=review_feedback,
                force_incomplete=force_reviewer_revision and review_round == 1,
            )
            review_result = review_support_draft(
                self.model,
                user_query,
                ordered_results,
                draft_response,
            )
            execution_trace.append(
                {
                    "stage": "review_round",
                    "round": review_round,
                    "decision": review_result.decision,
                    "feedback": review_result.feedback,
                }
            )
            if should_stop_agent_loop(
                ordered_results,
                review_result=review_result,
                supervisor_round=supervisor_round,
                review_round=review_round,
            ):
                final_response = draft_response
                break
            review_feedback = review_result.feedback

        if not final_response:
            final_response = draft_response
            if review_result.feedback:
                final_response += f"\n\n审核补充：{review_result.feedback}"

        notification_result: dict[str, Any] = {}
        try:
            notification_result = run_notification_specialist(
                self.notification_agent,
                content=final_response,
                recipient=recipient,
                thread_id=f"{thread_id}:notification",
                decision_resolver=decision_resolver,
                execution_trace=execution_trace,
            )
        except Exception as exc:
            notification_result = {
                "status": "failed",
                "recipient": recipient,
                "message": f"通知发送失败：{exc}",
            }
            execution_trace.append(
                {
                    "stage": "notification_error",
                    "error": str(exc),
                }
            )

        return {
            "route_plan": route_plan.model_dump(mode="json"),
            "task_results": [item.model_dump(mode="json") for item in ordered_results],
            "review_result": review_result.model_dump(mode="json"),
            "final_response": final_response,
            "notification_result": notification_result,
            "execution_trace": execution_trace,
            "loop_stop_reason": {
                "all_results_terminal": all_results_terminal(ordered_results),
                "review_approved": review_result.decision == "approved",
                "supervisor_rounds": supervisor_round,
            },
        }


if __name__ == "__main__":
    app = LangChainAgentLoopMultiAgentApp()
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
        print(f"[LangChain Advanced Demo {index}] {query}")
        output = app.handle_request(
            query,
            thread_id=f"lc-advanced-{index}",
            force_reviewer_revision=index == 1,
        )
        print(json.dumps(output["route_plan"], ensure_ascii=False, indent=2))
        print("-" * 100)
        print(output["final_response"])
        print("-" * 100)
        print(json.dumps(output["review_result"], ensure_ascii=False, indent=2))
        print("-" * 100)
        print(json.dumps(output["notification_result"], ensure_ascii=False, indent=2))
