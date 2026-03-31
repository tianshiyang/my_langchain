#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 1.x 版本：
router + supervisor + specialists + refund handoff
"""
import json
from typing import Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage

from .models import CustomerIntentResult, SupportContext, SubTask, TaskResult
from .support_shared import (
    classify_customer_query,
    create_logistics_agent,
    create_product_agent,
    create_refund_agent,
    get_default_model,
    run_logistics_specialist,
    run_product_specialist,
    run_refund_specialist,
    synthesize_support_response,
)


class LangChainCustomerSupportApp:
    """LangChain 多智能体客服示例。"""

    def __init__(self, model=None):
        self.model = model or get_default_model()
        self.logistics_agent = create_logistics_agent(self.model)
        self.product_agent = create_product_agent(self.model)
        self.refund_agent = create_refund_agent(self.model)
        self.supervisor_agent = self._build_supervisor_agent()

    def _build_supervisor_agent(self):
        app = self

        @tool
        def logistics_specialist(task_json: str) -> str:
            """处理物流查询任务。仅在 task.intent=物流查询 时调用。"""
            task = SubTask.model_validate_json(task_json)
            result = run_logistics_specialist(app.logistics_agent, task)
            return result.model_dump_json(ensure_ascii=False)

        @tool
        def product_specialist(task_json: str) -> str:
            """处理商品咨询任务。仅在 task.intent=商品咨询 时调用。"""
            task = SubTask.model_validate_json(task_json)
            result = run_product_specialist(app.product_agent, task)
            return result.model_dump_json(ensure_ascii=False)

        @tool
        def refund_specialist(task_json: str, runtime: ToolRuntime[SupportContext, Any]) -> str:
            """处理退款任务。仅在 task.intent=退款 时调用。"""
            task = SubTask.model_validate_json(task_json)
            thread_id = runtime.context.thread_id
            refund_thread_id = f"{thread_id}:{task.task_id}"
            result = run_refund_specialist(app.refund_agent, task, refund_thread_id)
            return result.model_dump_json(ensure_ascii=False)

        system_prompt = """
你是客服总控 supervisor。
你会收到一个原始用户问题，以及一个已经生成好的 route_plan JSON。
你的职责：
1. 严格依据 route_plan 决定要调用哪些 specialist。
2. 一个 sub_task 对应一次 specialist 调用。
3. 如果某个 specialist 返回 blocked_waiting_user，不要假装任务已经完成。
4. 只有当所有任务都处于 done、blocked_waiting_user、escalated_to_human 三种终态之一时，才输出最终回复。
5. 最终回复里要：
   - 先回答已经处理完的部分
   - 再向用户追问缺失信息
   - 如果需要人工审核，明确说明后续动作
"""
        return create_agent(
            self.model,
            tools=[logistics_specialist, product_specialist, refund_specialist],
            system_prompt=system_prompt,
            context_schema=SupportContext,
            name="customer_support_supervisor",
        )

    def route(self, user_query: str) -> CustomerIntentResult:
        """对用户问题做结构化路由。"""
        return classify_customer_query(self.model, user_query)

    def handle_request(self, user_query: str, thread_id: str = "demo-thread") -> dict[str, Any]:
        """处理一轮用户请求。"""
        route_plan = self.route(user_query)
        serialized_tasks = [
            task.model_dump_json(ensure_ascii=False)
            for task in route_plan.sub_tasks
        ]
        supervisor_input = {
            "user_query": user_query,
            "route_plan": route_plan.model_dump(mode="json"),
            "serialized_tasks": serialized_tasks,
        }

        control_message = (
            f"用户问题：{user_query}\n\n"
            f"route_plan=\n{json.dumps(route_plan.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
            "请按 route_plan 处理：\n"
            "1. 若某个 task.intent=物流查询，则调用 logistics_specialist(task_json)\n"
            "2. 若某个 task.intent=商品咨询，则调用 product_specialist(task_json)\n"
            "3. 若某个 task.intent=退款，则调用 refund_specialist(task_json)\n"
            "4. 只针对 route_plan.sub_tasks 中存在的任务调用工具\n"
            "5. 最终输出用户可读总结，不要输出思维链\n"
        )

        try:
            result = self.supervisor_agent.invoke(
                {"messages": [HumanMessage(content=control_message)]},
                context=SupportContext(thread_id=thread_id),
            )
            final_answer = getattr(result["messages"][-1], "content", "")
            supervisor_mode = "agent_supervisor"
        except Exception:
            task_results: list[TaskResult] = []
            for task in route_plan.sub_tasks:
                if task.intent == "物流查询":
                    task_results.append(run_logistics_specialist(self.logistics_agent, task))
                elif task.intent == "商品咨询":
                    task_results.append(run_product_specialist(self.product_agent, task))
                elif task.intent == "退款":
                    refund_thread_id = f"{thread_id}:{task.task_id}"
                    task_results.append(run_refund_specialist(self.refund_agent, task, refund_thread_id))

            final_answer = synthesize_support_response(
                self.model,
                user_query,
                task_results,
                clarification_question=route_plan.clarification_question,
            )
            result = {"task_results": task_results}
            supervisor_mode = "deterministic_fallback"
        return {
            "route_plan": route_plan.model_dump(mode="json"),
            "supervisor_result": result,
            "final_answer": final_answer,
            "business_completion_hint": {
                "need_clarification": route_plan.need_clarification,
                "missing_slots": route_plan.missing_slots,
            },
            "execution_mode": supervisor_mode,
            "debug_input": supervisor_input,
        }


if __name__ == "__main__":
    app = LangChainCustomerSupportApp()
    scenarios = [
        "帮我查一下订单 A1001 的物流到哪了",
        "无线耳机 Pro 有货吗，续航怎么样？",
        "订单 A1003 我想退款，原因是买错了",
        "帮我查订单 A1001 的物流，另外我想把 A1003 退掉，原因是买错了",
    ]

    for index, query in enumerate(scenarios, start=1):
        print("\n" + "=" * 100)
        print(f"[LangChain Demo {index}] {query}")
        output = app.handle_request(query, thread_id=f"lc-demo-{index}")
        print(json.dumps(output["route_plan"], ensure_ascii=False, indent=2))
        print("-" * 100)
        print(output["final_answer"])
