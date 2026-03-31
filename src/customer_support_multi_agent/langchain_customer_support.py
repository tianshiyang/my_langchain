#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 1.x 版本：
router + supervisor + specialists + refund handoff

整体架构：
1. route() - 接收用户原始问题，通过 LLM 判断意图并生成执行计划（route_plan）
2. handle_request() - 由 supervisor agent 驱动，根据 route_plan 调用对应 specialist
3. specialist（物流、商品、退款）各自处理一类细分任务
4. 最终由 synthesize_support_response() 汇总结果返回给用户

异常降级机制：
- 若 supervisor agent 调用失败（如模型服务不可用），会回退到确定性分发模式，
  直接遍历 sub_tasks 并串行调用对应 specialist，不再经过 supervisor。
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
    """LangChain 多智能体客服示例。

    组成：
    - logistics_agent：处理物流查询（订单轨迹、预计到达时间等）
    - product_agent：处理商品咨询（库存、价格、参数等）
    - refund_agent：处理退款申请（含独立 thread 隔离）
    - supervisor_agent：总控智能体，负责根据 route_plan 分发任务

    使用流程：
        app = LangChainCustomerSupportApp()
        result = app.handle_request("帮我查一下订单 A1001 的物流到哪了")
    """

    def __init__(self, model=None):
        # 默认使用共享模型，也可传入自定义模型（如 GPT-4、Claude 等）
        self.model = model or get_default_model()
        # 创建各专业领域的 agent 实例
        self.logistics_agent = create_logistics_agent(self.model)
        self.product_agent = create_product_agent(self.model)
        self.refund_agent = create_refund_agent(self.model)
        # 构建总控 supervisor agent
        self.supervisor_agent = self._build_supervisor_agent()

    def _build_supervisor_agent(self):
        """构建 supervisor agent，绑定三个 specialist tools。

        supervisor 的核心职责是根据 route_plan 决定调用哪个 tool，
        并在所有任务达到终态（done / blocked_waiting_user / escalated_to_human）时
        汇总输出最终回复给用户。
        """
        app = self

        from pydantic import BaseModel

        class LogisticsSpecialistArgs(BaseModel):
            """物流查询工具的参数。"""
            task_json: str

        class ProductSpecialistArgs(BaseModel):
            """商品咨询工具的参数。"""
            task_json: str

        class RefundSpecialistArgs(BaseModel):
            """退款工具的参数。

            注意：thread_id 不在此声明，由 LangChain 的 ToolRuntime 框架注入，
            通过 runtime.context.thread_id 获取。
            """
            task_json: str

        @tool(args_schema=LogisticsSpecialistArgs)
        def logistics_specialist(task_json: str) -> str:
            """处理物流查询任务。仅在 task.intent=物流查询 时调用。"""
            print(f"\n[TOOL CALL] logistics_specialist 被调用")
            print(f"  └─ 输入参数 task_json: {task_json}")
            task = SubTask.model_validate_json(task_json)
            print(f"  └─ 解析后 task: task_id={task.task_id}, intent={task.intent}, slots={task.slots}")
            print(f"[TOOL START] logistics_specialist 开始执行...")
            result = run_logistics_specialist(app.logistics_agent, task)
            print(f"[TOOL DONE] logistics_specialist 返回 status={result.status}")
            print(f"  └─ answer: {result.answer[:80]}{'...' if len(result.answer) > 80 else ''}")
            return result.model_dump_json(ensure_ascii=False)

        @tool(args_schema=ProductSpecialistArgs)
        def product_specialist(task_json: str) -> str:
            """处理商品咨询任务。仅在 task.intent=商品咨询 时调用。"""
            print(f"\n[TOOL CALL] product_specialist 被调用")
            print(f"  └─ 输入参数 task_json: {task_json}")
            task = SubTask.model_validate_json(task_json)
            print(f"  └─ 解析后 task: task_id={task.task_id}, intent={task.intent}, slots={task.slots}")
            print(f"[TOOL START] product_specialist 开始执行...")
            result = run_product_specialist(app.product_agent, task)
            print(f"[TOOL DONE] product_specialist 返回 status={result.status}")
            print(f"  └─ answer: {result.answer[:80]}{'...' if len(result.answer) > 80 else ''}")
            return result.model_dump_json(ensure_ascii=False)

        @tool(args_schema=RefundSpecialistArgs)
        def refund_specialist(task_json: str, runtime: ToolRuntime[SupportContext, Any]) -> str:
            """处理退款任务。仅在 task.intent=退款 时调用。

            thread_id 通过 runtime.context.thread_id 获取，
            拼接为层级化 ID "{原始thread_id}:{task_id}" 以隔离不同退款请求。
            """
            print(f"\n[TOOL CALL] refund_specialist 被调用")
            print(f"  └─ 输入参数 task_json: {task_json}")
            thread_id = runtime.context.thread_id
            print(f"  └─ runtime context thread_id: {thread_id}")
            task = SubTask.model_validate_json(task_json)
            refund_thread_id = f"{thread_id}:{task.task_id}"
            print(f"  └─ 解析后 task: task_id={task.task_id}, intent={task.intent}, slots={task.slots}")
            print(f"  └─ 拼接 refund_thread_id: {refund_thread_id}")
            print(f"[TOOL START] refund_specialist 开始执行...")
            result = run_refund_specialist(app.refund_agent, task, refund_thread_id)
            print(f"[TOOL DONE] refund_specialist 返回 status={result.status}, human_required={result.human_required}")
            print(f"  └─ answer: {result.answer[:80]}{'...' if len(result.answer) > 80 else ''}")
            return result.model_dump_json(ensure_ascii=False)

        # supervisor 的 system prompt，定义其行为约束
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
        """对用户问题做结构化路由。

        内部调用 classify_customer_query，根据 LLM 判断结果返回：
        - intent：总体意图分类
        - sub_tasks：需要执行的子任务列表
        - need_clarification / missing_slots：是否需要追问用户
        """
        return classify_customer_query(self.model, user_query)

    def handle_request(self, user_query: str, thread_id: str = "demo-thread") -> dict[str, Any]:
        """处理一轮用户请求。

        Args:
            user_query: 用户的原始问题
            thread_id: 会话线程 ID，用于上下文隔离和日志追踪

        Returns:
            包含以下键的字典：
            - route_plan：路由计划（意图、子任务列表等）
            - supervisor_result：supervisor 的原始执行结果
            - final_answer：最终返回给用户的可读文本
            - business_completion_hint：业务补充提示（是否需追问、缺失字段）
            - execution_mode：执行模式（"agent_supervisor" 或 "deterministic_fallback"）
            - debug_input：调试用的原始输入
        """
        # 第一步：路由分析，生成执行计划
        print(f"\n{'='*60}")
        print(f"[ROUTER] 开始分析用户问题: {user_query}")
        route_plan = self.route(user_query)
        print(f"[ROUTER] 路由结果: intents={route_plan.intents}")
        print(f"[ROUTER] 生成 sub_tasks 数量: {len(route_plan.sub_tasks)}")
        for i, t in enumerate(route_plan.sub_tasks):
            print(f"  sub_task[{i}]: task_id={t.task_id}, intent={t.intent}, slots={t.slots}, missing_slots={t.missing_slots}")
        if route_plan.need_clarification:
            print(f"[ROUTER] 需要追问: {route_plan.clarification_question}")

        serialized_tasks = [
            task.model_dump_json(ensure_ascii=False)
            for task in route_plan.sub_tasks
        ]

        # 组装 supervisor 输入
        supervisor_input = {
            "user_query": user_query,
            "route_plan": route_plan.model_dump(mode="json"),
            "serialized_tasks": serialized_tasks,
        }

        # 构造给 supervisor 的控制指令（包含 user_query + route_plan + 操作指引）
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

        # 优先使用 supervisor agent 模式
        try:
            print(f"\n[SUPERVISOR] supervisor agent 开始调度，thread_id={thread_id}")
            result = self.supervisor_agent.invoke(
                {"messages": [HumanMessage(content=control_message)]},
                context=SupportContext(thread_id=thread_id),
            )
            print(f"[SUPERVISOR] supervisor agent 调用完成")
            print(f"[SUPERVISOR] 返回的 messages 数量: {len(result.get('messages', []))}")
            final_answer = getattr(result["messages"][-1], "content", "")
            supervisor_mode = "agent_supervisor"
            print(f"[SUPERVISOR] 最终回复(前100字): {final_answer[:100]}")
        except Exception as e:
            print(f"\n[SUPERVISOR] supervisor agent 调用异常: {e}")
            print(f"[FALLBACK] 降级为确定性分发模式")
            # supervisor 调用失败时，降级为确定性分发：直接遍历 sub_tasks 串行调用 specialist
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
            print(f"[FALLBACK] synthesize_support_response 汇总完成")
            print(f"[FALLBACK] 各任务结果:")
            for tr in task_results:
                print(f"  └─ [{tr.intent}|{tr.status}] {tr.answer[:60]}{'...' if len(tr.answer)>60 else ''}")
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
    # 演示模式：遍历多个典型场景并打印结果
    app = LangChainCustomerSupportApp()
    scenarios = [
        # "帮我查一下订单 A1001 的物流到哪了",
        # "无线耳机 Pro 有货吗，续航怎么样？",
        # "订单 A1003 我想退款，原因是买错了",
        "帮我查订单 A1001 的物流，另外我想把 A1003 退掉，原因是买错了",
    ]

    for index, query in enumerate(scenarios, start=1):
        print("\n" + "=" * 100)
        print(f"[LangChain Demo {index}] {query}")
        output = app.handle_request(query, thread_id=f"lc-demo-{index}")
        print(json.dumps(output["route_plan"], ensure_ascii=False, indent=2))
        print("-" * 100)
        print(output["final_answer"])
