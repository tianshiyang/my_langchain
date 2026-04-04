import unittest

from langgraph.types import Command

from src.customer_support_multi_agent.advanced_shared import (
    build_support_draft,
    default_human_decision,
    review_support_draft,
)
from src.customer_support_multi_agent.langgraph_agent_loop_multi_agent import build_refund_subgraph
from src.customer_support_multi_agent.models import (
    ReviewResult,
    SubTask,
    TaskResult,
    all_results_terminal,
    should_stop_agent_loop,
)


class _ExplodingStructuredModel:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("structured output unavailable")


class ExplodingModel:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("llm unavailable")

    def with_structured_output(self, *_args, **_kwargs):
        return _ExplodingStructuredModel()


class CustomerSupportAdvancedTests(unittest.TestCase):
    def setUp(self):
        self.model = ExplodingModel()

    def test_terminal_helpers(self):
        task_results = [
            TaskResult(task_id="t1", intent="物流查询", status="done", answer="物流已返回"),
            TaskResult(
                task_id="t2",
                intent="退款",
                status="escalated_to_human",
                answer="已升级人工审核",
                human_required=True,
            ),
        ]
        review = ReviewResult(decision="approved", feedback="通过")

        self.assertTrue(all_results_terminal(task_results))
        self.assertTrue(
            should_stop_agent_loop(
                task_results,
                review_result=review,
                supervisor_round=1,
                review_round=1,
            )
        )

    def test_reviewer_requests_revision_when_answer_missing(self):
        task_results = [
            TaskResult(task_id="t1", intent="物流查询", status="done", answer="订单 A1001 运输中"),
            TaskResult(task_id="t2", intent="退款", status="done", answer="订单 A1004 已提交退款申请"),
        ]
        draft = build_support_draft(
            self.model,
            "帮我查物流并处理退款",
            task_results,
            force_incomplete=True,
        )
        review = review_support_draft(self.model, "帮我查物流并处理退款", task_results, draft)

        self.assertEqual(review.decision, "revise")
        self.assertIn("退款", review.feedback)

    def test_reviewer_approves_complete_draft(self):
        task_results = [
            TaskResult(task_id="t1", intent="物流查询", status="done", answer="订单 A1001 运输中"),
            TaskResult(
                task_id="t2",
                intent="退款",
                status="escalated_to_human",
                answer="订单 A1003 需要人工审核",
                human_required=True,
            ),
        ]
        draft = (
            "订单 A1001 运输中。"
            "\n订单 A1003 需要人工审核，售后团队会继续跟进。"
        )
        review = review_support_draft(self.model, "帮我查物流并处理退款", task_results, draft)

        self.assertEqual(review.decision, "approved")

    def test_default_human_decision_edits_notification(self):
        action_request = {
            "name": "send_customer_confirmation",
            "args": {"recipient": "站内信", "content": "原始内容"},
        }
        decision = default_human_decision("send_customer_confirmation", action_request)

        self.assertEqual(decision["type"], "edit")
        self.assertIn("人工已审阅", decision["args"]["content"])

    def test_refund_subgraph_interrupt_and_resume(self):
        graph = build_refund_subgraph()
        task = SubTask(
            task_id="task_refund_1",
            intent="退款",
            user_text="订单 A1004 我要退款，原因是买错了",
            slots={"order_id": "A1004", "refund_reason": "买错了"},
            missing_slots=[],
        )
        initial_state = {
            "task": task,
            "order_id": "",
            "reason": "",
            "eligibility": False,
            "audit_required": False,
            "manual_approval_required": False,
            "approval_decision": "",
            "completed_task_ids": [],
            "task_results": [],
            "execution_trace": [],
        }

        result = graph.invoke(initial_state, config={"configurable": {"thread_id": "refund-approve"}})
        self.assertIn("__interrupt__", result)

        resumed = graph.invoke(
            Command(resume={"decision": "approve"}),
            config={"configurable": {"thread_id": "refund-approve"}},
        )
        self.assertEqual(resumed["task_results"][0].status, "done")
        self.assertIn("退款单号", resumed["task_results"][0].answer)

    def test_refund_subgraph_reject_routes_to_human(self):
        graph = build_refund_subgraph()
        task = SubTask(
            task_id="task_refund_2",
            intent="退款",
            user_text="订单 A1004 我要退款，原因是买错了",
            slots={"order_id": "A1004", "refund_reason": "买错了"},
            missing_slots=[],
        )
        initial_state = {
            "task": task,
            "order_id": "",
            "reason": "",
            "eligibility": False,
            "audit_required": False,
            "manual_approval_required": False,
            "approval_decision": "",
            "completed_task_ids": [],
            "task_results": [],
            "execution_trace": [],
        }

        graph.invoke(initial_state, config={"configurable": {"thread_id": "refund-reject"}})
        resumed = graph.invoke(
            Command(resume={"decision": "reject"}),
            config={"configurable": {"thread_id": "refund-reject"}},
        )
        self.assertEqual(resumed["task_results"][0].status, "escalated_to_human")


if __name__ == "__main__":
    unittest.main()
