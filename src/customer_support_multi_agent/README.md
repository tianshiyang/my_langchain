# 客服多智能体示例

这个目录包含两套实现：

- `langchain_customer_support.py`
  使用 LangChain 1.x 的高层 agent 模式，组合了：
  `router + supervisor + specialists + refund handoff`
- `langgraph_customer_support.py`
  使用 LangGraph 1.x 的显式工作流，组合了：
  `classify -> plan_tasks -> parallel specialists -> refund subflow -> synthesize`

另外新增了一组“进阶教学版”：

- `langchain_agent_loop_multi_agent.py`
  使用 LangChain 1.x 演示：
  `router -> supervisor agent loop -> specialists -> reviewer -> notification`
- `langgraph_agent_loop_multi_agent.py`
  使用 LangGraph 1.x 演示：
  `classify -> plan -> dispatch -> refund_subgraph/human_gate -> synthesize -> review -> notify`

## 目录说明

- `models.py`
  共享契约：
  `CustomerIntentResult`、`SubTask`、`TaskResult`、`RefundState`
- `support_shared.py`
  共享 mock 业务数据、工具、结构化路由、specialist agent 工厂
- `advanced_shared.py`
  reviewer、notification、退款审批 HITL、draft/review 共用逻辑

## 运行方式

在项目根目录执行：

```bash
python -m src.customer_support_multi_agent.langchain_customer_support
python -m src.customer_support_multi_agent.langgraph_customer_support
python -m src.customer_support_multi_agent.langchain_agent_loop_multi_agent
python -m src.customer_support_multi_agent.langgraph_agent_loop_multi_agent
```

## 关键设计点

- 前门统一契约：`CustomerIntentResult`
- specialist 统一输出：`TaskResult`
- 退款流程在 LangChain 版里用 handoff 风格 stateful agent
- 退款流程在 LangGraph 版里用显式 subflow
- reviewer 统一输出：`ReviewResult`
- 进阶版会显式记录 `execution_trace`，方便解释 loop 为什么停止
- 结束条件不是“模型觉得完了”，而是：
  `done | blocked_waiting_user | escalated_to_human` 的任务都已生成用户可见回复
