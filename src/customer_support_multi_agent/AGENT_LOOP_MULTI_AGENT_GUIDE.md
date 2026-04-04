# Agent Loop + Multi-Agent 对照说明

这组进阶 demo 统一使用一个常见售后场景：

`帮我查订单 A1001 的物流；如果订单 A1004 可以退款就帮我申请退款，原因是买错了；最后把处理结果发给我确认。`

对应实现文件：

- `langchain_agent_loop_multi_agent.py`
- `langgraph_agent_loop_multi_agent.py`
- `advanced_shared.py`

## 1. 这次 demo 想讲清楚什么

这套示例不只是“多调用几个工具”，而是想把 4 个容易混在一起的概念拆开：

1. `agent loop` 到底是什么
2. `multi-agent` 里主 agent 和子 agent 怎么通信
3. loop 为什么会停，不是靠模型一句“我做完了”
4. reviewer 和 human-in-the-loop 各自负责什么

这里的统一约定是：

- 子 agent 不靠自然语言告诉主 agent 自己“完成了”
- 子 agent 必须返回结构化 `TaskResult`
- reviewer 必须返回结构化 `ReviewResult`
- 主流程只根据状态和结构化输出来决定停不停

## 2. 子 agent 怎么通知主 agent

所有 specialist 都统一返回 `TaskResult`：

- `status`
  只能是 `done | blocked_waiting_user | escalated_to_human`
- `answer`
  当前阶段要给用户看的结果
- `follow_up_questions`
  如果还缺信息，需要追问什么
- `human_required`
  是否已经进入人工审核
- `raw_data`
  给调试和后续逻辑看的结构化数据

这意味着主 agent 不需要猜：

- 如果是 `done`，这个子任务结束
- 如果是 `blocked_waiting_user`，这个子任务停止自动推进，转成向用户追问
- 如果是 `escalated_to_human`，这个子任务停止自动推进，明确告知已转人工

所以“子 agent 怎么通知主 agent”这件事，本质上靠的是共享契约，而不是靠 prompt 约定。

## 3. LangChain 版的实现原理

### 主结构

LangChain 版是：

`router -> supervisor agent -> specialist tools -> reviewer -> notification`

其中 supervisor 仍然是高层 `create_agent`，但外面再包了一层 Python 守护循环。

这样做的原因是：

- 内层保留 LangChain 高层 agent 的使用方式
- 外层可以显式控制 pending tasks、review 次数和 fallback

### 它的 loop 藏在哪里

LangChain 版其实有两层 loop：

1. `create_agent` 内部的 tool-calling loop
2. `handle_request()` 外部的 Python loop

外层 loop 每一轮都会：

1. 把 `pending_task_ids` 告诉 supervisor
2. 让 supervisor 选择要调用哪个 specialist tool
3. specialist 返回 `TaskResult`
4. 把已完成 task 从 pending 里移除

如果本轮没有任何进展，就触发 deterministic fallback，避免 supervisor 卡死。

### LangChain 版怎么判断 loop 可以停

LangChain 版不是靠 supervisor 文本说“处理完了”来停，而是同时满足：

1. 所有任务都已经返回 `TaskResult`
2. 所有任务状态都属于终态
3. reviewer 返回 `approved`

如果 reviewer 打回：

- 重新生成 `draft_response`
- 带着 reviewer 反馈再 review 一次

如果 review 超过上限：

- 直接返回最后一版草稿
- 追加 reviewer 反馈，避免无限循环

## 4. LangGraph 版的实现原理

### 主结构

LangGraph 版是显式状态图：

`classify -> plan -> dispatch -> specialists/refund_subgraph -> synthesize -> review -> notify`

和 LangChain 最大的区别是：

- 任务怎么流转是写在图上的
- 停止条件是节点间路由，不是 prompt 里的口头约束

### 退款为什么单独做成 subgraph

退款天然就是一个独立小流程：

1. 收集订单号
2. 收集退款原因
3. 检查退款资格
4. 自动退款 / 转人工审核 / 人工批准后再放行

所以最适合单独做成 `refund_subgraph`。

这个子图最终仍然只向主图输出一件事：

- `task_results: [TaskResult(...)]`

所以主图仍然不需要知道退款子图内部用了几步。

### LangGraph 版怎么判断 loop 可以停

LangGraph 版把这些条件都放在 state 里：

- `pending_task_ids`
- `task_results`
- `review_decision`
- `review_round`
- `supervisor_round`

停下来的条件是：

1. `pending_task_ids` 已清空
2. `task_results` 全部处于终态
3. reviewer 返回 `approved`

如果 reviewer 返回 `revise`：

- 图会从 `review` 回到 `synthesize`

如果 review 达到上限：

- 图仍然结束，但会把 reviewer 反馈附加到最终输出

这就是 LangGraph 相比 LangChain 更直观的地方：
“什么时候停”不是藏在 prompt 里，而是图上的条件分支。

## 5. reviewer agent 在干什么

reviewer 不是业务执行者，也不是人工审批人。

它只做一件事：

- 检查 `draft_response` 是否覆盖了所有 `TaskResult`

它会重点检查：

- 有没有漏掉某个子任务结果
- 有没有遗漏待补信息
- 有没有漏写“已进入人工审核”

因此 reviewer 的位置在：

`所有 specialist 完成之后，通知用户之前`

它的价值是防止：

- 物流答了，但退款忘了说
- 已经转人工审核，却被草稿误写成“已成功退款”
- 缺少退款原因，但草稿没有继续追问用户

## 6. 子 agent 的人机交互怎么工作

### LangChain 版

LangChain 版在两个地方用了 `HumanInTheLoopMiddleware`：

- 高金额退款放行前
- 发送最终通知前

中断后会拿到 action request，再通过 `Command(resume=...)` 恢复。

demo 里为了方便一次跑通，提供了自动决策逻辑：

- 退款审批默认 `approve`
- 通知默认 `edit`，会自动给内容加上“人工已审阅”

### LangGraph 版

LangGraph 版把退款审批放在子图节点里，用 `interrupt(...)` 显式暂停。

这意味着：

- 图执行到审批节点就会停下
- 外部收到中断值
- 外部传入 `Command(resume=...)`
- 子图从同一个审批节点继续

这个过程比 middleware 更“裸露”，也更容易看清楚图执行到底停在了哪里。

## 7. 两套实现最核心的区别

可以把它们理解成：

- LangChain：高层 agent 比较省代码，但 loop 和停止规则更多依赖你在外围加护栏
- LangGraph：图更啰嗦，但状态流转、停止条件、人工中断位置都更清楚

如果你要教学和解释原理，LangGraph 更直观。
如果你要快速搭一个可工作的 agent 组合，LangChain 更轻便。

## 8. 你运行时该重点观察什么

建议先跑两个进阶 demo：

```bash
python -m src.customer_support_multi_agent.langchain_agent_loop_multi_agent
python -m src.customer_support_multi_agent.langgraph_agent_loop_multi_agent
```

重点观察：

1. `execution_trace`
   看每一步到底是谁调用了谁
2. `task_results`
   看子 agent 如何用统一结构通知主流程
3. `review_result`
   看 reviewer 为什么会通过或打回
4. `__interrupt__`
   看 LangGraph 退款审批是怎么暂停和恢复的
