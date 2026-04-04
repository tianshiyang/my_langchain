# 客服场景记忆方案

基于 `customer_support_multi_agent` 的实际实现，分析客服场景中的记忆方案。

## 客服场景的特点

1. **多轮交互**：用户可能在一次会话中多次变更需求
2. **任务型对话**：需要完成特定任务（查物流、退款等）
3. **状态流转**：复杂任务有多个阶段，需要状态管理
4. **跨任务记忆**：一次对话可能涉及多个不同任务

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                   LangChainCustomerSupportApp                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │   Router    │───▶│  Supervisor  │───▶│   Specialists    │    │
│  │  (意图分类)  │    │   Agent      │    │ (物流/商品/退款)  │    │
│  └─────────────┘    └──────────────┘    └──────────────────┘    │
│         │                   │                    │               │
│         └───────────────────┴────────────────────┘               │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │  SupportContext │                          │
│                    │   (thread_id)   │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Checkpointer (InMemorySaver)                 │
│                                                                  │
│  thread_id="lc-demo-1" ──────────────────────────────▶ 对话历史  │
│  thread_id="lc-demo-1:task_xxx" ───────────────────▶ 退款任务状态 │
└─────────────────────────────────────────────────────────────────┘
```

## 短期记忆实现

### 1. SupportContext（会话上下文）

```python
# models.py
class SupportContext(BaseModel):
    """LangChain supervisor 的上下文"""
    thread_id: str = "default-thread"
```

**作用**：通过 `thread_id` 隔离不同的对话会话。

```python
# langchain_customer_support.py
result = self.supervisor_agent.invoke(
    {"messages": [HumanMessage(content=control_message)]},
    context=SupportContext(thread_id=thread_id),  # 会话隔离
)
```

### 2. RefundAgentState（任务状态）

退款任务使用专门的状态模式，支持多步骤流程：

```python
# support_shared.py
class RefundAgentState(AgentState):
    """退款 agent 的状态"""
    current_step: NotRequired[Literal[
        "collect_order",      # 收集订单号
        "collect_reason",     # 收集退款原因
        "resolve_refund",     # 决策退款
        "completed"           # 完成
    ]]
    order_id: NotRequired[str]
    reason: NotRequired[str]
    requires_human_review: NotRequired[bool]
    refund_status: NotRequired[Literal[
        "pending", "submitted", "human_review", "declined"
    ]]
```

### 3. 多级 Thread ID 隔离

```python
# langchain_customer_support.py

# 1. 对话级别 thread_id
result = self.supervisor_agent.invoke(
    {"messages": [HumanMessage(content=control_message)]},
    context=SupportContext(thread_id=thread_id),  # e.g., "lc-demo-1"
)

# 2. 退款任务级别 thread_id（隔离不同退款请求）
refund_thread_id = f"{thread_id}:{task.task_id}"  # e.g., "lc-demo-1:task_b491c218"
result = run_refund_specialist(app.refund_agent, task, refund_thread_id)
```

**为什么需要两级隔离？**
- 一个用户可能同时有多个退款请求在处理
- 每个退款任务需要独立的状态流转
- 最终结果需要关联回原始会话

## 动态阶段切换（Middleware）

退款 agent 使用 middleware 实现动态阶段切换：

```python
# support_shared.py
@wrap_model_call
def apply_refund_stage(request: ModelRequest, handler):
    """根据退款阶段动态切换 prompt 和工具"""
    current_step = request.state.get("current_step", "collect_order")
    stage_config = REFUND_STAGE_CONFIG[current_step]

    # 动态更新系统提示和可用工具
    request = request.override(
        system_prompt=stage_config["prompt"].format(**request.state),
        tools=stage_config["tools"],
    )
    return handler(request)
```

**阶段配置**：

```python
REFUND_STAGE_CONFIG = {
    "collect_order": {
        "prompt": "你是退款客服专员。当前阶段：收集订单号...",
        "tools": [record_refund_order_id],
    },
    "collect_reason": {
        "prompt": "你是退款客服专员。当前阶段：收集退款原因...",
        "tools": [record_refund_reason],
    },
    "resolve_refund": {
        "prompt": "你是退款客服专员。当前阶段：退款决策...",
        "tools": [check_refund_eligibility, submit_refund_request, ...],
    },
    "completed": {...}
}
```

## 消息修剪与摘要

### 消息修剪示例

```python
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """修剪消息，保留系统消息和最近交互"""
    messages = state.get("messages")
    if len(messages) <= 3:
        return None

    # 保留第一条消息（通常是系统提示）和最近的消息
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            messages[0],  # 系统消息
            *messages[-3:]  # 最近 3 条
        ]
    }
```

## 完整代码：创建带记忆的客服 Agent

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, NotRequired
from typing_extensions import Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

# ============================================
# 1. 定义状态
# ============================================
class CustomerSupportState(AgentState):
    """客服状态"""
    current_step: NotRequired[str]
    customer_id: NotRequired[str]
    intent: NotRequired[str]
    status: NotRequired[Literal["pending", "resolved", "escalated"]]

class SupportContext(BaseModel):
    """会话上下文"""
    thread_id: str
    customer_id: str = ""

# ============================================
# 2. 定义工具
# ============================================
@tool
def lookup_order(order_id: str) -> str:
    """查询订单"""
    return f"订单 {order_id} 信息..."

@tool
def create_refund(order_id: str, reason: str) -> str:
    """创建退款"""
    return f"退款已创建: {order_id}, 原因: {reason}"

# ============================================
# 3. 定义阶段中间件
# ============================================
TICKET_STAGES = {
    "intake": {
        "prompt": "收集客户信息和问题",
        "tools": [lookup_order],
    },
    "processing": {
        "prompt": "处理客户问题",
        "tools": [create_refund],
    },
    "resolved": {
        "prompt": "确认解决并结束",
        "tools": [],
    }
}

@wrap_model_call
def apply_stage(request: ModelRequest, handler):
    step = request.state.get("current_step", "intake")
    config = TICKET_STAGES[step]

    request = request.override(
        system_prompt=config["prompt"],
        tools=config["tools"],
    )
    return handler(request)

# ============================================
# 4. 创建 Agent
# ============================================
def create_support_agent(model):
    return create_agent(
        model,
        tools=[lookup_order, create_refund],
        state_schema=CustomerSupportState,
        middleware=[apply_stage],
        checkpointer=InMemorySaver(),  # 短期记忆
    )

# ============================================
# 5. 使用
# ============================================
agent = create_support_agent(model)

# 第一个问题
result1 = agent.invoke(
    {"messages": [HumanMessage("我要退款订单 A1001")]},
    context=SupportContext(thread_id="session_001", customer_id="cust_123")
)
print(result1["messages"][-1].content)

# 继续对话（自动记住之前的状态）
result2 = agent.invoke(
    {"messages": [HumanMessage("原因是买错了")]},
    context=SupportContext(thread_id="session_001", customer_id="cust_123")
)
print(result2["messages"][-1].content)
```

## 长期记忆在客服场景的应用

### 用户偏好存储

```python
from langgraph.store.postgres import PostgresStore

# 存储用户偏好
store.put(
    ("customer_preference",),
    customer_id,
    {
        "preferred_contact": "email",
        "language": "zh-CN",
        "vip_level": "gold",
        "previous_complaints": 2,
    }
)

# 在工具调用时自动注入
@tool
def get_customer_history(runtime: ToolRuntime):
    customer_id = runtime.context.customer_id
    pref = store.get(("customer_preference",), customer_id)

    if pref and pref.value.get("vip_level") == "gold":
        return "VIP 客户，优先处理"
    return "普通客户"
```

## 架构对比

### 简单方案 vs 完整方案

| 特性 | 简单方案 | 完整方案 |
|------|---------|---------|
| 对话历史 | InMemorySaver | PostgresSaver |
| 状态管理 | 消息堆叠 | 显式状态机 |
| 跨会话记忆 | 无 | PostgresStore |
| 阶段切换 | 静态提示 | Middleware 动态切换 |
| Token 控制 | 无 | Trim + Summarization |
| PII 处理 | 无 | PIIMiddleware |

## 最佳实践总结

1. **Thread ID 设计**：使用层级化 ID（如 `session_id:task_id`）隔离不同粒度的对话
2. **状态机模式**：复杂任务使用显式状态机而非消息堆叠
3. **Middleware 解耦**：阶段切换逻辑放在 Middleware 中，保持工具函数纯净
4. **Checkpoint 持久化**：生产环境使用 PostgresSaver，支持故障恢复
5. **长期记忆分层**：用户偏好放 Store，任务状态放 Checkpointer
6. **Token 预算**：设置预警阈值，提前修剪或摘要

## 相关文件

- `src/customer_support_multi_agent/support_shared.py` - 退款 agent 完整实现
- `src/customer_support_multi_agent/langchain_customer_support.py` - 多 agent 调度
- `src/langchain/short_term_memory.py` - 短期记忆模式
- `src/langchain/context_engineering.py` - 长期记忆模式
