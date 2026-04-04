# 短期记忆实现方案

短期记忆用于在**单次对话会话内**保持上下文，让 Agent 能够理解多轮对话的历史内容。

## 核心组件

### 1. Checkpointer（检查点器）

Checkpointer 是短期记忆的核心，负责持久化对话状态。

#### InMemorySaver（内存型）

适合开发测试，数据存储在内存中，会话结束即丢失。

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[...],
    checkpointer=checkpointer
)

# 同一 thread_id 下的对话会保持连续
result = agent.invoke(
    {"messages": [HumanMessage("你好")]},
    config={"configurable": {"thread_id": "user_123_session_1"}}
)
```

#### PostgresSaver（PostgreSQL 型）

适合生产环境，支持分布式部署和数据持久化。

```python
from langgraph.checkpoint.postgres import PostgresSaver

db_uri = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"

with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    # checkpointer.setup()  # 首次运行需要初始化表结构

    agent = create_agent(
        model,
        tools=[...],
        checkpointer=checkpointer
    )

    result = agent.invoke(
        {"messages": [HumanMessage("你好")]},
        config={"configurable": {"thread_id": "user_123_session_1"}}
    )
```

### 2. Message Trimming（消息修剪）

当对话历史过长时，修剪旧消息以控制 token 使用量。

```python
from langchain.agents.middleware import before_model, AgentState
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """
    裁剪用户消息，保留系统消息和最近的消息
    """
    messages = state.get("messages")
    if len(messages) <= 3:
        return None  # 不需要修剪

    first_msg = messages[0]  # 通常是系统消息
    # 保留最近的消息
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=[...],
    middleware=[trim_messages]
)
```

### 3. SummarizationMiddleware（摘要中间件）

当 token 数量达到阈值时，自动将旧消息压缩为摘要。

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    chatGptLLM,
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model=chatGptLLM,        # 用于生成摘要的模型
            trigger=("tokens", 4000), # 触发阈值（4000 tokens）
            keep=20                   # 保留最近 20 条消息
        )
    ],
    checkpointer=checkpointer
)
```

**工作原理：**
1. 每次模型调用前检查 token 总数
2. 超过阈值时，将旧消息压缩为一条摘要消息
3. 保留最近的 N 条消息不变

### 4. Thread Isolation（线程隔离）

通过 `thread_id` 隔离不同的对话会话。

```python
# 对话 A
result_a = agent.invoke(
    {"messages": [HumanMessage("我叫张三")]},
    config={"configurable": {"thread_id": "session_A"}}
)

# 对话 B（与 A 完全独立）
result_b = agent.invoke(
    {"messages": [HumanMessage("我叫李四")]},
    config={"configurable": {"thread_id": "session_B"}}
)

# 继续对话 A
result_a2 = agent.invoke(
    {"messages": [HumanMessage("我叫什么名字？")]},
    config={"configurable": {"thread_id": "session_A"}}  # 会记住"张三"
)
```

## 完整代码示例

基于 `src/langchain/short_term_memory.py`：

```python
#!/user/bin/env python
# -*- coding: utf-8 -*-
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware, dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from provider import chatGptLLM

# ============================================
# 1. 定义自定义状态和上下文
# ============================================
class CustomAgentState(AgentState):
    user_id: str
    user_name: str
    preference: dict

class CustomContext(BaseModel):
    user_id: str
    user_name: str

# ============================================
# 2. 定义工具
# ============================================
@tool
def get_user_info(runtime: ToolRuntime[CustomAgentState]) -> str:
    """获取用户信息"""
    cur_user_id = runtime.state.get("user_id")
    cur_user_info = user_info[cur_user_id]
    return f"用户名: {cur_user_info['username']}, 邮箱：{cur_user_info['email']}"

DB_URI = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    # ============================================
    # 3. 消息修剪中间件
    # ============================================
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """修剪消息，控制历史长度"""
        messages = state.get("messages")
        if len(messages) <= 3:
            return None

        first_msg = messages[0]
        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    # ============================================
    # 4. 动态系统提示
    # ============================================
    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        user_name = request.runtime.context["user_name"]
        return f"You are a helpful assistant. Address the user as {user_name}."

    # ============================================
    # 5. 创建 Agent（集成所有短期记忆组件）
    # ============================================
    agent = create_agent(
        chatGptLLM,
        tools=[get_user_info],
        middleware=[
            trim_messages,                    # 消息修剪
            SummarizationMiddleware(          # 摘要压缩
                model=chatGptLLM,
                trigger=("tokens", 4000),
                keep=20
            ),
            dynamic_system_prompt              # 动态提示
        ],
        state_schema=CustomAgentState,
        context_schema=CustomContext,
        checkpointer=checkpointer             # 状态持久化
    )

    # ============================================
    # 6. 发起多轮对话
    # ============================================
    result = agent.invoke(
        {
            "messages": [HumanMessage("我的邮箱是什么呢？")],
            "user_id": "user_123",
            "preferences": {"theme": "dark"}
        },
        config=RunnableConfig(
            configurable={"thread_id": "1"}
        ),
        context=CustomContext(user_name="John Smith")
    )

    print(result['messages'][-1].content)
```

## 架构图

```
┌─────────────────────────────────────────────────────┐
│                  用户多轮对话                        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            @before_model trim_messages              │
│         (修剪旧消息，保留最近 N 条)                  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│         SummarizationMiddleware                     │
│    (当 tokens > 4000 时，压缩旧消息为摘要)          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              PostgresSaver (Checkpointer)           │
│           (持久化状态到 PostgreSQL)                 │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                    LLM / Agent                      │
└─────────────────────────────────────────────────────┘
```

## 关键配置参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `thread_id` | 会话唯一标识 | 用户 ID 或会话 ID |
| `trim_messages` 阈值 | 保留消息条数 | 3-5 条 |
| `SummarizationMiddleware` 触发值 | token 触发阈值 | 4000-6000 |
| `SummarizationMiddleware` keep | 保留消息条数 | 15-20 |

## 适用场景

- **客服对话**：多轮交互理解用户问题
- **复杂任务**：分步骤执行的任务状态保持
- **长对话处理**：自动管理 token 消耗

## 后续文档

- [长期记忆实现方案](./long_term_memory.md) - 跨会话持久化
- [企业生产环境最佳实践](./enterprise_best_practices.md)
