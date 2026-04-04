# LangChain 记忆功能架构概述

## 什么是记忆？

在 LangChain 的 Agent 系统中，**记忆（Memory）** 是让 Agent 能够在多轮对话中保持上下文连续性的机制。没有记忆，Agent 每次交互都是独立的，无法理解对话历史。

## 短期记忆 vs 长期记忆

| 类型 | 作用范围 | 存储介质 | 典型用途 |
|------|---------|---------|---------|
| **短期记忆** | 单次对话会话内 | 内存、Checkpointer | 保持对话上下文、任务状态 |
| **长期记忆** | 跨会话持久化 | 数据库、向量存储 | 用户偏好、实体信息、领域知识 |

## LangChain Memory API 核心概念

### 1. Checkpointer（检查点器）

Checkpointer 用于在对话过程中持久化 Agent 的**状态**，使得：
- 对话可以在中断后恢复
- 同一个 thread_id 下的对话保持连续
- Agent 可以追踪多轮交互中的中间状态

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# 内存型检查点（适合开发/测试）
checkpointer = InMemorySaver()

# PostgreSQL 检查点（适合生产环境）
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/mydb"
)
```

### 2. ChatMessageHistory（聊天消息历史）

管理对话中的消息序列，包括 HumanMessage、AI 消息、SystemMessage 等。

### 3. Store（存储）

用于**跨会话**的持久化存储，可以存储：
- 用户偏好设置
- 实体信息
- 知识图谱

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(
    "postgresql://user:pass@localhost:5432/mydb"
)
```

### 4. Middleware（中间件）

中间件在模型调用前后执行，用于修改消息、注入上下文：

- **SummarizationMiddleware**: 当 token 接近限制时，压缩历史消息
- **PIIMiddleware**: 检测和处理 PII（个人身份信息）
- **dynamic_prompt**: 动态注入系统提示

## 记忆组件在 Agent 执行流程中的位置

```
用户输入
    ↓
┌─────────────────────────────────────────┐
│            Middleware 层                 │
│  (PII 过滤 → 消息修剪 → 动态提示注入)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│         Checkpointer (短期记忆)          │
│     (保存当前会话状态、消息历史)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│            Agent / LLM                  │
│         (理解、推理、决策)               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│           Store (长期记忆)              │
│   (用户偏好、实体信息、跨会话知识)        │
└─────────────────────────────────────────┘
    ↓
  响应输出
```

## 关键 API 总结

| 组件 | 用途 | 持久化 | 典型场景 |
|------|------|--------|---------|
| `InMemorySaver` | 单机内存存储 | 会话级 | 开发测试 |
| `PostgresSaver` | 分布式状态存储 | 会话级 | 生产环境 |
| `PostgresStore` | 跨会话语义存储 | 永久 | 用户偏好 |
| `SummarizationMiddleware` | 消息摘要压缩 | - | 长对话 |
| `PIIMiddleware` | 隐私数据处理 | - | 合规需求 |

## 后续文档

- [短期记忆实现方案](./short_term_memory.md)
- [长期记忆实现方案](./long_term_memory.md)
- [企业生产环境最佳实践](./enterprise_best_practices.md)
- [客服场景记忆方案](./memory_in_customer_support.md)
