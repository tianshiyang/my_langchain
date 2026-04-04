# 企业生产环境最佳实践

## 1. 存储选型对比

### Checkpointer 存储

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **InMemorySaver** | 无需部署、低延迟 | 数据丢失、无法扩展 | 开发测试 |
| **PostgresSaver** | 支持分布式、数据持久化 | 需要维护数据库 | 生产环境（单实例） |
| **RedisSaver** | 高性能、支持分布式 | 需要额外基础设施 | 高并发生产环境 |
| **MySQLSaver** | 兼容性好、运维成熟 | 性能略低 | 已有 MySQL 基础设施 |

### Store 存储

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **PostgresStore** | 支持复杂查询、事务 | 全文搜索能力弱 | 结构化数据、用户偏好 |
| **MilvusStore** | 向量搜索强大 | 需要维护 | 语义记忆、相似度检索 |
| **RedisStore** | 超高性能 | 内存有限 | 临时缓存、会话状态 |
| **ElasticsearchStore** | 强大搜索能力 | 资源消耗大 | 日志、审计 |

## 2. Token 限制管理策略

### 分层管理策略

```
┌─────────────────────────────────────────────────────┐
│              Tier 1: 实时检查 (每次请求)            │
│         - 计算当前 token 数量                        │
│         - 未超过 50% 阈值：直接使用                 │
│         - 超过 50% 阈值：触发 Tier 2               │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            Tier 2: 消息修剪 (@before_model)         │
│         - 保留系统消息 + 最近 N 条                  │
│         - 推荐保留 3-5 条消息                        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Tier 3: 摘要压缩 (Summarization)          │
│         - 触发阈值：4000-6000 tokens                │
│         - 保留最近 15-20 条消息                     │
│         - 旧消息压缩为单条摘要                      │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Tier 4: 会话拆分 (最后手段)            │
│         - 创建新 thread_id                         │
│         - 传递关键上下文到新会话                    │
└─────────────────────────────────────────────────────┘
```

### 配置推荐

```python
# 推荐的中间件配置
agent = create_agent(
    model,
    tools=[...],
    middleware=[
        # Tier 2: 消息修剪
        trim_messages,  # 保留最近 3-5 条

        # Tier 3: 摘要压缩
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),  # 4k tokens 触发
            keep=20                      # 保留 20 条
        ),
    ],
    checkpointer=PostgresSaver.from_conn_string(DB_URI)
)
```

## 3. 隐私合规（PII 处理）

### PIIMiddleware

`PIIMiddleware` 可以自动检测和处理用户输入中的个人身份信息。

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model,
    tools=[...],
    middleware=[
        PIIMiddleware(
            "email",                    # 检测邮箱
            strategy="mask",             # 处理策略：mask / redact / block
            apply_to_output=True,       # 同时处理输出
        ),
        PIIMiddleware(
            "phone_number",
            strategy="mask",
            apply_to_output=False,
        ),
    ]
)

# 当用户输入包含 PII 时，会自动处理
result = agent.invoke({
    "messages": [{"role": "user", "content": "张三：15999999999@qq.com，我的邮箱是什么"}]
})
# 实际发送给模型的是：***：*********@qq.com，我的邮箱是什么
```

### 处理策略

| 策略 | 效果 | 适用场景 |
|------|------|---------|
| `mask` | 部分隐藏，如 `***@qq.com` | 保留可读性 |
| `redact` | 完全移除，如 `[REDACTED]` | 高敏感数据 |
| `block` | 拒绝处理 | 强制合规 |

### 自定义 PII 处理

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def sanitize_input(request: ModelRequest, handler):
    """自定义 PII 处理中间件"""
    content = request.messages[0].content

    # 邮箱处理
    content = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', content)

    # 手机号处理
    content = re.sub(r'1[3-9]\d{9}', '[PHONE]', content)

    # 身份证处理
    content = re.sub(r'\d{17}[\dXx]', '[ID]', content)

    request = request.override(messages=[
        request.messages[0].__class__(content=content)
    ])

    return handler(request)
```

## 4. 性能优化

### 连接池配置

```python
# PostgreSQL 连接池配置
with PostgresSaver.from_conn_string(
    "postgresql://user:pass@host:5432/db?"
    "max_connections=20&"        # 最大连接数
    "min_connections=5&"         # 最小连接数
    "connection_timeout=10&"      # 超时时间
) as checkpointer:
    ...
```

### 异步优化

```python
# 对于高并发场景，考虑异步实现
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

async def async_invoke(agent, input_data, thread_id):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: agent.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
    )
    return result
```

### 批量处理

```python
# 对于多用户场景，批量处理减少数据库往返
async def batch_get_context(user_ids: list[str], store: PostgresStore):
    """批量获取用户上下文"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            loop.run_in_executor(executor, store.get, ("context",), uid)
            for uid in user_ids
        ]
        return await asyncio.gather(*futures)
```

## 5. 高可用架构

### 分布式 Checkpointer

```
                    ┌─────────────────┐
                    │   Load Balancer  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ Agent 1  │      │ Agent 2  │      │ Agent 3  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────┴──────┐
                    │  PostgreSQL │
                    │  (主从复制)  │
                    └─────────────┘
```

### 多级缓存

```
┌─────────────────────────────────────────────────────┐
│                  L1: Redis Cache                    │
│            (热点数据，毫秒级延迟)                     │
│               TTL: 5-10 分钟                        │
└─────────────────────────────────────────────────────┘
                        │ miss
                        ▼
┌─────────────────────────────────────────────────────┐
│              L2: PostgreSQL                         │
│            (主存储，秒级延迟)                        │
└─────────────────────────────────────────────────────┘
                        │ miss
                        ▼
┌─────────────────────────────────────────────────────┐
│              L3: Vector Store (Milvus)              │
│            (知识库，语义搜索)                        │
└─────────────────────────────────────────────────────┘
```

## 6. 运维建议

### 监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|---------|
| `memory.thread_count` | 活跃会话数 | > 10000 |
| `memory.avg_tokens` | 平均 token 消耗 | > 3000 |
| `memory.checkpointer.latency` | 状态读写延迟 | > 100ms |
| `memory.store.latency` | Store 读写延迟 | > 200ms |

### 容量规划

```python
# 估算存储需求
sessions_per_day = 100000          # 每日会话数
avg_thread_size_kb = 50           # 平均线程大小(KB)
retention_days = 30               # 保留天数

storage_per_day = sessions_per_day * avg_thread_size_kb
total_storage = storage_per_day * retention_days

print(f"每日存储需求: {storage_per_day / 1024:.2f} MB")
print(f"30天总存储: {total_storage / 1024 / 1024:.2f} GB")
```

### 备份策略

```sql
-- PostgreSQL 检查点表备份
BACKUP TABLE checkpointer.messages
TO '/backup/checkpointer_messages_$(date +%Y%m%d).sql';

-- 增量备份（每天）
-- 使用 pg_dump --incremental 或 WAL 日志归档
```

## 7. 安全考虑

### 数据加密

```python
# 敏感数据加密存储
from cryptography.fernet import Fernet

class EncryptedStore:
    def __init__(self, store: PostgresStore, key: bytes):
        self.store = store
        self.cipher = Fernet(key)

    def put(self, namespace, key, value):
        # 加密敏感字段
        if isinstance(value, dict):
            value = {k: self.cipher.encrypt(str(v).encode()).decode()
                    if self._is_sensitive(k) else v
                    for k, v in value.items()}
        return self.store.put(namespace, key, value)

    def _is_sensitive(self, field_name: str) -> bool:
        sensitive_fields = {'password', 'token', 'secret', 'key'}
        return field_name.lower() in sensitive_fields
```

### 访问控制

```python
# 基于用户 ID 的访问隔离
def secure_store_get(store, namespace, key, user_id: str):
    # 确保用户只能访问自己的数据
    if not key.startswith(user_id):
        raise PermissionError("Access denied")
    return store.get(namespace, key)
```

## 后续文档

- [客服场景记忆方案](./memory_in_customer_support.md)
