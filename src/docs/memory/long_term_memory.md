# 长期记忆实现方案

长期记忆用于在**跨会话**之间持久化信息，包括用户偏好、实体信息、领域知识等。

## 核心组件：Store

LangGraph 的 `Store` 接口提供了跨会话的语义存储能力。

### 1. PostgresStore（PostgreSQL 存储）

适合生产环境，支持复杂的查询和事务。

```python
from langgraph.store.postgres import PostgresStore

db_uri = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"

with PostgresStore.from_conn_string(db_uri) as store:
    # store.setup()  # 首次运行需要初始化表结构

    # 存储数据
    store.put(
        namespace=("user_preference",),  # 命名空间
        key="user_123",                  # 用户 ID
        value={
            "tone": "friendly",
            "greeting": "你好",
            "sign_off": "祝好"
        }
    )

    # 查询数据
    result = store.get(namespace=("user_preference",), key="user_123")
    print(result.value)  # {'tone': 'friendly', 'greeting': '你好', 'sign_off': '祝好'}
```

### 2. 命名空间（Namespace）

命名空间用于组织不同类型的长期记忆：

```python
# 用户偏好
store.put(("writing_style",), user_id, {...})

# 实体信息
store.put(("entity",), company_name, {...})

# 知识库
store.put(("knowledge",), topic, {...})
```

## Dynamic Prompt Injection（动态提示注入）

通过 `dynamic_prompt` 中间件，在每次请求时动态注入用户相关的长期记忆。

```python
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call

@dynamic_prompt
def inject_user_preference(request: ModelRequest) -> str:
    """
    根据用户 ID 动态生成系统提示
    """
    user_id = request.runtime.context.user_id
    user_pref = store.get(("preference",), user_id)

    if user_pref:
        return f"用户 {user_pref.value['name']} 喜欢 {user_pref.value['style']} 的回复风格"
    return "默认助手风格"
```

## 完整代码示例

基于 `src/langchain/context_engineering.py`：

```python
#!/user/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field

from provider import qwenLLM

# ============================================
# 1. 定义上下文模型
# ============================================
@dataclass
class Context:
    user_id: str

# ============================================
# 2. 定义数据结构
# ============================================
class WritingInfo(BaseModel):
    """用户写作风格偏好"""
    tone: Literal["admin", "user", "guest"] = Field(description="用户喜欢的写作风格")
    greeting: str = Field(description="问候语")
    sign_off: str = Field(description="用户的签名")

class UserInfo(BaseModel):
    """用户基本信息"""
    user_id: str = Field(description="用户id")
    username: str = Field(description="用户名")
    email: str = Field(description="用户邮箱")

# 模拟用户数据库
all_users: list[UserInfo] = [
    UserInfo(user_id="user_id_001", username="张三", email="zhangsan@qq.com"),
    UserInfo(user_id="user_id_002", username="李四", email="lisi@qq.com"),
]

# 命名空间
namespace = ("writing_style", )

# ============================================
# 3. 定义工具
# ============================================
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> UserInfo:
    """获取用户信息"""
    user_id = runtime.context.user_id
    return list(filter(lambda user: user.user_id == user_id, all_users))[0]

@tool
def extract_and_save_user_writing(query: str, runtime: ToolRuntime[Context]):
    """
    提取并保存用户的写作风格偏好
    当用户表达偏好时调用此工具
    """
    chain = qwenLLM.with_structured_output(WritingInfo)
    result = chain.invoke(query)

    store = runtime.store
    user_id = runtime.context.user_id

    # 存入长期记忆
    store.put(namespace, user_id, {
        "tone": result.tone,
        "greeting": result.greeting,
        "sign_off": result.sign_off,
    })
    return "保存用户写作喜好成功！"

# ============================================
# 4. 中间件：注入写作风格
# ============================================
@wrap_model_call
def inject_writing_style(request: ModelRequest, handler: Callable[[ModelRequest], ModelRequest]):
    """
    在每次模型调用前，从 Store 中读取用户写作风格并注入
    """
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    writing_style = store.get(namespace, user_id)

    if writing_style:
        style = writing_style.value
        style_context = f"""用户的创作风格：
            - 语气: {style.get('tone', 'professional')}
            - 问候语: "{style.get('greeting', 'Hi')}"
            - 签名: "{style.get('sign_off', 'Best')}"
        """
        # 将风格信息注入到消息中
        messages = [
            *request.messages,
            HumanMessage(style_context)
        ]
        request = request.override(messages=messages)

    return handler(request)

# ============================================
# 5. 配置
# ============================================
db_uri = "postgresql://postgres:postgres@localhost:5432/my_langchain?client_encoding=utf8"
config = RunnableConfig(
    configurable={"thread_id": "chat_conversation_001"}
)

# ============================================
# 6. 创建 Agent（集成长期记忆组件）
# ============================================
with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    with PostgresStore.from_conn_string(db_uri) as store:
        # 初始化数据库表
        checkpointer.setup()
        store.setup()

        agent = create_agent(
            qwenLLM,
            tools=[get_user_info, extract_and_save_user_writing],
            middleware=[inject_writing_style],     # 动态注入写作风格
            checkpointer=checkpointer,             # 短期记忆
            store=store,                          # 长期记忆
            context_schema=Context
        )

        # 第一次对话：提取并保存用户偏好
        # agent.invoke({
        #     "messages": [HumanMessage(
        #         "请保存我喜欢的写作风格：语气是幽默，签名需要获取我当前的名字，问候语为你好"
        #     )]
        # }, context=Context(user_id="user_id_001"), config=config)

        # 第二次对话：自动使用保存的偏好
        result = agent.invoke(
            {
                "messages": [HumanMessage("请写一个关于程序员的冷笑话")]
            },
            context=Context(user_id="user_id_001"),
            config=config
        )

        print(result['messages'][-1].pretty_print())
```

## 长期记忆的数据流

```
┌─────────────────────────────────────────────────────┐
│                   新对话开始                         │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              runtime.context.user_id                 │
│                  (获取用户 ID)                       │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           PostgresStore (长期记忆)                  │
│    store.get(("writing_style",), user_id)          │
│              (查询用户偏好)                          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           inject_writing_style 中间件                │
│        (将偏好注入到消息上下文中)                    │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                    LLM / Agent                      │
│        (使用用户的个性化偏好生成回复)                 │
└─────────────────────────────────────────────────────┘
```

## 存储结构设计

### 示例：用户写作风格存储

```python
namespace = ("writing_style",)

# 存储结构
{
    "user_001": {
        "tone": "humorous",
        "greeting": "你好呀",
        "sign_off": "笑口常开",
        "updated_at": "2026-04-02"
    },
    "user_002": {
        "tone": "formal",
        "greeting": "您好",
        "sign_off": "此致敬礼",
        "updated_at": "2026-04-01"
    }
}
```

### 示例：实体记忆存储

```python
namespace = ("entity", "company")

# 存储公司实体信息
{
    "AcmeCorp": {
        "name": "Acme Corporation",
        "industry": "Technology",
        "products": ["Widget Pro", "Gadget Plus"],
        "support_email": "support@acme.com"
    }
}
```

## 适用场景

| 场景 | 存储内容 | 读取时机 |
|------|---------|---------|
| **用户偏好** | 语气风格、问候语、签名 | 每次请求前 |
| **实体信息** | 联系人、公司、产品详情 | 按需查询 |
| **历史交互** | 过往问题、已提供的答案 | 新对话开始时 |
| **知识积累** | 领域知识、常见问题 | RAG 增强 |

## 短期记忆 vs 长期记忆

| 特性 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| **存储介质** | Checkpointer (PostgresSaver/InMemorySaver) | Store (PostgresStore) |
| **生命周期** | 会话级 | 永久 |
| **用途** | 对话历史、任务状态 | 用户偏好、实体信息 |
| **访问方式** | thread_id 自动关联 | 主动查询 |
| **数据特点** | 消息流 | 结构化数据 |

## 后续文档

- [企业生产环境最佳实践](./enterprise_best_practices.md) - 存储选型、隐私合规
- [客服场景记忆方案](./memory_in_customer_support.md)
