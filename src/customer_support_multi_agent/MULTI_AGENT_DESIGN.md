# 客服多智能体系统设计文档

## 1. 整体架构

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│          Router（意图分类器）             │
│  classify_customer_query()              │
│  判断意图 + 提取槽位 + 生成执行计划        │
└─────────────────┬───────────────────────┘
                  │ CustomerIntentResult
                  ▼
┌─────────────────────────────────────────┐
│    Supervisor Agent（总控智能体）          │
│  _build_supervisor_agent()              │
│  根据 route_plan 决定调用哪个 specialist  │
│  tools: logistics_specialist            │
│         product_specialist              │
│         refund_specialist               │
└────┬──────────┬──────────────┬───────────┘
     │          │              │
     ▼          ▼              ▼
┌──────────┐┌──────────┐┌──────────────────┐
│ Logistics││ Product  ││  Refund Agent    │
│Specialist││Specialist││  (Handoff 状态机)  │
└──────────┘└──────────┘└──────────────────┘
     │          │              │
     ▼          ▼              ▼
┌─────────────────────────────────────────┐
│        TaskResult 列表                   │
│  由 synthesize_support_response() 汇总   │
└─────────────────┬───────────────────────┘
                  ▼
            Final Answer
```

---

## 2. 各组件职责

### 2.1 Router（`classify_customer_query`）

**位置**：`support_shared.py`

**职责**：接收用户原始文本，返回结构化的路由计划 `CustomerIntentResult`。

**意图类型**（三种互斥/可并存）：

| 意图 | 必要槽位 | 说明 |
|------|---------|------|
| `物流查询` | `order_id` | 订单轨迹、运单号、ETA |
| `商品咨询` | `product_name` 或 `order_id` | 价格、库存、规格参数 |
| `退款` | `order_id` + `refund_reason` | 资格判断 + 提交/升级人工 |

**降级策略**：

1. 优先用 LLM 结构化输出（`model.with_structured_output(CustomerIntentResult)`）
2. 若 LLM 调用失败，回退到 `keyword_classify_customer_query`（纯规则兜底）

**输出结构**：

```python
CustomerIntentResult(
    intents: list[IntentType],           # 识别到的意图列表
    sub_tasks: list[SubTask],             # 每个意图对应一个原子任务
    missing_slots: list[str],             # 所有任务缺失槽位的并集
    need_clarification: bool,             # 是否需要追问
    clarification_question: str,           # 追问文案
    confidence: float,
)
```

---

### 2.2 Supervisor Agent（`LangChainCustomerSupportApp._build_supervisor_agent`）

**位置**：`langchain_customer_support.py`

**本质**：一个 LangChain `create_agent`，拥有三个 `tool`，通过 `system_prompt` 约束其行为。

**Tools 注册**：

| Tool | 对应 intent | 实现函数 |
|------|------------|---------|
| `logistics_specialist` | 物流查询 | `run_logistics_specialist(app.logistics_agent, task)` |
| `product_specialist` | 商品咨询 | `run_product_specialist(app.product_agent, task)` |
| `refund_specialist` | 退款 | `run_refund_specialist(app.refund_agent, task, refund_thread_id)` |

**Supervisor 的行为约束**（来自 system_prompt）：

1. 严格依据 `route_plan` 决定调用哪些 specialist
2. 每个 `sub_task` 对应一次 specialist 调用
3. 若 specialist 返回 `blocked_waiting_user`，不要假装完成
4. 只有所有任务达到终态（`done` / `blocked_waiting_user` / `escalated_to_human`）才输出最终回复
5. 最终回复先回答已完成部分，再追问缺失信息，有人工审核要明确后续动作

**降级机制**：`handle_request` 中 `try/except` 捕获 supervisor 调用失败，自动回退到**确定性分发模式**，直接遍历 `sub_tasks` 串行调用对应 specialist，跳过 supervisor。

---

### 2.3 Specialist Agents

#### 2.3.1 Logistics Specialist（`create_logistics_agent`）

**Tools**：`lookup_order`、`query_logistics`

**职责**：根据 `order_id` 查询订单 + 物流状态，返回给用户。

**兜底逻辑**：若 agent 调用失败，直接调用 `query_logistics_service()` 获取结果。

#### 2.3.2 Product Specialist（`create_product_agent`）

**Tools**：`lookup_order`、`search_product_info`

**职责**：根据商品名或订单号，查询商品信息（价格、库存、卖点）。

**兜底逻辑**：若 agent 调用失败，通过 `search_product_service()` 直接获取结果。

#### 2.3.3 Refund Specialist（`create_refund_agent`）

**Tools**：`record_refund_order_id`、`record_refund_reason`、`check_refund_eligibility`、`submit_refund_request`、`escalate_refund_case`、`decline_refund_case`

**特点**：使用 **Handoff（状态机）模式**，退款流程分阶段推进：

```
collect_order → collect_reason → resolve_refund → completed
```

| 阶段 | Prompt | 可用 Tools |
|------|--------|-----------|
| `collect_order` | 请用户提供订单号 | `record_refund_order_id` |
| `collect_reason` | 确认了订单号，继续收集退款原因 | `record_refund_reason` |
| `resolve_refund` | 已知订单号+原因，检查资格并决策 | `check_refund_eligibility` + 决策类 tools |
| `completed` | 总结结果，结束本轮 | 无 |

**退款决策树**（`resolve_refund` 阶段）：

```
check_refund_eligibility(order_id)
    │
    ├── eligible=true, audit_required=false
    │       └── submit_refund_request() → 退款单号
    │
    ├── eligible=true, audit_required=true
    │       └── escalate_refund_case() → 人工工单
    │
    └── eligible=false
            └── decline_refund_case() → 拒绝，说明原因
```

**Thread 隔离**：`refund_specialist` 使用 `InMemorySaver` checkpointer，同一 `thread_id` 内的多轮退款对话共享状态，不会被其他订单请求干扰。

**Refund Thread ID 构造**：`{原始thread_id}:{task_id}`，确保同一用户不同退款任务的状态严格隔离。

---

### 2.4 兜底与容错

| 场景 | 处理方式 |
|------|---------|
| Router LLM 分类失败 | 回退到 `keyword_classify_customer_query` 规则兜底 |
| Supervisor agent 调用失败 | 降级为确定性分发：直接遍历 `sub_tasks` 串行调用 specialist |
| Specialist agent 调用失败 | 直接调用底层 `*_service()` 函数获取结果 |
| 槽位缺失（`missing_slots` 非空） | 返回 `blocked_waiting_user` 状态，直接向用户追问 |

---

## 3. 数据流详解

以 **"帮我查订单 A1001 的物流，另外我想把 A1003 退掉，原因是买错了"** 为例：

### 第一步：Router 分类

输入文本 → LLM 结构化输出 → 生成 `CustomerIntentResult`：

```json
{
  "intents": ["物流查询", "退款"],
  "sub_tasks": [
    {
      "task_id": "task_abc12345",
      "intent": "物流查询",
      "slots": {"order_id": "A1001"},
      "missing_slots": []
    },
    {
      "task_id": "task_def67890",
      "intent": "退款",
      "slots": {"order_id": "A1003", "refund_reason": "买错了"},
      "missing_slots": []
    }
  ],
  "need_clarification": false
}
```

### 第二步：Supervisor 分发

Supervisor 收到 route_plan，执行以下决策：

1. 识别到 `task_abc12345` intent=物流查询 → 调用 `logistics_specialist(task_json)`
2. 识别到 `task_def67890` intent=退款 → 调用 `refund_specialist(task_json)`
3. 等待两个调用返回，汇总输出最终回复

### 第三步：并行执行 Specialists

```
logistics_specialist         refund_specialist
─────────────────────        ─────────────────────
lookup_order(A1001)           读取 slots: A1003 + 买错了
query_logistics(A1001)       进入 resolve_refund 阶段
返回物流状态                  check_refund_eligibility(A1003)
                              → eligible=true, audit_required=true
                              escalate_refund_case()
                              → TK-xxx 工单
```

### 第四步：汇总响应

`synthesize_support_response()` 将两个 `TaskResult` 合并成一段自然语言返回：

> 订单 A1001 当前物流状态为"运输中"，运单号 SF1234567890，预计送达时间 2026-04-02。
>
> 订单 A1003 因金额较高，需要人工审核才能退款，已为您创建工单 TK-xxx，审核结果会在 1-3 个工作日内通知您。

---

## 4. 关键设计决策

### 4.1 为什么用 Supervisor Agent 而不是直接分发？

Supervisor 的优势在于**意图理解 + 动态决策**。给定同一 route_plan，它可以根据任务数量、返回状态自主决定调用顺序和次数，而不是硬编码的 if-else。此外，Supervisor 在多意图场景下可以**并行调用**多个 specialist（通过 LangChain agent 的 tool calling 机制），提升响应速度。

### 4.2 退款为什么用 Handoff 状态机？

退款是**多轮交互**极强的场景：用户可能只说"我要退款"而没给订单号，需要追问；给了订单号后还要收集原因。Handoff 状态机通过分阶段 `system_prompt` + 阶段性 `tools`，确保 agent 在每个阶段**只推进最关键的一个字段**，不会跳跃也不会遗漏。

### 4.3 Thread ID 隔离

退款是写操作，若不隔离，同一用户的多笔退款请求会互相覆盖状态。`refund_thread_id = f"{thread_id}:{task_id}"` 确保每个退款子任务有独立的状态存储。

### 4.4 异常降级保证可用性

整个链路设计了多层兜底：

```
LLM 分类失败 → 规则兜底
Supervisor 调用失败 → 确定性分发（跳过 agent 直接调 service）
Agent 执行失败 → 直接调底层 service 函数
```

即使所有 agent 都不可用，系统仍能返回结构化结果（虽然是兜底的）。

---

## 5. 文件清单

| 文件 | 职责 |
|------|------|
| `models.py` | 共享类型定义：`SubTask`、`TaskResult`、`CustomerIntentResult`、`SupportContext`、`RefundState` |
| `support_shared.py` | 核心逻辑：Router、三个 Specialist Agent工厂、退款状态机、兜底 service 函数 |
| `langchain_customer_support.py` | 入口类 `LangChainCustomerSupportApp`、Supervisor Agent 构建、`handle_request()` 流程编排 |
