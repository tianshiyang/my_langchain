# LangChain 1.0 技能(Skill)加载方式演示

本目录展示了 LangChain 1.0 中三种推荐的技能加载方式，参考官方文档实现。

## 目录结构

```
skills_demo/
├── README.md              # 本文件
├── __init__.py            # 包入口
├── direct_tools.py        # 方式一：直接工具模式
├── progressive_disclosure.py  # 方式二：渐进式披露模式
└── sub_agent.py           # 方式三：子Agent协作模式
```

## 三种方式概览

### 方式一：直接工具模式 (Direct Tools)

最简单的技能加载方式，直接用 `@tool` 装饰器定义工具，创建 agent 时通过 `tools` 参数传入。

**适用场景：**
- 工具数量较少（<10个）
- 工具描述简单明确
- 不需要复杂的权限控制或动态过滤

**核心代码：**

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"{location} 晴天，26度"

agent = create_agent(
    model=llm,
    tools=[get_weather],  # 直接传入工具列表
    system_prompt="你是一个天气助手"
)
```

---

### 方式二：渐进式披露模式 (Progressive Disclosure)

LangChain 推荐的技能加载方式，适合技能内容很大、需要按需加载的场景。

**核心思想：**
1. 技能内容定义成结构化数据（`Skill` TypedDict）
2. 通过中间件将技能描述注入 system prompt
3. Agent 使用 `load_skill` 工具按需加载完整内容

**适用场景：**
- 技能/知识库内容很大（如详细文档、操作手册）
- 技能数量多，不可能全部塞进 system prompt
- 需要避免 context 溢出

**核心代码：**

```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

# 1. 定义技能
class Skill(TypedDict):
    name: str
    description: str      # 简短描述，用于 system prompt
    content: str          # 完整内容，按需加载

SKILLS = [
    {"name": "code_review", "description": "代码审查指南", "content": "..."},
    {"name": "data_analysis", "description": "数据分析技能", "content": "..."},
]

# 2. 创建技能加载工具
@tool
def load_skill(skill_name: str) -> str:
    for skill in SKILLS:
        if skill_name == skill["name"]:
            return skill["content"]
    return "技能不存在"

# 3. 创建技能中间件
class SkillsMiddleware(AgentMiddleware):
    tools = [load_skill]

    def wrap_model_call(self, request, handler):
        # 将技能描述注入 system prompt
        skills_text = "\n".join([f"- {s['name']}: {s['description']}" for s in SKILLS])
        new_prompt = request.system_message + f"\n\n可用技能：\n{skills_text}"
        return handler(request.override(system_message=new_prompt))

# 4. 创建 agent
agent = create_agent(
    llm,
    middleware=[SkillsMiddleware()],
    system_prompt="你是一个专业助手..."
)
```

---

### 方式三：子Agent协作模式 (Sub-Agent)

处理复杂多领域任务的推荐方式。

**架构：**

```
┌─────────────────────────────────────┐
│         Supervisor Agent            │
│  (tools=[schedule, email, search])  │
└──────┬──────────┬──────────┬────────┘
       │          │          │
   ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
   │Calendar│  │ Email │  │Search │
   │ Agent │  │ Agent │  │ Agent │
   └───────┘  └───────┘  └───────┘
```

**适用场景：**
- 任务需要多个专业领域协作
- 每个领域需要独立的推理和工具调用
- 需要层次化的任务分解和协调

**核心步骤：**

1. **定义底层工具**
2. **创建 Sub-Agent**（每个专业领域一个）
3. **将 Sub-Agent 包装成主 Agent 的工具**
4. **创建 Supervisor Agent**（协调者）

---

## 使用方式

```python
# 运行单个 demo
from skills_demo import demo_direct_tools, demo_progressive_skills, demo_sub_agent

demo_direct_tools()          # 演示直接工具模式
demo_progressive_skills()    # 演示渐进式披露模式
demo_sub_agent()             # 演示子Agent协作模式
```

或者直接运行文件：

```bash
python -m skills_demo.direct_tools
python -m skills_demo.progressive_disclosure
python -m skills_demo.sub_agent
```

## 参考资料

- [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview/)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)