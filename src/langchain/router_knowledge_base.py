#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/7 22:39
@Author  : tianshiyang
@File    : router_knowledge_base.py
"""
import operator
from typing import TypedDict, Literal, Annotated

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from provider import get_default_model


# 定义状态
class AgentInput(TypedDict):
    """每个子智能体的简单输入状态"""
    query: str

class AgentOutput(TypedDict):
    """每个子智能体的输出"""
    source: str
    result: str

class Classification(TypedDict):
    """一个单一的路由决策：应调用哪个智能体，以及使用什么查询内容"""
    source: Literal["github", "notion", "slack"]
    query: str

class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add] # Reducer 负责收集并行执行的结果
    final_answer: str

# 2.为每个垂直行业定义工具
@tool
def search_code(query: str, repo: str = "main") -> str:
    """在 GitHub 仓库中搜索代码"""
    return f"在 {repo} 中找到了与 '{query}' 匹配的代码：位于 src/auth.py 的身份验证中间件"

@tool
def search_issues(query: str) -> str:
    """搜索 GitHub 问题和拉取请求"""
    return f"找到3个与'{query}'匹配的问题：#142（API认证文档）、#89（OAuth流程）、#203（令牌刷新）"

@tool
def search_prs(query: str) -> str:
    """搜索拉取请求以获取实现细节."""
    return f"PR #156 添加了JWT认证，PR #178 更新了OAuth范围"


@tool
def search_notion(query: str) -> str:
    """在Notion工作区搜索文档。."""
    return f"找到文档：'API认证指南' - 涵盖OAuth2流程、API密钥和JWT令牌"


@tool
def get_page(page_id: str) -> str:
    """通过ID获取特定的Notion页面。"""
    return f"页面内容：逐步认证设置说明"


@tool
def search_slack(query: str) -> str:
    """搜索Slack消息和线程."""
    return f"在#engineering频道发现讨论：'使用Bearer令牌进行API认证，详情请参阅刷新流程文档'"


@tool
def get_thread(thread_id: str) -> str:
    """获取特定的Slack线程。"""
    return f"线程讨论了API密钥轮换的最佳实践"

# 3. 创建agent
github_agent = create_agent(
    get_default_model(),
    tools=[search_code, search_issues, search_prs],
    system_prompt="你是一名GitHub专家。通过搜索仓库、问题和拉取请求，回答关于代码、API参考和实现细节的问题"
)

notion_agent = create_agent(
    get_default_model(),
    tools=[search_notion, get_page],
    system_prompt="你是一名Notion专家。通过搜索组织的Notion工作区来回答有关内部流程、政策和团队文档的问题"
)

slack_agent = create_agent(
    get_default_model(),
    tools=[search_slack, get_thread],
    system_prompt="你是一个Slack专家。通过搜索相关线程和讨论来回答问题，团队成员在这些地方分享了知识和解决方案"
)

# 4. 构建路由工作流程

router_llm = get_default_model()
# 为分类器定义结构化输出模式
class ClassificationResult(BaseModel):
    # 将用户查询分类为面向特定智能体的子问题的结果
    classifications: list[Classification] = Field(
        description="要调用的智能体列表及其对应的目标子问题"
    )

def classify_query(state: RouterState) -> dict:
    """对查询进行分类，并确定需要调用哪些智能体"""
    structured_llm = router_llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke([
        SystemMessage(content="请提供您要分析的具体查询（query）内容，我将根据上述规则判断需咨询哪些知识库，并为每个相关来源生成针对性的子问题。"),
        HumanMessage(content=state["query"])
    ])
    print("#"*30)
    print(result.classifications)
    print("#" * 30)
    return {"classifications": result.classifications}

def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类结果，将任务分发（fan out）给相应的智能体"""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]

def query_github(state: AgentInput) -> dict:
    """查询GitHub的Agent"""
    result = github_agent.invoke({
        "messages": HumanMessage(content=state["query"])
    })
    return {
        "results": [{"source": "github", "result": result["messages"][-1].content}]
    }

def query_notion(state: AgentInput) -> dict:
    """查询 Notion 智能体"""
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput) -> dict:
    """查询Slack智能体."""
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}

def synthesize_results(state: RouterState) -> dict:
    """正在整合所有智能体的查询结果，生成连贯、完整的回答"""
    if not state["results"]:
        return {"final_answer": "未从任何知识源中找到相关结果"}
    # 为综合处理而格式化结果
    formatted = [
        f"**{r['source'].title()}: **\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        SystemMessage(content=f"""
            综合以下搜索结果，回答原始问题：{state['query']}
            - 整合多个来源的信息，避免重复
            - 突出最相关且可操作的内容
            - 指出各来源之间存在的任何不一致之处
            - 保持回答简洁、条理清晰
        """),
        HumanMessage(content="\n\n".join(formatted))
    ])
    return {"final_answer": synthesis_response.content}

# 5. 整理工作流程

workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)

result = workflow.invoke({
    "query": "如何对 API 请求进行身份验证?"
})

print("Original query:", result["query"])
print("\nClassifications:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['query']}")
print("\n" + "=" * 60 + "\n")
print("Final Answer:")
print(result["final_answer"])