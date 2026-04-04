#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
方式三：子Agent协作模式 (Sub-Agent Collaboration)

这是处理复杂多领域任务的推荐方式，适合：
- 任务需要多个专业领域协作
- 每个领域需要独立的推理和工具调用
- 需要层次化的任务分解和协调

核心思想：
1. 为每个专业领域创建专门的 sub-agent
2. 将 sub-agent 包装成主 agent 的工具
3. 主 agent（Supervisor）负责协调和决策

架构图：
┌─────────────────────────────────────┐
│         Supervisor Agent            │
│  (tools=[schedule, email, search])  │
└──────┬──────────┬──────────┬────────┘
       │          │          │
   ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
   │Calendar│  │ Email │  │Search │
   │ Agent │  │ Agent │  │ Agent │
   └───────┘  └───────┘  └───────┘

官方文档：https://docs.langchain.com/oss/python/langgraph/overview
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from provider import get_default_model


# =============================================================================
# 1. 定义底层工具
# =============================================================================

@tool
def create_event(title: str, start_time: str, end_time: str, attendees: list[str]) -> str:
    """
    创建日历事件

    Args:
        title: 事件标题
        start_time: 开始时间，格式如 "2026-04-10 14:00"
        end_time: 结束时间，格式如 "2026-04-10 15:00"
        attendees: 参会人邮箱列表

    Returns:
        创建结果描述
    """
    return f"✅ 日程已创建：{title}\n时间：{start_time} - {end_time}\n参会人：{', '.join(attendees)}"


@tool
def get_available_slots(date: str, duration_minutes: int) -> list[str]:
    """
    查询可用时间段

    Args:
        date: 日期，格式如 "2026-04-10"
        duration_minutes: 需要时长（分钟）

    Returns:
        可用时间列表
    """
    # 实际项目中这里会查询日历 API
    return ["09:00", "10:00", "14:00", "15:00", "16:00"]


@tool
def send_email(to: list[str], subject: str, body: str, cc: list[str] = None) -> str:
    """
    发送电子邮件

    Args:
        to: 收件人邮箱列表
        subject: 邮件主题
        body: 邮件正文
        cc: 抄送列表（可选）

    Returns:
        发送结果描述
    """
    cc_str = f"\n抄送：{', '.join(cc)}" if cc else ""
    return f"✅ 邮件已发送\n收件人：{', '.join(to)}\n主题：{subject}{cc_str}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    网络搜索

    Args:
        query: 搜索关键词
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    # 实际项目中这里会调用搜索 API
    results = [
        f"1. {query} - 维基百科",
        f"2. {query} 最新动态 - 新闻网站",
        f"3. {query} 官方文档",
    ]
    return "\n".join(results[:max_results])


# =============================================================================
# 2. 创建 Sub-Agents（专门领域的 Agent）
# =============================================================================

# 日程管理 Agent
calendar_agent = create_agent(
    model=get_default_model(),
    tools=[create_event, get_available_slots],
    system_prompt=(
        "你是一个专业的日程助理。\n"
        "职责：\n"
        "- 理解用户的日程需求（自然语言如'下周二下午2点'）\n"
        "- 使用 get_available_slots 检查时间可用性\n"
        "- 使用 create_event 创建日程\n"
        "- 最后确认日程详情"
    ),
)

# 邮件管理 Agent
email_agent = create_agent(
    model=get_default_model(),
    tools=[send_email],
    system_prompt=(
        "你是一个专业的邮件助理。\n"
        "职责：\n"
        "- 根据用户需求撰写专业邮件\n"
        "- 提取收件人信息\n"
        "- 生成合适的主题和正文\n"
        "- 使用 send_email 发送\n"
        "- 最后确认发送内容"
    ),
)

# 搜索 Agent
search_agent = create_agent(
    model=get_default_model(),
    tools=[web_search],
    system_prompt=(
        "你是一个专业的搜索助理。\n"
        "职责：\n"
        "- 理解搜索意图\n"
        "- 使用 web_search 执行搜索\n"
        "- 整理和总结搜索结果"
    ),
)


# =============================================================================
# 3. 将 Sub-Agents 包装成工具
# =============================================================================

@tool
def schedule_meeting(request: str) -> str:
    """
    处理日程安排请求。

    当用户想要创建日历事件、会议时使用此工具。
    例如："下周二下午2点开会"、"安排一个1小时的演示会议"

    Args:
        request: 自然语言日程请求

    Returns:
        日程创建结果
    """
    result = calendar_agent.invoke({
        "messages": [HumanMessage(request)]
    })
    return result["messages"][-1].text


@tool
def send_notification(request: str) -> str:
    """
    处理邮件发送请求。

    当用户想要发送通知、提醒邮件时使用此工具。
    例如："给团队发一封会议提醒"、"通知客户项目上线"

    Args:
        request: 自然语言邮件请求

    Returns:
        邮件发送结果
    """
    result = email_agent.invoke({
        "messages": [HumanMessage(request)]
    })
    return result["messages"][-1].text


@tool
def search_information(request: str) -> str:
    """
    处理信息搜索请求。

    当用户需要查找信息时使用此工具。
    例如："搜索最新的AI新闻"、"查找Python最佳实践"

    Args:
        request: 自然语言搜索请求

    Returns:
        搜索结果
    """
    result = search_agent.invoke({
        "messages": [HumanMessage(request)]
    })
    return result["messages"][-1].text


# =============================================================================
# 4. 创建 Supervisor Agent（主 Agent）
# =============================================================================

supervisor_agent = create_agent(
    model=get_default_model(),
    tools=[schedule_meeting, send_notification, search_information],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "你是一个智能助手，可以协调日程、邮件和搜索任务。\n\n"
        "当用户提出需求时：\n"
        "1. 分解任务，确定需要哪些能力\n"
        "2. 调用相应工具\n"
        "3. 协调结果，给出完整回复\n\n"
        "示例：\n"
        "- '下周二下午2点开会并通知参会人' -> 调用日程+邮件工具\n"
        "- '安排会议后搜索相关信息' -> 调用日程+搜索工具"
    ),
)


# =============================================================================
# 5. 使用
# =============================================================================

def demo_sub_agent():
    """演示子Agent协作模式"""

    print("=" * 70)
    print("方式三：子Agent协作模式 (Sub-Agent Collaboration)")
    print("=" * 70)

    print("""
    架构特点：
    - 每个专业领域有独立的 Agent
    - Sub-Agent 被包装成主 Agent 的工具
    - 主 Agent 负责任务分解和协调
    - 支持并行调用多个 Sub-Agent
    """)

    config = {"configurable": {"thread_id": "demo-001"}}

    # 示例1：简单搜索
    print("\n【示例1】信息搜索:")
    result = supervisor_agent.invoke({
        "messages": [HumanMessage("搜索一下今天有什么AI新闻")]
    }, config)
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例2：发送邮件
    print("\n【示例2】发送邮件:")
    result = supervisor_agent.invoke({
        "messages": [HumanMessage(
            "给团队成员 test@example.com 发一封邮件，"
            "主题是项目启动通知，内容包含项目目标和计划"
        )]
    }, config)
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例3：日程安排
    print("\n【示例3】日程安排:")
    result = supervisor_agent.invoke({
        "messages": [HumanMessage(
            "帮我安排一个4月15日下午3点的主题会议，时长1小时，"
            "参会人：manager@example.com, tech@example.com"
        )]
    }, config)
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()


if __name__ == "__main__":
    demo_sub_agent()