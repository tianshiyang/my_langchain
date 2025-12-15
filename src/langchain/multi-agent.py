#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/15 22:02
@Author  : tianshiyang
@File    : multi-agent.py
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from provider import qwenLLM


# ============================================================================
# 第一步 1: 定义最底层的工具
# ============================================================================
def create_calendar_event(
        title: str,
        start_time: str,
        end_time: str,
        attendees: list[str],
        location: str = ""
):
    """
    创建一个日程
    Args:
        title: 日程主题
        start_time: 开始时间 格式如2025-12-15 12:00:00
        end_time: 结束时间 格式如2025-12-15 14:00:00
        attendees: 参会人的邮件列表
        location: 参会地点

    Returns: 创建日程成功的提示文案

    """
    return f"创建成功：{title}从{start_time}到{end_time}, 包含{len(attendees)}个参会者，日程地点：{location}"

@tool
def send_email(
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] = []
) -> str:
    """
    使用邮件API发送邮件
    Args:
        to: 发送方，格式邮箱集合
        subject: 主题
        body: 内容
        cc: 抄送

    Returns:

    """
    return f"邮件已经发送给{''.join(to)} - 主题：{subject}"

@tool
def get_available_time_slots(
        attendees: list[str],
        date: str,
        duration_minutes: int,
) -> list[str]:
    """
    检查指定参会者在特定日期的日历可用性
    Args:
        attendees: 参会者
        date: 某一天
        duration_minutes: 持续时间

    Returns:
        返回用户已经存在日程的时间段
    """
    return ["09:00", "14:00", "16:00"]

# ============================================================================
# 第二步：创建特定的sub-agent
# ============================================================================

calendar_agent = create_agent(
    qwenLLM,
    tools=[create_calendar_event, get_available_time_slots],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="日程的创建等待审核",
        ),
    ],
    system_prompt=("你是一个日程安排助理" 
                   "解析自然语言调度请求（例如，‘下周二下午2点’）"
                   "在需要时使用get_available_time_slots来检查可用性。"
                   "使用create_calendar_event来安排事件"
                   "在最后的回复中一定要确认你的计划"
                   )
)

email_agent = create_agent(
    qwenLLM,
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="发送邮件等待审核"
        )
    ],
    system_prompt=(
        "你是一个发送邮件的助手. "
        "根据自然语言要求撰写专业电子邮件. "
        "提取收件人信息，制作合适的主题行和正文。"
        "使用send_email去发送邮件"
        "在最后的回复中一定要确认发送的内容。"
    )
)

# ============================================================================
# 步骤3：将sub-agents包装为主agent的工具
# ============================================================================

@tool
def schedule_event(request: str) -> str:
    """
    当用户想要创建、修改或检查日历约会时，使用此选项。处理日期/时间解析、可用性检查和事件创建。
    Args:
        request: 自然语言调度请求（例如，下星期二下午2点，与设计团队开会）
    Returns:

    """
    result = calendar_agent.invoke({
        "messages": [HumanMessage(request)]
    })
    return result['messages'][-1].text

@tool
def manage_email(request: str) -> str:
    """
    用自然语言发送邮件。当用户想要发送通知、提醒或任何电子邮件时，使用此选项处理收件人提取、主题生成和电子邮件
    Args:
        request: 自然语言的电子邮件请求（例如，“向他们发送关于……的提醒”）

    Returns:

    """
    result = email_agent.invoke({
        "messages": [HumanMessage(request)]
    })
    return result['messages'][-1].text

# ============================================================================
# 第四步：创建主agent
# ============================================================================
supervisor_agent = create_agent(
    qwenLLM,
    tools=[schedule_event, manage_email],
    checkpointer=InMemorySaver(),
    system_prompt=("你是一个乐于助人的机器人"
                   "你可以创建日程和发送电子邮件。"
                   "将用户请求分解为适当的工具调用，并协调结果"
                   "当一个请求涉及多个操作时，请按顺序使用多个工具"
                   )
)

# ============================================================================
# 步骤5：使用主agent
# ============================================================================
if __name__ == '__main__':
    config = RunnableConfig(
        configurable={
            "thread_id": 6
        }
    )
    user_request = (
        "下周二下午两点和设计团队开一个1小时的会议，给他们发一封电子邮件，提醒他们检查新的原型。"
    )
    print("\n" + "="*80 + "\n")
    interrupts = []
    for step in supervisor_agent.stream({
            "messages": [HumanMessage(user_request)]
        },
        config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)

    resume = {}
    for interrupt_ in interrupts:
        for request in interrupt_.value["action_requests"]:
            if request['name'] == 'send_email':
                edited_action = interrupt_.value["action_requests"][0].copy()
                edited_action["args"]["subject"] = "原型提醒"
                resume[interrupt_.id] = {
                    "decisions": [{"type": "edit", "edited_action": edited_action}]
                }
            else:
                resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    print(Command(resume=resume), 'resume===')

    for step in supervisor_agent.stream(
        Command(resume=resume),
        config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)