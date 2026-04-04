#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/5 23:24
@Author  : tianshiyang
@File    : customer_support.py
"""
from typing import Literal, NotRequired, Callable

import uuid
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from provider import get_default_model

# 定义可能的工作流步骤
"""
warranty_collector：保修信息收集员
issue_classifier：问题分类器
resolution_specialist： 问题解决专员
"""
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """客户服务流程的状态"""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]] # 服务器内和服务期外
    issue_type: NotRequired[Literal["hardware", "software"]] # 问题类型，硬件、软件


############################创建用于管理工作流状态的工具#############################
################################################################################

@tool
def record_warranty_status(
        status: Literal["in_warranty", "out_of_warranty"],
        runtime: ToolRuntime[None, SupportState]
) -> Command:
    # 记录客户的保修状态，并转入问题分类环节
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录为：{status}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier"
        },
    )

@tool
def record_issue_type(
        issue_type: Literal["hardware", "software"],
        runtime: ToolRuntime[None, SupportState]
) -> Command:
    """记录问题类型，并转接至解决方案专员。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型记录为{issue_type}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist"
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """将该案件升级转交给人工客服专员。"""
    # 在真实系统中，这将创建一个工单、通知相关人员等。
    return f"正在转接至人工客服。原因：{reason}"

@tool
def provide_solution(solution: str) -> str:
    """为客户的问题提供解决方案"""
    return f"解决方案已提供: {solution}"

WARRANTY_COLLECTOR_PROMPT = """
你是一名协助客户解决设备问题的客服专员。

当前阶段：保修状态核实

在此步骤中，你需要：

热情地向客户问好
询问客户的设备是否仍在保修期内
使用 record_warranty_status 记录客户的回答，并进入下一步
请保持对话自然友好，一次只问一个问题。
"""

ISSUE_CLASSIFIER_PROMPT = """
你是一名协助客户解决设备问题的客服专员。

当前阶段：问题分类

客户信息：保修状态为 {warranty_status}

在此步骤中，你需要：

请客户描述他们遇到的问题
判断问题是硬件问题（如物理损坏、零件故障）还是软件问题（如应用崩溃、性能异常）
使用 record_issue_type 记录问题分类，并进入下一步
如果问题不明确，请先提出澄清性问题，再进行分类。
"""

RESOLUTION_SPECIALIST_PROMPT = """
你是一名协助客户解决设备问题的客服专员。

当前阶段：问题解决

客户信息：保修状态为 {warranty_status}，问题类型为 {issue_type}

在此步骤中，你需要：

1. 如果是软件问题：使用 provide_solution 提供具体的故障排除步骤
2. 如果是硬件问题：
   - 若在保修期内：使用 provide_solution 说明保修维修流程
   - 若已过保：使用 escalate_to_human 将客户转接至人工专员，以便提供付费维修选项
请确保你的解决方案具体、清晰且有帮助
"""

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}

@wrap_model_call
def apply_step_config(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
    """根据当前步骤配置智能体行为"""
    # 获取当前步骤（首次交互默认为warranty_collector）
    current_step = request.state.get("current_step", "warranty_collector")
    # 查找步骤配置
    stage_config = STEP_CONFIG[current_step]
    # 校验状态是否存在
    for key in stage_config['requires']:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # 使用状态值格式化提示（支持{warranty_status}、{issue_type}等）
    system_prompt = stage_config["prompt"].format(**request.state)

    # 注入系统提示和步骤特定工具
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

agent = create_agent(
    get_default_model(),
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model=get_default_model(),
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )],
    checkpointer=InMemorySaver(),
)

# Configuration for this conversation thread
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Turn 1: Initial message - starts with warranty_collector step
print("=== Turn 1: Warranty Collection ===")
result = agent.invoke(
    {"messages": [HumanMessage("嗨，我的手机屏幕裂了")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()

# Turn 2: User responds about warranty
print("\n=== Turn 2: Warranty Response ===")
result = agent.invoke(
    {"messages": [HumanMessage("是的，它还在保修期内")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 3: User describes the issue
print("\n=== Turn 3: Issue Description ===")
result = agent.invoke(
    {"messages": [HumanMessage("屏幕因掉落而物理性破裂")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 4: Resolution
print("\n=== Turn 4: Resolution ===")
result = agent.invoke(
    {"messages": [HumanMessage("我该怎么办?")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()