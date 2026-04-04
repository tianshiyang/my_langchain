#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
方式一：直接工具模式 (Direct Tools)

这是最简单直接的方式，适合：
- 工具数量较少（<10个）
- 工具描述简单明确
- 不需要复杂的权限控制或动态过滤

核心流程：
1. 用 @tool 装饰器定义工具
2. 创建 agent 时通过 tools 参数传入
3. Agent 直接调用工具

官方文档：https://docs.langchain.com/oss/python/langchain/agents
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from provider import get_default_model


# =============================================================================
# 1. 定义工具（Skills）
# =============================================================================

@tool
def get_weather(location: str) -> str:
    """
    获取指定位置的天气信息

    Args:
        location: 城市名称，如"北京"、"上海"

    Returns:
        天气描述字符串
    """
    # 实际项目中这里会调用天气 API
    return f"{location}今天晴天，26摄氏度，湿度45%"


@tool
def search_news(keyword: str, limit: int = 5) -> str:
    """
    搜索新闻

    Args:
        keyword: 搜索关键词
        limit: 返回数量，默认5条

    Returns:
        新闻列表字符串
    """
    # 实际项目中这里会调用新闻 API
    news = [
        f"1. {keyword}最新动态 - 今天",
        f"2. {keyword}行业分析 - 本周",
        f"3. {keyword}市场报告 - 本月",
    ]
    return "\n".join(news[:limit])


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式，如 "2 + 2"、"100 * 0.15"

    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# =============================================================================
# 2. 创建 Agent（直接传入工具列表）
# =============================================================================

# 创建带有多个工具的 agent
assistant_agent = create_agent(
    model=get_default_model(),
    tools=[get_weather, search_news, calculate],  # 工具列表
    system_prompt=(
        "你是一个多功能助手，可以查询天气、搜索新闻、执行计算。"
        "根据用户需求选择合适的工具响应。"
    ),
)


# =============================================================================
# 3. 使用 Agent
# =============================================================================

def demo_direct_tools():
    """演示直接工具模式"""

    print("=" * 70)
    print("方式一：直接工具模式 (Direct Tools)")
    print("=" * 70)

    # 示例1：查询天气
    print("\n【示例1】查询天气:")
    result = assistant_agent.invoke({
        "messages": [HumanMessage("北京今天天气怎么样？")]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例2：搜索新闻
    print("\n【示例2】搜索新闻:")
    result = assistant_agent.invoke({
        "messages": [HumanMessage("搜索关于人工智能的最新新闻")]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例3：计算
    print("\n【示例3】执行计算:")
    result = assistant_agent.invoke({
        "messages": [HumanMessage("计算 125 * 8 + 300 等于多少？")]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()


if __name__ == "__main__":
    demo_direct_tools()