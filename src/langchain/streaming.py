#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 15:18
@Author  : tianshiyang
@File    : streaming.py
"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from provider import get_default_model

class UserRequest(BaseModel):
    messages: list[HumanMessage | AIMessage | ToolMessage]


@tool
def get_weather(city: str):
    """
    获取传入城市的天气信息
    Args:
        city: 城市名称

    Returns:
        返回该城市的天气情况
    """
    weather_data = {
        "北京": "晴天，温度 15°C，空气质量良好",
        "上海": "多云，温度 18°C，有轻微雾霾",
        "深圳": "阴天，温度 22°C，可能有小雨",
        "成都": "小雨，温度 12°C，湿度较高"
    }
    return weather_data.get(city, f"抱歉，暂时没有{city}的天气数据")

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    执行基本的数学计算

    参数:
        operation: 运算类型，支持 "add"(加), "subtract"(减), "multiply"(乘), "divide"(除)
        a: 第一个数字
        b: 第二个数字

    返回:
        计算结果字符串
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "错误：除数不能为零"
    }

    if operation not in operations:
        return f"不支持的运算类型：{operation}。支持的类型：add, subtract, multiply, divide"

    try:
        result = operations[operation](a, b)
        return f"{a} {operation} {b} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

def example_1_stream_mode_values():
    """
    示例1：使用 stream_mode="values"

    这是默认模式，每个步骤后返回完整的状态字典。

    ⚠️ 重要：messages 列表会随着 Agent 执行不断增长！
    所以需要用 messages[-1] 来获取最新添加的消息！
    """
    print("\n" + "=" * 70)
    print("示例 1：stream_mode='values'（默认模式）")
    print("=" * 70)

    agent = create_agent(
        get_default_model(),
        tools=[calculator]
    )

    print("\n问题：25 乘以 8 等于多少？")
    print("\n流式输出（values 模式）：")
    print("-" * 70)

    chunk_count = 0
    for chunk in agent.stream(
        UserRequest(messages=[HumanMessage("25 乘以 8 等于多少？")]),
        stream_mode="values"
    ):
        chunk_count += 1
        print("*"*60)
        print(f"chunk_count: {chunk_count}")
        print(f"类型: {type(chunk)}")
        print(f"Chunk 的键: {list(chunk.keys())}")

        if 'messages' in chunk:
            messages = chunk['messages']
            print(f"消息总数: {len(messages)}")
            print(f"\n📋 当前 messages 列表中的所有消息：")
            for i, msg in enumerate(chunk['messages'], 1):
                msg_type = msg.__class__.__name__
                print(f"  {i}. {msg_type}", end="")
                if hasattr(msg, 'content') and msg.content:
                    print(f" - {msg.content[:50]}...")
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f" - 调用工具: {msg.tool_calls[0]['name']}")
                else:
                    print()

        # 获取最后一条消息
        last_message = chunk['messages'][-1]
        last_message_type = last_message.__class__.__name__
        print(f"最新消息的类型, {last_message_type}")
        if last_message_type == "AIMessage":
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"AI准备调用工具: {last_message.tool_calls[0]['name']}")
                print(f"  → 工具参数: {last_message.tool_calls[0]['args']}")
            elif hasattr(last_message, 'content') and last_message.content:
                print(f"  → AI 最终回答: {last_message.content[:100]}...")
        elif last_message_type == "ToolMessage":
            print(f"  → 工具执行结果: {last_message.content}")

# ============================================================================
# 示例 2：stream_mode="updates"
# ============================================================================
def example_2_stream_mode_updates():
    """

    返回的是每一个工具输出的结果，即AIMessage OR ToolMessage
    其实就是本次更新的结果

    """
    print("\n" + "=" * 70)
    print("示例 2：stream_mode='updates'")
    print("=" * 70)

    agent = create_agent(
        get_default_model(),
        tools=[calculator]
    )

    print("\n问题：10 加 20 等于多少？")
    print("\n流式输出（updates 模式）：")
    print("-" * 70)

    chunks = agent.stream(
        UserRequest(messages=[HumanMessage("25 乘以 8 等于多少？")]),
        stream_mode="updates"
    )

    chunk_count = 0
    for chunk in chunks:
        chunk_count += 1
        print(f"\n【Chunk {chunk_count}】")
        print(f"类型: {type(chunk)}")
        print(f"Chunk 的键（节点/工具名）: {list(chunk.keys())}")

        for key, value  in chunk.items():
            cur_messages = value["messages"]
            print("*" * 60)
            print(f"本次更新的消息的数量, {len(cur_messages)}")
            for i, msg in enumerate(cur_messages, 1):
                msg_type = msg.__class__.__name__
                print(f"消息类型：{msg_type}")
                if msg_type == "ToolMessage":
                    print(f"工具返回的结果: {msg.content[:50]}...")
                elif msg_type == "AIMessage":
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"工具调用名称: {msg.tool_calls[0]['name']}")
                        print(f"工具调用参数: {msg.tool_calls[0]['args']}")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"工具返回的结果: {msg.content[:50]}...")

# ============================================================================
# 示例 3：stream_mode="messages"
# ============================================================================

def example_3_stream_mode_messages():
    """
    示例3：使用 stream_mode="messages"

    逐 token 返回 LLM 生成的消息，类似打字机效果。
    同时也会返回工具调用相关的消息（AIMessageChunk with tool_calls 和 ToolMessage）

    返回类型：tuple（元组，2个元素）

    字段结构：
    (
        message_chunk,  # 第一个元素：AIMessageChunk(1. AI真实的回复-token by token  2.决定调用什么工具的结构化输出 )、ToolMessage（工具调用返回结果） 等消息对象
        metadata       # 第二个元素：dict，包含元数据和 LangGraph 执行信息
    )

    特点：
    - 返回的是元组，不是字典
    - 第一个元素是消息对象（AIMessageChunk、ToolMessage 等）
    - 第二个元素是元数据（包含 LangGraph 执行信息和模型信息）
    - 会返回 Agent 执行过程中的所有消息，包括工具调用和最终答案
    - 适合需要实时显示 AI 回答的场景（如聊天界面）

    """
    print("\n" + "=" * 70)
    print("示例 3：stream_mode='messages'")
    print("=" * 70)

    agent = create_agent(
        get_default_model(),
        tools=[calculator]
    )

    print("\n问题：1 加 2 等于多少？")
    print("\n流式输出（messages 模式）：")
    print("-" * 70)

    chunk_count = 0
    current_step = None
    current_node = None
    full_content = ""

    # Token 统计
    tool_call_tokens = None  # 工具调用阶段的 token 统计
    final_answer_tokens = None  # 最终答案阶段的 token 统计
    for chunk in agent.stream(
        UserRequest(messages=[HumanMessage("1+2等于多少")]),
        stream_mode="messages"
    ):
        chunk_count += 1
        # messages 模式返回的是元组 (message_chunk, metadata)
        if isinstance(chunk, tuple) and len(chunk) == 2:
            message_chunk, metadata = chunk

            # 获取langgraph信息
            step = metadata.get("langgraph_step", 'N/A')
            node = metadata.get('langgraph_node', 'N/A')

            # 如果步骤或节点变化，显示提示
            if step != current_step or node != current_node:
                if current_step is not None:
                    print()  # 换行
                print(f"\n【步骤 {step} - 节点: {node}】")
                current_step = step
                current_node = node

            # 根据消息类型处理
            msg_type = message_chunk.__class__.__name__
            if msg_type == "AIMessageChunk":
                # 检查是否是工具调用
                if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                    print(f"  → AI 决定调用工具: {message_chunk.tool_calls[0]['name']}")
                    print(f"  → 工具参数: {message_chunk.tool_calls[0]['args']}")
                elif hasattr(message_chunk, 'content') and message_chunk.content:
                    # 实时打印内容（打字机效果）
                    print(message_chunk.content, end="", flush=True)
                    full_content += message_chunk.content

            elif msg_type == "ToolMessage":
                print(f"\n  → 工具执行结果: {message_chunk.content}")
                print(f"  → 工具名称: {message_chunk.name}")
                # ⚠️ 注意：ToolMessage 没有 token 统计（工具执行不消耗 LLM token）

            # 检查是否有 usage_metadata（token 统计），最后一个chunk内容为'',并且有usage_metadata字段
            if hasattr(message_chunk, "usage_metadata") and message_chunk.usage_metadata:
                usage = message_chunk.usage_metadata
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)

                # 判断是工具调用阶段还是最终答案阶段
                finish_reason = message_chunk.response_metadata.get('finish_reason', '') if hasattr(message_chunk,
                                                                                                    'response_metadata') else ''
                if finish_reason == 'tool_calls':
                    # 工具调用阶段的 token 统计
                    tool_call_tokens = usage
                    print(
                        f"  → [工具调用阶段结束] Token 使用: 输入={input_tokens}, 输出={output_tokens}, 总计={total_tokens}")
                else:
                    # 最终答案阶段的 token 统计
                    final_answer_tokens = usage
                    print(
                        f"  → [最终答案阶段] Token 使用: 输入={input_tokens}, 输出={output_tokens}, 总计={total_tokens}")

    print("\n\n" + "-" * 70)
    print(f"完整回答: {full_content}")
    print(f"总 chunk 数: {chunk_count}")

    # 显示 Token 统计总结
    print("\n" + "-" * 70)
    print("📊 Token 使用统计：")
    if tool_call_tokens:
        print(f"  工具调用阶段: {tool_call_tokens.get('total_tokens', 0)} tokens")
    if final_answer_tokens:
        print(f"  最终答案阶段: {final_answer_tokens.get('total_tokens', 0)} tokens")
        # ⚠️ 注意：最终答案阶段的 total_tokens 已经包含了完整的上下文（包括工具调用阶段）
        # 所以这是本次请求的总 token 消耗
        print(f"  ⚠️ 总 Token 消耗: {final_answer_tokens.get('total_tokens', 0)} tokens")
        print(f"    （最终答案阶段的 total_tokens 已包含完整上下文）")



if __name__ == '__main__':
    # example_1_stream_mode_values()
    # example_2_stream_mode_updates()
    example_3_stream_mode_messages()
    # TODO, Update、Values怎么获取token