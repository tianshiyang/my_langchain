#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/10 19:03
@Author  : tianshiyang
@File    : mcp_tool_weather.py
"""
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """获取location的天气信息"""
    return "纽约总是阳光明媚"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=999,
    )