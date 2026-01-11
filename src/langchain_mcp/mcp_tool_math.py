#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/10 18:58
@Author  : tianshiyang
@File    : langchain_mcp.py
"""
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两数之和"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """两数的乘积"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")