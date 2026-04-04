#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 1.0 技能(Skill)加载和使用方式演示

本文档展示 LangChain 中推荐的三种技能加载方式：

1. 直接工具模式 (Direct Tools)       - 简单直接，适合工具数量较少
2. 渐进式披露模式 (Progressive)       - 技能按需加载，适合大量技能场景
3. 子Agent协作模式 (Sub-Agent)        - 复杂任务分解，适合多领域协作

参考官方文档：
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/oss/python/deepagents/overview/
"""

from direct_tools import demo_direct_tools
from progressive_disclosure import demo_progressive_skills
from sub_agent import demo_sub_agent

__all__ = [
    "demo_direct_tools",
    "demo_progressive_skills",
    "demo_sub_agent",
]
