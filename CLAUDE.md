# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 LangChain/LangGraph 学习项目，包含多种 AI 应用示例：RAG、Agent、多智能体系统、向量检索等。

## 环境配置

- Python 版本：`.python-version` (3.13)
- 依赖管理：`.venv` 虚拟环境
- 安装依赖：`pip install -r requirements.txt`
- 运行入口：`main.py`

## 常用命令

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行客户支持多智能体示例
python -m src.customer_support_multi_agent.langchain_customer_support
python -m src.customer_support_multi_agent.langgraph_customer_support
python -m src.customer_support_multi_agent.langchain_agent_loop_multi_agent
python -m src.customer_support_multi_agent.langgraph_agent_loop_multi_agent

# 运行 LangChain 示例
python -m src.langchain.quick_start

# 运行 Agent 推理模式示例
python -m src.langchain.agent_patterns.chain_of_thought
python -m src.langchain.agent_patterns.react_agent
python -m src.langchain.agent_patterns.self_ask
python -m src.langchain.agent_patterns.plan_and_resolve
python -m src.langchain.agent_patterns.tree_of_thoughts
python -m src.langchain.agent_patterns.reflection

# 运行 Milvus RAG 示例
python -m src.milvus.rag
```

## 项目结构

```
src/
├── langchain/          # LangChain 核心功能演示 (agent, tools, RAG, memory, streaming 等)
│   └── agent_patterns/ # Agent 推理模式学习 (CoT, ReAct, Self-Ask, Plan&Resolve, ToT, Reflection)
├── langgraph_study/    # LangGraph 工作流编排学习
├── milvus/             # Milvus 向量数据库集成 (RAG, hybrid search, SQL agent)
├── customer_support_multi_agent/  # 客户支持多智能体系统（两套实现：LangChain/LangGraph）
├── provider/           # LLM 提供者配置 (OpenAI, Qwen, Gemini, MiniMax)
├── utils/              # 工具模块 (embeddings 等)
├── langchain_mcp/      # MCP (Model Context Protocol) 工具示例
├── langgraph_study/    # LangGraph 自定义 agent 示例
└── docs/               # 项目文档

tests/
└── test_customer_support_advanced.py  # 高级客户支持测试
```

## 核心依赖

- `langchain>=0.3.0`, `langchain-core>=0.3.0`
- `langgraph>=0.2.0`
- `langchain-openai`, `langchain-google-genai`, `langchain-qwq`
- `langchain-milvus`
- `pydantic>=2.0.0`
- `python-dotenv>=1.0.0`

## 环境变量配置 (.env)

项目使用多个 LLM 提供者：
- `XIAO_AI_BASE_URL` / `XIAO_AI_API_KEY` - 小爱 AI
- `QWEN_BASE_URL` / `QWEN_API_KEY` - 阿里千问
- `GOOGLE_API_KEY` - Google Gemini
- `DASHSCOPE_API_KEY` - 阿里 DashScope (embedding)
- `MINIMAX_BASE_URL` / `MINIMAX_API_KEY` - MiniMax
- `MILVUS_URI` / `MILVUS_TOKEN` - Milvus 向量数据库连接

## 关键模块说明

### provider/llms.py
统一封装多个 LLM：`chatGptLLM`、`qwenLLM`、`google_gemini`、`minimax_llm`，通过 `get_default_model()` 获取默认模型。

### milvus/
- `client.py` - Milvus 客户端配置
- `rag.py` - RAG (Retrieval-Augmented Generation) 实现
- `hybrid_search.py` - 混合搜索
- `sql_agent_langchain.py` / `sql_agent_postgres.py` - SQL Agent

### langchain/agent_patterns/
六种经典 Agent 推理模式的独立学习示例，均使用 LangGraph StateGraph 编排（CoT 除外）：
- `chain_of_thought.py` - 思维链（Zero-shot / Few-shot CoT 对比）
- `react_agent.py` - ReAct 推理+行动循环（bind_tools + ToolNode）
- `self_ask.py` - 自问自答（问题分解 + 搜索验证）
- `plan_and_resolve.py` - 先规划后执行（Planner/Executor/Replanner）
- `tree_of_thoughts.py` - 思维树（多候选生成 + 评估 + 择优）
- `reflection.py` - 反思迭代（Generator/Reflector 循环改进）

### customer_support_multi_agent/
两套客服多智能体实现，均包含 router、supervisor、specialists、退款流程：
- LangChain 版本使用高层 agent 模式
- LangGraph 版本使用显式工作流和子图

## 编码规范

- 文件头注释包含 `@Author` 和 `@Time`
- 中文注释
- 模块级导入使用相对导入（如 `from milvus import CONNECTION_ARGS`）
