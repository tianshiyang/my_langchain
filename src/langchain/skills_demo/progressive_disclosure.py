#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
方式二：渐进式披露模式 (Progressive Disclosure Skills)

这是 LangChain 推荐的技能加载方式，适合：
- 技能/知识库内容很大（如详细文档、操作手册）
- 需要按需加载，避免 context 溢出
- 技能数量多，不可能全部塞进 system prompt

核心思想：
1. 技能内容定义成结构化数据（Skill TypedDict）
2. 通过中间件将技能描述注入 system prompt（让 Agent 知道有哪些技能）
3. Agent 使用 load_skill 工具按需加载完整技能内容

官方文档：https://docs.langchain.com/oss/python/deepagents/overview/
"""

from typing import TypedDict, Callable
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from provider import get_default_model


# =============================================================================
# 1. 定义技能数据结构
# =============================================================================

class Skill(TypedDict):
    """
    技能定义结构

    - name: 技能唯一标识符
    - description: 1-2句话描述，用于系统提示（让Agent知道何时调用）
    - content: 完整技能内容，按需加载
    """
    name: str
    description: str
    content: str


# 技能库 - 存放所有可用技能
SKILLS: list[Skill] = [
    {
        "name": "code_review",
        "description": "代码审查指南，包括代码质量检查、性能优化建议、安全漏洞识别。",
        "content": """
# 代码审查技能

## 审查清单

### 代码质量
- [ ] 变量命名清晰
- [ ] 函数职责单一
- [ ] 避免重复代码
- [ ] 适当的注释

### 性能
- [ ] 避免 N+1 查询
- [ ] 合理使用缓存
- [ ] 循环优化

### 安全
- [ ] 防止 SQL 注入
- [ ] 输入验证
- [ ] 敏感信息不硬编码
- [ ] 权限检查

## 示例审查意见
"建议：将 `SELECT *` 改为具体字段，使用参数化查询避免 SQL 注入风险"
"""
    },
    {
        "name": "data_analysis",
        "description": "数据分析技能，包括数据清洗、统计计算、可视化建议。",
        "content": """
# 数据分析技能

## 数据清洗步骤
1. 缺失值处理：删除/填充/插值
2. 异常值检测：3σ原则、IQR方法
3. 数据类型转换
4. 重复记录删除

## 常用统计指标
- 均值、中位数、众数
- 标准差、方差
- 相关系数
- 百分位数

## 可视化建议
- 分布：直方图、箱线图
- 关系：散点图、热力图
- 趋势：折线图
- 构成：饼图、堆叠柱状图
"""
    },
    {
        "name": "writing_assistant",
        "description": "写作助手技能，包括文案撰写、邮件格式、技术文档规范。",
        "content": """
# 写作助手技能

## 文案撰写原则
1. 清晰简洁：删除冗余词汇
2. 结构化：使用标题、列表、表格
3. 读者导向：考虑读者背景和需求
4. 行动导向：明确告知读者下一步

## 邮件格式
- 主题：简洁明了，包含关键信息
- 称呼：根据关系选择正式/非正式
- 正文：结论先行，再提供细节
- 签名：包含联系方式

## 技术文档规范
- 使用主动语态
- 代码块标注语言
- 提供示例
- 更新日志记录变更
"""
    }
]


# =============================================================================
# 2. 创建技能加载工具
# =============================================================================

@tool
def load_skill(skill_name: str) -> str:
    """
    加载指定技能的完整内容到对话上下文。

    当你需要处理以下类型请求时使用此工具：
    - 代码审查相关
    - 数据分析相关
    - 写作/文档相关

    Args:
        skill_name: 技能名称，如 "code_review"、"data_analysis"

    Returns:
        技能的完整内容说明
    """
    for skill in SKILLS:
        if skill_name == skill["name"]:
            return f"✅ 已加载技能：{skill_name}\n\n{skill['content']}"

    available = ", ".join(s["name"] for s in SKILLS)
    return f"❌ 技能 '{skill_name}' 不存在。\n可用技能：{available}"


# =============================================================================
# 3. 创建技能中间件
# =============================================================================

class SkillsMiddleware(AgentMiddleware):
    """
    技能中间件：将技能描述注入系统提示

    作用：
    - 在系统提示中列出所有可用技能及其描述
    - 让 Agent 知道在什么情况下应该调用 load_skill
    - 不直接暴露完整技能内容，避免 context 溢出
    """

    tools = [load_skill]

    def __init__(self):
        # 生成技能列表提示
        skills_list = []
        for skill in SKILLS:
            skills_list.append(f"- **{skill['name']}**: {skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """
        将技能描述动态注入系统提示
        """
        skills_addendum = (
            "\n\n" + "=" * 50 + "\n"
            "## 可用技能\n\n"
            f"{self.skills_prompt}\n\n"
            "=" * 50 + "\n"
            "💡 提示：当需要处理特定领域的任务时，使用 load_skill 工具"
            "加载对应的技能指南。"
        )

        # 将技能描述追加到系统提示
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)

        return handler(modified_request)


# =============================================================================
# 4. 创建支持技能的 Agent
# =============================================================================

skills_agent = create_agent(
    model=get_default_model(),
    # 注意：不在这里直接传工具，而是通过中间件注入
    middleware=[SkillsMiddleware()],
    system_prompt=(
        "你是一个专业助手，可以帮助用户完成代码审查、数据分析、写作等任务。\n"
        "当你需要处理特定领域的专业任务时，可以加载对应的技能指南。"
    ),
    checkpointer=InMemorySaver(),
)


# =============================================================================
# 5. 使用 Agent
# =============================================================================

def demo_progressive_skills():
    """演示渐进式技能披露模式"""

    print("=" * 70)
    print("方式二：渐进式披露模式 (Progressive Disclosure)")
    print("=" * 70)

    print("""
    核心理念：
    - 不是把所有技能内容都塞进 system prompt
    - 而是先告诉 Agent 有哪些技能可用
    - Agent 根据需要主动加载完整技能内容
    """)

    # 示例1：请求代码审查指导
    print("\n【示例1】请求代码审查技能:")
    result = skills_agent.invoke({
        "messages": [HumanMessage(
            "我需要审查一段 Python 代码，帮我加载代码审查技能"
        )]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例2：数据分析任务
    print("\n【示例2】数据分析任务:")
    result = skills_agent.invoke({
        "messages": [HumanMessage(
            "我有一些销售数据需要分析，包括缺失值处理和异常值检测，"
            "请加载数据分析技能并给出建议"
        )]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()

    # 示例3：写作协助
    print("\n【示例3】写作协助:")
    result = skills_agent.invoke({
        "messages": [HumanMessage(
            "帮我写一封正式的商务邮件，主题是项目进度汇报"
        )]
    })
    for msg in result["messages"]:
        if hasattr(msg, "pretty_print"):
            msg.pretty_print()


if __name__ == "__main__":
    demo_progressive_skills()