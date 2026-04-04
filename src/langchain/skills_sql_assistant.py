#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/8 16:19
@Author  : tianshiyang
@File    : skills_sql_assistant.py
"""
from typing import TypedDict, Callable

import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from provider import get_default_model


# 1. 定义技能
class Skill(TypedDict):
    """一种可以逐步向智能体披露的技能。"""
    name: str  # 该技能的唯一标识符
    description: str  # 在系统提示中显示的 1-2 句话描述
    content: str  # 包含详细说明的完整技能内容。


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "用于销售数据分析的数据库结构和业务逻辑，包括客户、订单和收入信息。",
        "content": """
            # 销售分析数据结构
            ## 数据表
            ### customers（客户表）
            - customer_id（主键）
            - name（姓名）
            - email（邮箱）
            - signup_date（注册日期）
            - status（状态：active/active 表示活跃，inactive 表示非活跃）
            - customer_tier（客户等级：bronze/银/金/platinum）
            
            ### orders（订单表）
            - order_id（主键）
            - customer_id（外键，关联 customers 表）
            - order_date（下单日期）
            - status（状态：pending/待处理、completed/已完成、cancelled/已取消、refunded/已退款）
            - total_amount（订单总金额）
            - sales_region（销售区域：north/北区、south/南区、east/东区、west/西区）
            
            ### order_items（订单明细）
            - item_id（主键）
            - order_id（外键，关联 orders 表）
            - product_id（产品 ID）
            - quantity（数量）
            - unit_price（单价）
            - discount_percent（折扣百分比）
            ## 业务逻辑
            
            活跃客户：status = 'active' 且 signup_date ≤ 当前日期 - 90 天
            
            收入计算：仅统计 status = 'completed' 的订单。orders 表中的 total_amount 字段已包含折扣后的金额。
            
            客户终身价值（CLV）：某客户所有已完成订单金额的总和。
            
            高价值订单：total_amount > 1000 的订单
            
            示例查询
            -- 获取最近一个季度收入排名前 10 的客户
                SELECT
                c.customer_id,
                c.name,
                c.customer_tier,
                SUM(o.total_amount) as total_revenue
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
                GROUP BY c.customer_id, c.name, c.customer_tier
                ORDER BY total_revenue DESC
                LIMIT 10;
            """
    },
    {
        "name": "inventory_management",
        "description": "用于库存跟踪的数据库结构和业务逻辑，包括产品、仓库和库存水平。",
        "content": """
            # 库存管理数据结构
            ## 数据表
            ### products（产品表）
            - product_id（主键）
            - product_name（产品名称）
            - sku（库存单位编码）
            - category（类别）
            - unit_cost（单位成本）
            - reorder_point（再订货点，即触发补货的最低库存水平）
            - discontinued（是否已停产，布尔值）
            
            ### warehouses（仓库表）
            - warehouse_id（主键）
            - warehouse_name（仓库名称）
            - location（位置）
            - capacity（容量）
            
            ### inventory（库存表）
            - inventory_id（主键）
            - product_id（外键，关联 products 表）
            - warehouse_id（外键，关联 warehouses 表）
            - quantity_on_hand（当前库存数量）
            - last_updated（最后更新时间）
            
            ### stock_movements（库存变动记录表）
            - movement_id（主键）
            - product_id（外键，关联 products 表）
            - warehouse_id（外键，关联 warehouses 表）
            - movement_type（变动类型：inbound/入库、outbound/出库、transfer/调拨、adjustment/调整）
            - quantity（数量：入库为正，出库为负）
            - movement_date（变动日期）
            - reference_number（参考单号）
            
            业务逻辑
            
            可用库存：inventory 表中 quantity_on_hand > 0 的记录
            
            需补货的产品：所有仓库中该产品的总库存量 ≤ 其 reorder_point 的产品
            
            仅限活跃产品：除非特别分析停产商品，否则应排除 discontinued = true 的产品
            
            库存估值：每个产品的 quantity_on_hand × unit_cost
            
            示例查询
            -- 查找所有仓库中库存低于再订货点的产品
                SELECT
                p.product_id,
                p.product_name,
                p.reorder_point,
                SUM(i.quantity_on_hand) as total_stock,
                p.unit_cost,
                (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
                FROM products p
                JOIN inventory i ON p.product_id = i.product_id
                WHERE p.discontinued = false
                GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
                HAVING SUM(i.quantity_on_hand) <= p.reorder_point
                ORDER BY units_to_reorder DESC;
            """
    }
]

# 2. 创建技能加载工具
@tool
def load_skill(skill_name: str) -> str:
    """将某项技能的完整内容加载到智能体的上下文中。
    当你需要详细了解如何处理某一特定类型的请求时，请使用此功能。
    这将为你提供该技能领域的全面说明、政策和操作指南。

    参数：
    skill_name：要加载的技能名称（例如："expense_reporting"、"travel_booking"）"""
    # 查找和返回技能
    for skill in SKILLS:
        if skill_name == skill["name"]:
            return f"加载技能：{skill_name}\n\n{skill['content']}"
    # 技能找不到
    available = ", ".join(s["name"] for s in SKILLS)
    return f"技能 '{skill_name}' 找不到. 可用的技能: {available}"

# 3.构建技能中间件
class SkillsMiddleware(AgentMiddleware):
    """一种中间件，用于将技能描述注入到系统提示（system prompt）中。"""
    tools = [load_skill]

    def __init__(self):
        """从 SKILLS 初始化并生成技能提示（skills prompt）。"""
        skills_list = []
        for skill in SKILLS:
            skills_list.append(f"- **{skill['name']}**: {skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """同步：将技能描述注入到系统提示中。"""
        skills_addendum = (f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n" +
                "当您需要详细了解如何处理某一特定类型的请求时，请使用 load_skill 工具。")
        print(f"request.system_message.content_blocks: {request.system_message.content_blocks}")
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

# 4.创建具有技能支持的agent
agent = create_agent(
    get_default_model(),
    system_prompt="你是一个 SQL 查询助手，帮助用户编写针对业务数据库的查询语句。",
    checkpointer=InMemorySaver(),
    middleware=[SkillsMiddleware()]
)

# 5.测试渐进式披露
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "写一个sql去查找所有的用户 "
                    "谁在上个月下了超过 1000 美元的订单？"
                ),
            }
        ]
    },
    config
)

for message in result["messages"]:
    if hasattr(message, "pretty_print"):
        message.pretty_print()
    else:
        print(f"{message.type}: {message.content}")