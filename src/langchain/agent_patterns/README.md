Agent 推理模式学习教程

背景

你的项目中已有大量 Agent 相关代码（create_agent tool-calling 模式、LangGraph 工作流、手写 Agent Loop 等），但尚未系统性地覆盖经典推理模式。本计划将在 src/langchain/agent_patterns/ 下创建 6 个独立的学习示例。



六种模式概览

flowchart LR
    subgraph simple [基础模式]
        CoT["Chain-of-Thought\n逐步推理"]
        ReAct["ReAct\n推理+行动循环"]
    end
    subgraph intermediate [进阶模式]
        SelfAsk["Self-Ask\n自问自答+验证"]
        PlanResolve["Plan-and-Resolve\n先规划后执行"]
    end
    subgraph advanced [高级模式]
        ToT["Tree of Thoughts\n多路径探索"]
        Reflection["Reflection\n生成+反思迭代"]
    end
    CoT --> ReAct
    ReAct --> SelfAsk
    SelfAsk --> PlanResolve
    PlanResolve --> ToT
    ToT --> Reflection



模式 1: Chain-of-Thought (CoT) -- 思维链

核心思想: 通过 prompt 引导 LLM 输出逐步推理过程，而非直接给出答案。不涉及工具调用，纯推理增强。

适用场景: 数学推理、逻辑分析、复杂问答

实现方式: 仅通过 prompt engineering，在 system message 中加入 "Let's think step by step" 类指令。

代码要点:





直接使用 get_default_model() 进行 LLM 调用



对比有/无 CoT prompt 的输出差异



展示 Zero-shot CoT 和 Few-shot CoT 两种变体

# Zero-shot CoT: 只需加一句 "请一步步思考"
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑推理专家。请一步步思考，展示完整推理过程，最后给出答案。"),
    ("human", "{question}")
])
chain = cot_prompt | get_default_model()



模式 2: ReAct (Reason + Act) -- 推理与行动

核心思想: Thought -> Action -> Observation 循环。LLM 先推理要做什么，执行工具，观察结果，再决定下一步。

适用场景: 需要与外部工具交互的探索性任务（搜索、查数据库、调 API）

项目中已有实现:





custom_agent_langgraph.py -- LangGraph 版本，使用 ToolNode + tools_condition



agent_loop_demo.py -- 手写版本，规则驱动的 Planner

新示例将:





用 LangGraph StateGraph 构建标准 ReAct 循环



使用真实 LLM（get_default_model）+ bind_tools



展示完整的 Thought/Action/Observation 轨迹

flowchart TD
    Start([用户提问]) --> Think[Thought: LLM 推理]
    Think --> Decide{需要工具?}
    Decide -->|是| Act[Action: 调用工具]
    Act --> Observe[Observation: 工具返回]
    Observe --> Think
    Decide -->|否| Answer([最终回答])

关键代码结构:

# LangGraph ReAct 核心：条件边决定是继续调工具还是结束
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")



模式 3: Self-Ask -- 自问自答

核心思想: 面对复杂问题时，LLM 自动将其分解为子问题，逐个回答子问题后综合得出最终答案。每个中间答案可通过搜索工具验证。

适用场景: 多跳推理（如 "X 的创始人的出生地在哪个国家？"），需要多步信息检索的场景

与 ReAct 的区别: Self-Ask 更注重问题分解和事实验证，ReAct 更通用

实现方式:





使用 StateGraph 管理子问题列表



每轮 LLM 生成下一个子问题或判断已可回答



子问题通过搜索工具获取答案



最终综合所有中间答案

class SelfAskState(TypedDict):
    question: str
    sub_questions: Annotated[list[str], operator.add]
    intermediate_answers: Annotated[list[str], operator.add]
    final_answer: str



模式 4: Plan-and-Resolve -- 先规划后执行

核心思想: 与 ReAct "走一步看一步"不同，先让 LLM 制定完整执行计划，然后按计划逐步执行。执行中可以 Replan。

适用场景: 步骤明确的多步任务（写报告、数据分析流水线、多步操作）

已有参考: langchain_customer_support.py 中 supervisor 的任务分派有类似思路

实现方式:





Planner 节点：生成任务步骤列表



Executor 节点：按顺序执行每个步骤



Replanner 节点：根据已完成步骤决定是否需要调整计划



三者通过 LangGraph 条件边连接

flowchart TD
    Start([输入任务]) --> Plan[Planner: 制定计划]
    Plan --> Execute[Executor: 执行当前步骤]
    Execute --> Replan{Replanner: 是否需要调整?}
    Replan -->|需要调整| Plan
    Replan -->|继续执行| Execute
    Replan -->|全部完成| Answer([最终结果])

关键代码结构:

class PlanState(TypedDict):
    task: str
    plan: list[str]           # 计划步骤
    current_step: int         # 当前执行到哪一步
    step_results: Annotated[list[str], operator.add]
    final_answer: str



模式 5: Tree of Thoughts (ToT) -- 思维树

核心思想: 生成多个候选思路/方案，对每个方案独立评估打分，选择最优路径继续深入。类似于搜索树的广度优先+剪枝。

适用场景: 创意写作、方案比选、需要探索多种可能性的决策问题

与 CoT 的区别: CoT 是单条推理链，ToT 是多条并行推理后择优

实现方式:





Generator 节点：对同一问题生成 N 个候选方案



Evaluator 节点：用结构化输出对每个方案打分



Selector 节点：选出最佳方案



可选递归深入最佳方案

class ThoughtCandidate(BaseModel):
    thought: str = Field(description="候选思路")
    reasoning: str = Field(description="推理过程")

class Evaluation(BaseModel):
    scores: dict[str, float] = Field(description="各维度评分")
    best_index: int = Field(description="最佳方案索引")
    justification: str = Field(description="选择理由")



模式 6: Reflection -- 反思迭代

核心思想: Generator 生成初始输出，Reflector 对其进行批判和改进建议，Generator 据此修改，循环迭代直到质量满意。

适用场景: 代码生成/审查、文章写作/润色、任何需要质量把关的生成任务

已有参考: custom_agent_langgraph.py 中的 grade_documents + rewrite_question 路径本质上是一种简化的 Reflection

实现方式:





Generator 节点：生成/改进内容



Reflector 节点：提供批评和改进建议



条件边：达到最大迭代次数或质量满意时退出

flowchart TD
    Start([输入任务]) --> Gen[Generator: 生成内容]
    Gen --> Reflect[Reflector: 批评与建议]
    Reflect --> Check{质量满足 or 达到上限?}
    Check -->|否| Gen
    Check -->|是| Answer([最终输出])



文件结构

在 src/langchain/agent_patterns/ 下创建：

src/langchain/agent_patterns/
  __init__.py
  chain_of_thought.py      # 模式 1: CoT
  react_agent.py            # 模式 2: ReAct
  self_ask.py               # 模式 3: Self-Ask
  plan_and_resolve.py       # 模式 4: Plan-and-Resolve
  tree_of_thoughts.py       # 模式 5: ToT
  reflection.py             # 模式 6: Reflection

所有文件遵循项目现有风格：





文件头 @Author / @Time 注释



中文注释



使用 provider 模块的 get_default_model() / chatGptLLM



每个文件包含独立的 if __name__ == "__main__" 入口



可通过 python -m src.langchain.agent_patterns.xxx 运行

每个示例的内容结构





模块顶部注释: 解释该模式的核心思想、与其他模式的区别



State 定义: 使用 TypedDict 或 dataclass 定义状态



核心节点/函数: 各个处理步骤



Graph 编排: 使用 LangGraph StateGraph 串联（CoT 除外，它不需要 Graph）



main 入口: 含 1-2 个测试用例，打印完整执行轨迹

