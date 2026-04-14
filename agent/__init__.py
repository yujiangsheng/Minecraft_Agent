"""
agent — 智能体核心模块

包含决策引擎、规划器、反思系统、技能生成器和策略演化器，
构成完整的"感知 → 决策 → 执行 → 反思 → 进化"认知循环。

模块:
  core          AgentCore — 感知评估、模式切换、LLM 流水线决策
  planner       Planner — 优先级识别、技能选择、计划生成
  reflection    QuickReflection / LongReflection — 局部偏差检测 + 整局复盘
  skill_builder SkillBuilder — 从经验中自动发明可复用技能
  intent_understanding IntentUnderstanding — 自然语言意图分解 + 操作语义存储
  evolution     EvolutionOperator — 遗传算法多版本配置优化
"""

from agent.core import AgentCore, AgentState, AgentDecision
from agent.planner import Planner, Plan
from agent.reflection import QuickReflection, LongReflection, ReflectionResult
from agent.skill_builder import SkillBuilder
from agent.intent_understanding import IntentUnderstanding, IntentDecomposition, StructuredPrecondition, TaskExecutionPlan
from agent.evolution import EvolutionOperator, CandidateConfig

__all__ = [
    'AgentCore',
    'AgentState',
    'AgentDecision',
    'Planner',
    'Plan',
    'QuickReflection',
    'LongReflection',
    'ReflectionResult',
    'SkillBuilder',
    'IntentUnderstanding',
    'IntentDecomposition',
    'StructuredPrecondition',
    'TaskExecutionPlan',
    'EvolutionOperator',
    'CandidateConfig'
]
