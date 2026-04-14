"""
Minecraft/Luanti 智能体 - 配置模块

作者: Jiangsheng Yu
许可证: MIT License

定义智能体的所有可调参数，包括风险阈值、记忆配置、LLM 配置、
规划器配置、反思配置和演化配置。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class RiskThresholds:
    """风险阈值配置"""
    health_critical: int = 5      # 生命值危急线
    health_low: int = 8           # 生命值低线
    hunger_critical: int = 3      # 饥饿值危急线
    hunger_low: int = 6           # 饥饿值低线
    combat_avoid_health: int = 8  # 避免战斗的生命值阈值
    night_danger_level: float = 0.7  # 夜晚危险等级


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    episodic_max_size: int = 100        # 情景记忆最大数量
    semantic_max_rules: int = 50        # 语义规则最大数量
    skills_max_count: int = 30          # 技能最大数量
    trajectory_window: int = 20         # 轨迹窗口大小
    retrieval_top_k: int = 5            # 检索返回数量
    similarity_threshold: float = 0.6   # 相似度阈值
    
    # 检索权重
    retrieval_weights: Dict[str, float] = field(default_factory=lambda: {
        "episodic": 0.4,
        "semantic": 0.3,
        "skills": 0.3
    })


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "local"                          # 默认使用本地 Ollama
    model: str = "qwen2.5:7b-instruct"              # 默认模型：通义千问2.5 7B（轻量高效）
    api_key: str = ""                                # API Key（本地模型无需）
    api_base: str = "http://localhost:11434/v1"      # Ollama OpenAI 兼容端点
    temperature: float = 0.5                         # 生成温度（适当提高以鼓励动作多样性）
    max_tokens: int = 1024                           # 最大 token（缩短以加快响应）
    timeout: int = 60                                # 超时时间（加速决策循环）


@dataclass
class PlannerConfig:
    """规划器配置"""
    max_actions_per_plan: int = 5     # 单次计划最大动作数
    replan_on_deviation: bool = True  # 偏差时重新规划
    conservative_mode_triggers: List[str] = field(default_factory=lambda: [
        "night",
        "low_health",
        "low_hunger",
        "hostile_nearby",
        "no_weapon"
    ])


@dataclass
class ReflectionConfig:
    """反思模块配置"""
    quick_reflection_enabled: bool = True     # 启用快速反思
    long_reflection_enabled: bool = True      # 启用长期复盘
    auto_store_memory: bool = True            # 自动存储记忆
    failure_threshold: float = 0.3            # 失败阈值
    reflection_interval: int = 10             # 反思间隔（动作数）


@dataclass
class EvolutionConfig:
    """演化配置"""
    enabled: bool = True                  # 启用演化
    population_size: int = 5              # 种群大小
    elite_count: int = 2                  # 精英数量
    mutation_rate: float = 0.2            # 变异率
    evolution_interval: int = 10          # 演化间隔（episode 数）


@dataclass
class AgentConfig:
    """智能体总配置"""
    name: str = "MinecraftAgent"
    version: str = "1.0.0"
    
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    
    # 技能优先级
    skill_priority: Dict[str, float] = field(default_factory=lambda: {
        "shelter_building": 0.9,
        "food_gathering": 0.85,
        "tool_crafting": 0.8,
        "resource_mining": 0.7,
        "exploration": 0.5,
        "combat": 0.4
    })
    
    # 模式优先级
    mode_priority: Dict[str, int] = field(default_factory=lambda: {
        "survive": 100,
        "retreat": 95,
        "gather": 70,
        "craft": 65,
        "build": 60,
        "explore": 50,
        "combat": 30
    })


# 默认配置实例
DEFAULT_CONFIG = AgentConfig()
