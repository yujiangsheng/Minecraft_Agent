"""
Evolution Operator - 策略演化器

作者: Jiangsheng Yu
许可证: MIT License

通过遗传算法优化智能体配置：
  - 种群管理：保持多个候选配置并行评估
  - 适应度评估：基于生存时间、资源收集、工具制作等指标
  - 精英选择 + 交叉变异：保留优质配置基因并探索新解
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import json
import copy
import random

from config import AgentConfig, DEFAULT_CONFIG
from utils import LLMClient, LLMResponse
from prompts import PromptTemplates


@dataclass
class CandidateConfig:
    """候选配置"""
    config_id: str
    generation: int
    
    # 可变配置
    planner_prompt_delta: str = ""
    reflection_prompt_delta: str = ""
    risk_thresholds: Dict[str, float] = field(default_factory=dict)
    memory_retrieval_weights: Dict[str, float] = field(default_factory=dict)
    skill_priority: Dict[str, float] = field(default_factory=dict)
    
    # 评测结果
    fitness_score: float = 0.0
    evaluation_episodes: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 来源
    parent_ids: List[str] = field(default_factory=list)
    mutation_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "generation": self.generation,
            "planner_prompt_delta": self.planner_prompt_delta,
            "reflection_prompt_delta": self.reflection_prompt_delta,
            "risk_thresholds": self.risk_thresholds,
            "memory_retrieval_weights": self.memory_retrieval_weights,
            "skill_priority": self.skill_priority,
            "fitness_score": self.fitness_score,
            "evaluation_episodes": self.evaluation_episodes,
            "metrics": self.metrics,
            "parent_ids": self.parent_ids,
            "mutation_applied": self.mutation_applied
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateConfig":
        return cls(
            config_id=data.get("config_id", ""),
            generation=data.get("generation", 0),
            planner_prompt_delta=data.get("planner_prompt_delta", ""),
            reflection_prompt_delta=data.get("reflection_prompt_delta", ""),
            risk_thresholds=data.get("risk_thresholds", {}),
            memory_retrieval_weights=data.get("memory_retrieval_weights", {}),
            skill_priority=data.get("skill_priority", {}),
            fitness_score=data.get("fitness_score", 0.0),
            evaluation_episodes=data.get("evaluation_episodes", 0),
            metrics=data.get("metrics", {}),
            parent_ids=data.get("parent_ids", []),
            mutation_applied=data.get("mutation_applied", [])
        )


@dataclass
class EvolutionInsight:
    """演化洞察"""
    version: str
    reason: str  # why_good 或 why_bad
    is_elite: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "reason": self.reason,
            "is_elite": self.is_elite
        }


class EvolutionOperator:
    """策略演化器"""
    
    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None):
        
        self.config = config or DEFAULT_CONFIG
        self.llm = llm_client
        
        # 种群
        self.population: List[CandidateConfig] = []
        self.generation: int = 0
        
        # 精英存档
        self.elites: List[CandidateConfig] = []
        
        # 演化历史
        self.evolution_history: List[Dict[str, Any]] = []
        
        # 配置
        self.population_size = self.config.evolution.population_size
        self.elite_count = self.config.evolution.elite_count
        self.mutation_rate = self.config.evolution.mutation_rate
    
    def _generate_config_id(self) -> str:
        """生成配置 ID"""
        import hashlib
        import time
        content = f"{self.generation}_{time.time()}_{random.random()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def initialize_population(self, base_config: AgentConfig = None) -> List[CandidateConfig]:
        """初始化种群"""
        base = base_config or self.config
        self.population = []
        
        for i in range(self.population_size):
            candidate = CandidateConfig(
                config_id=self._generate_config_id(),
                generation=0,
                risk_thresholds={
                    "health_critical": base.risk_thresholds.health_critical + random.uniform(-1, 1),
                    "health_low": base.risk_thresholds.health_low + random.uniform(-1, 1),
                    "hunger_critical": base.risk_thresholds.hunger_critical + random.uniform(-0.5, 0.5),
                    "hunger_low": base.risk_thresholds.hunger_low + random.uniform(-0.5, 0.5),
                },
                memory_retrieval_weights={
                    "episodic": 0.4 + random.uniform(-0.1, 0.1),
                    "semantic": 0.3 + random.uniform(-0.1, 0.1),
                    "skill": 0.3 + random.uniform(-0.1, 0.1)
                },
                skill_priority=dict(base.skill_priority)
            )
            
            # 归一化权重
            total = sum(candidate.memory_retrieval_weights.values())
            for k in candidate.memory_retrieval_weights:
                candidate.memory_retrieval_weights[k] /= total
            
            self.population.append(candidate)
        
        return self.population
    
    def evaluate_candidate(self,
                           candidate: CandidateConfig,
                           episode_results: List[Dict[str, Any]]) -> float:
        """评估候选配置"""
        if not episode_results:
            return 0.0
        
        # 计算适应度分数
        total_score = 0.0
        
        for episode in episode_results:
            episode_score = 0.0
            
            # 生存奖励
            if episode.get("survived", False):
                episode_score += 50
            
            # 步数奖励（活得越久越好）
            steps = episode.get("steps", 0)
            episode_score += min(steps * 0.5, 30)  # 最多30分
            
            # 资源收集奖励
            resources = episode.get("resources_collected", 0)
            episode_score += min(resources * 2, 20)  # 最多20分
            
            # 工具制作奖励
            tools_crafted = episode.get("tools_crafted", 0)
            episode_score += min(tools_crafted * 5, 15)  # 最多15分
            
            # 受伤惩罚
            damage_taken = episode.get("damage_taken", 0)
            episode_score -= damage_taken * 0.5
            
            # 死亡惩罚
            if episode.get("died", False):
                episode_score -= 20
            
            total_score += max(0, episode_score)
        
        # 平均分
        fitness = total_score / len(episode_results)
        
        # 更新候选配置
        candidate.fitness_score = fitness
        candidate.evaluation_episodes = len(episode_results)
        candidate.metrics = {
            "avg_steps": sum(e.get("steps", 0) for e in episode_results) / len(episode_results),
            "survival_rate": sum(1 for e in episode_results if e.get("survived", False)) / len(episode_results),
            "avg_resources": sum(e.get("resources_collected", 0) for e in episode_results) / len(episode_results)
        }
        
        return fitness
    
    def select_elites(self) -> List[CandidateConfig]:
        """选择精英"""
        # 按适应度排序
        sorted_pop = sorted(self.population, key=lambda c: c.fitness_score, reverse=True)
        
        # 选择前 N 个
        self.elites = sorted_pop[:self.elite_count]
        
        return self.elites
    
    def _mutate_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, float]:
        """变异阈值"""
        mutated = dict(thresholds)
        
        for key in mutated:
            if random.random() < self.mutation_rate:
                delta = random.uniform(-1, 1)
                mutated[key] = max(1, min(20, mutated[key] + delta))
        
        return mutated
    
    def _mutate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """变异权重"""
        mutated = dict(weights)
        
        for key in mutated:
            if random.random() < self.mutation_rate:
                delta = random.uniform(-0.1, 0.1)
                mutated[key] = max(0.1, min(0.8, mutated[key] + delta))
        
        # 归一化
        total = sum(mutated.values())
        for k in mutated:
            mutated[k] /= total
        
        return mutated
    
    def _mutate_priority(self, priority: Dict[str, float]) -> Dict[str, float]:
        """变异技能优先级"""
        mutated = dict(priority)
        
        for key in mutated:
            if random.random() < self.mutation_rate:
                delta = random.uniform(-0.1, 0.1)
                mutated[key] = max(0.1, min(1.0, mutated[key] + delta))
        
        return mutated
    
    def crossover(self, parent1: CandidateConfig, parent2: CandidateConfig) -> CandidateConfig:
        """交叉"""
        child = CandidateConfig(
            config_id=self._generate_config_id(),
            generation=self.generation + 1,
            parent_ids=[parent1.config_id, parent2.config_id]
        )
        
        # 阈值交叉
        child.risk_thresholds = {}
        for key in parent1.risk_thresholds:
            if random.random() < 0.5:
                child.risk_thresholds[key] = parent1.risk_thresholds.get(key, 5)
            else:
                child.risk_thresholds[key] = parent2.risk_thresholds.get(key, 5)
        
        # 权重交叉
        child.memory_retrieval_weights = {}
        for key in parent1.memory_retrieval_weights:
            if random.random() < 0.5:
                child.memory_retrieval_weights[key] = parent1.memory_retrieval_weights.get(key, 0.33)
            else:
                child.memory_retrieval_weights[key] = parent2.memory_retrieval_weights.get(key, 0.33)
        
        # 归一化权重
        total = sum(child.memory_retrieval_weights.values())
        for k in child.memory_retrieval_weights:
            child.memory_retrieval_weights[k] /= total
        
        # 优先级交叉
        all_keys = set(parent1.skill_priority.keys()) | set(parent2.skill_priority.keys())
        child.skill_priority = {}
        for key in all_keys:
            if random.random() < 0.5:
                child.skill_priority[key] = parent1.skill_priority.get(key, 0.5)
            else:
                child.skill_priority[key] = parent2.skill_priority.get(key, 0.5)
        
        return child
    
    def mutate(self, candidate: CandidateConfig) -> CandidateConfig:
        """变异"""
        mutated = copy.deepcopy(candidate)
        mutated.config_id = self._generate_config_id()
        mutated.generation = self.generation + 1
        mutated.parent_ids = [candidate.config_id]
        mutated.mutation_applied = []
        
        # 变异阈值
        if random.random() < self.mutation_rate:
            mutated.risk_thresholds = self._mutate_thresholds(candidate.risk_thresholds)
            mutated.mutation_applied.append("risk_thresholds")
        
        # 变异权重
        if random.random() < self.mutation_rate:
            mutated.memory_retrieval_weights = self._mutate_weights(candidate.memory_retrieval_weights)
            mutated.mutation_applied.append("memory_weights")
        
        # 变异优先级
        if random.random() < self.mutation_rate:
            mutated.skill_priority = self._mutate_priority(candidate.skill_priority)
            mutated.mutation_applied.append("skill_priority")
        
        return mutated
    
    def evolve(self) -> List[CandidateConfig]:
        """执行一代演化"""
        if not self.population:
            return self.initialize_population()
        
        # 选择精英
        elites = self.select_elites()
        
        new_population = list(elites)  # 精英直接进入下一代
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父代（锦标赛选择）
            candidates = random.sample(self.population, min(3, len(self.population)))
            parent1 = max(candidates, key=lambda c: c.fitness_score)
            
            candidates = random.sample(self.population, min(3, len(self.population)))
            parent2 = max(candidates, key=lambda c: c.fitness_score)
            
            # 交叉
            if random.random() < 0.7:
                child = self.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
                child.config_id = self._generate_config_id()
                child.generation = self.generation + 1
            
            # 变异
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # 记录演化历史
        self.evolution_history.append({
            "generation": self.generation,
            "best_fitness": max(c.fitness_score for c in self.population),
            "avg_fitness": sum(c.fitness_score for c in self.population) / len(self.population),
            "elite_ids": [e.config_id for e in elites]
        })
        
        return self.population
    
    def generate_child_with_llm(self,
                                elite_configs: List[CandidateConfig],
                                failed_configs: List[CandidateConfig]) -> Optional[CandidateConfig]:
        """使用 LLM 生成新的候选配置"""
        if not self.llm:
            return None
        
        system_prompt = PromptTemplates.evolution_prompt()
        
        context = {
            "elite_configs": [e.to_dict() for e in elite_configs],
            "failed_configs": [f.to_dict() for f in failed_configs],
            "current_generation": self.generation,
            "evolution_history": self.evolution_history[-5:] if self.evolution_history else []
        }
        
        response = self.llm.generate_decision(system_prompt, context)
        
        if response.success and response.parsed:
            child_data = response.parsed.get("child_candidate", {})
            
            child = CandidateConfig(
                config_id=self._generate_config_id(),
                generation=self.generation + 1,
                planner_prompt_delta=child_data.get("planner_delta", ""),
                reflection_prompt_delta=child_data.get("reflection_delta", ""),
                risk_thresholds=child_data.get("risk_thresholds", {}),
                memory_retrieval_weights=child_data.get("memory_retrieval_weights", {}),
                skill_priority={
                    change.get("skill", ""): change.get("new_priority", 0.5)
                    for change in child_data.get("skill_priority_changes", [])
                },
                parent_ids=[e.config_id for e in elite_configs[:2]],
                mutation_applied=["llm_generated"]
            )
            
            return child
        
        return None
    
    def get_best_config(self) -> Optional[CandidateConfig]:
        """获取最佳配置"""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.fitness_score)
    
    def apply_config(self, candidate: CandidateConfig, agent_config: AgentConfig) -> AgentConfig:
        """应用候选配置到智能体配置"""
        new_config = copy.deepcopy(agent_config)
        
        # 应用阈值
        if candidate.risk_thresholds:
            for key, value in candidate.risk_thresholds.items():
                if hasattr(new_config.risk_thresholds, key):
                    setattr(new_config.risk_thresholds, key, value)
        
        # 应用权重
        if candidate.memory_retrieval_weights:
            new_config.memory.retrieval_weights = dict(candidate.memory_retrieval_weights)
        
        # 应用优先级
        if candidate.skill_priority:
            new_config.skill_priority.update(candidate.skill_priority)
        
        return new_config
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取演化摘要"""
        return {
            "current_generation": self.generation,
            "population_size": len(self.population),
            "elite_count": len(self.elites),
            "best_fitness": max(c.fitness_score for c in self.population) if self.population else 0,
            "avg_fitness": sum(c.fitness_score for c in self.population) / len(self.population) if self.population else 0,
            "history_length": len(self.evolution_history)
        }
    
    def save(self, filepath: str):
        """保存演化状态"""
        data = {
            "generation": self.generation,
            "population": [c.to_dict() for c in self.population],
            "elites": [e.to_dict() for e in self.elites],
            "evolution_history": self.evolution_history
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载演化状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.generation = data.get("generation", 0)
        self.population = [CandidateConfig.from_dict(c) for c in data.get("population", [])]
        self.elites = [CandidateConfig.from_dict(e) for e in data.get("elites", [])]
        self.evolution_history = data.get("evolution_history", [])
