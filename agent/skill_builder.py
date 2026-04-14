"""
Skill Builder - 技能自动生成器

作者: Jiangsheng Yu
许可证: MIT License

根据经验自动生成可复用技能：
  - 分析失败模式，发明预防技能
  - 分析成功模式，提取可复用技能
  - 每个技能包含触发条件、执行步骤和失败恢复策略
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json

from config import AgentConfig, DEFAULT_CONFIG
from memory import MemoryManager, Skill
from utils import LLMClient, LLMResponse
from prompts import PromptTemplates


@dataclass
class SkillCandidate:
    """技能候选"""
    skill_name: str
    purpose: str
    trigger_conditions: List[str]
    preconditions: List[str]
    steps: List[Dict[str, Any]]
    stop_conditions: List[str]
    failure_recovery: List[Dict[str, Any]]
    metrics: Dict[str, str]
    source_episodes: List[str] = field(default_factory=list)  # 来源的情景记忆 ID
    confidence: float = 0.5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCandidate":
        return cls(
            skill_name=data.get("skill_name", ""),
            purpose=data.get("purpose", ""),
            trigger_conditions=data.get("trigger_conditions", []),
            preconditions=data.get("preconditions", []),
            steps=data.get("steps", []),
            stop_conditions=data.get("stop_conditions", []),
            failure_recovery=data.get("failure_recovery", []),
            metrics=data.get("metrics", {}),
            source_episodes=data.get("source_episodes", []),
            confidence=data.get("confidence", 0.5)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "purpose": self.purpose,
            "trigger_conditions": self.trigger_conditions,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "stop_conditions": self.stop_conditions,
            "failure_recovery": self.failure_recovery,
            "metrics": self.metrics,
            "source_episodes": self.source_episodes,
            "confidence": self.confidence
        }
    
    def to_skill(self) -> Dict[str, Any]:
        """转换为 Skill 格式"""
        return {
            "name": self.skill_name,
            "purpose": self.purpose,
            "trigger_conditions": self.trigger_conditions,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "stop_conditions": self.stop_conditions,
            "failure_recovery": self.failure_recovery,
            "metrics": self.metrics,
            "priority": self.confidence
        }


class SkillBuilder:
    """技能生成器"""
    
    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        
        self.config = config or DEFAULT_CONFIG
        self.llm = llm_client
        self.memory = memory_manager
        
        # 技能候选列表
        self.candidates: List[SkillCandidate] = []
        
        # 生成的技能
        self.generated_skills: List[SkillCandidate] = []
        
        # 高频问题统计
        self.problem_frequency: Dict[str, int] = {}
    
    def _analyze_failure_patterns(self, 
                                   failure_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析失败模式"""
        patterns = []
        
        # 按 tags 分组
        tag_groups: Dict[str, List[Dict[str, Any]]] = {}
        for episode in failure_episodes:
            for tag in episode.get("tags", []):
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(episode)
        
        # 找出高频失败模式
        for tag, episodes in tag_groups.items():
            if len(episodes) >= 2:  # 至少出现2次
                patterns.append({
                    "tag": tag,
                    "count": len(episodes),
                    "episodes": episodes,
                    "lessons": [e.get("lesson", "") for e in episodes]
                })
        
        # 按频率排序
        patterns.sort(key=lambda x: x["count"], reverse=True)
        
        return patterns
    
    def _analyze_success_patterns(self,
                                   success_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析成功模式"""
        patterns = []
        
        # 按 tags 分组
        tag_groups: Dict[str, List[Dict[str, Any]]] = {}
        for episode in success_episodes:
            for tag in episode.get("tags", []):
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(episode)
        
        # 找出可复用的成功模式
        for tag, episodes in tag_groups.items():
            if len(episodes) >= 2:
                patterns.append({
                    "tag": tag,
                    "count": len(episodes),
                    "episodes": episodes,
                    "strategies": [e.get("summary", "") for e in episodes]
                })
        
        patterns.sort(key=lambda x: x["count"], reverse=True)
        
        return patterns
    
    def identify_skill_opportunities(self) -> List[Dict[str, Any]]:
        """识别技能生成机会"""
        opportunities = []
        
        if not self.memory:
            return opportunities
        
        # 获取失败和成功的情景记忆
        all_episodes = self.memory.episodic.to_prompt_format()
        failure_episodes = [e for e in all_episodes if e.get("outcome") == "failure"]
        success_episodes = [e for e in all_episodes if e.get("outcome") == "success"]
        
        # 分析失败模式
        failure_patterns = self._analyze_failure_patterns(failure_episodes)
        for pattern in failure_patterns[:3]:  # 取前3个高频失败模式
            opportunities.append({
                "type": "failure_prevention",
                "problem": pattern["tag"],
                "frequency": pattern["count"],
                "evidence": pattern["lessons"][:3],
                "suggested_skill_type": f"avoid_{pattern['tag']}"
            })
        
        # 分析成功模式
        success_patterns = self._analyze_success_patterns(success_episodes)
        for pattern in success_patterns[:3]:
            opportunities.append({
                "type": "success_replication",
                "pattern": pattern["tag"],
                "frequency": pattern["count"],
                "evidence": pattern["strategies"][:3],
                "suggested_skill_type": f"replicate_{pattern['tag']}"
            })
        
        return opportunities
    
    def generate_skill(self,
                       problem: str = None,
                       failure_cases: List[Dict[str, Any]] = None,
                       success_cases: List[Dict[str, Any]] = None) -> Optional[SkillCandidate]:
        """生成新技能"""
        
        if not self.llm:
            return self._generate_default_skill(problem)
        
        system_prompt = PromptTemplates.skill_builder_prompt()
        
        context = {
            "problem_to_solve": problem or "通用问题处理",
            "failure_cases": failure_cases or [],
            "success_cases": success_cases or [],
            "existing_skills": self.memory.skills.to_prompt_format() if self.memory else [],
            "requirements": [
                "技能必须有明确触发条件",
                "技能必须解决一个高频问题",
                "技能步骤必须短小、稳定、可执行",
                "必须包含失败恢复策略",
                "不要发明过于宽泛的技能"
            ]
        }
        
        response = self.llm.generate_decision(system_prompt, context)
        
        if response.success and response.parsed:
            candidate = SkillCandidate.from_dict(response.parsed)
            
            # 验证技能
            if self._validate_skill(candidate):
                self.candidates.append(candidate)
                return candidate
        
        return self._generate_default_skill(problem)
    
    def _generate_default_skill(self, problem: str) -> Optional[SkillCandidate]:
        """生成默认技能"""
        if not problem:
            return None
        
        # 根据问题类型生成基础技能
        skill_templates = {
            "hunger": SkillCandidate(
                skill_name="quick_food_gathering",
                purpose="快速获取食物解决饥饿问题",
                trigger_conditions=["hunger_low", "hunger_critical"],
                preconditions=[],
                steps=[
                    {"action": "find_food_source", "args": {}},
                    {"action": "gather_or_hunt", "args": {}},
                    {"action": "eat", "args": {}}
                ],
                stop_conditions=["hunger_satisfied", "no_food_available"],
                failure_recovery=[
                    {"condition": "no_food_source", "action": "explore", "args": {"radius": 50}}
                ],
                metrics={"success_signal": "hunger_restored", "risk_signal": "health_low"}
            ),
            "danger": SkillCandidate(
                skill_name="danger_avoidance",
                purpose="遇到危险时快速脱离",
                trigger_conditions=["danger_detected", "hostile_nearby"],
                preconditions=[],
                steps=[
                    {"action": "identify_threat", "args": {}},
                    {"action": "find_escape_route", "args": {}},
                    {"action": "run", "args": {"direction": "away"}}
                ],
                stop_conditions=["safe", "threat_lost"],
                failure_recovery=[
                    {"condition": "cornered", "action": "fight", "args": {}}
                ],
                metrics={"success_signal": "escaped", "risk_signal": "health_decreasing"}
            )
        }
        
        # 查找匹配的模板
        for key, template in skill_templates.items():
            if key in problem.lower():
                self.candidates.append(template)
                return template
        
        return None
    
    def _validate_skill(self, candidate: SkillCandidate) -> bool:
        """验证技能是否有效"""
        # 基本验证
        if not candidate.skill_name:
            return False
        
        if not candidate.trigger_conditions:
            return False
        
        if not candidate.steps or len(candidate.steps) == 0:
            return False
        
        if len(candidate.steps) > 10:
            return False  # 步骤过多
        
        # 检查是否与现有技能重复
        if self.memory:
            existing = self.memory.skills.get_by_name(candidate.skill_name)
            if existing:
                return False  # 已存在同名技能
        
        return True
    
    def commit_skill(self, candidate: SkillCandidate) -> Optional[Skill]:
        """将候选技能提交到技能库"""
        if not self.memory:
            return None
        
        if not self._validate_skill(candidate):
            return None
        
        skill = self.memory.store_skill(
            name=candidate.skill_name,
            purpose=candidate.purpose,
            trigger_conditions=candidate.trigger_conditions,
            preconditions=candidate.preconditions,
            steps=candidate.steps,
            stop_conditions=candidate.stop_conditions,
            failure_recovery=candidate.failure_recovery,
            metrics=candidate.metrics,
            priority=candidate.confidence
        )
        
        self.generated_skills.append(candidate)
        
        return skill
    
    def auto_generate_skills(self, min_frequency: int = 2) -> List[SkillCandidate]:
        """自动生成技能"""
        generated = []
        
        opportunities = self.identify_skill_opportunities()
        
        for opportunity in opportunities:
            if opportunity.get("frequency", 0) >= min_frequency:
                problem = opportunity.get("problem") or opportunity.get("pattern", "")
                
                # 获取相关案例
                failure_cases = []
                success_cases = []
                
                if self.memory:
                    all_episodes = self.memory.episodic.to_prompt_format()
                    for episode in all_episodes:
                        if problem in episode.get("tags", []):
                            if episode.get("outcome") == "failure":
                                failure_cases.append(episode)
                            elif episode.get("outcome") == "success":
                                success_cases.append(episode)
                
                candidate = self.generate_skill(
                    problem=problem,
                    failure_cases=failure_cases[:5],
                    success_cases=success_cases[:5]
                )
                
                if candidate:
                    generated.append(candidate)
        
        return generated
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_candidates": len(self.candidates),
            "total_generated": len(self.generated_skills),
            "problem_frequency": self.problem_frequency
        }
