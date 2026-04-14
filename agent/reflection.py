"""
Reflection - 反思模块

作者: Jiangsheng Yu
许可证: MIT License

包含两层反思机制：
  - QuickReflection: 局部反思，每次动作执行后触发，识别偏差类型并提出即时修正
  - LongReflection:  整局复盘，每局结束后触发，总结成功/失败模式并更新规则和技能
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json

from config import AgentConfig, DEFAULT_CONFIG
from memory import MemoryManager
from utils import LLMClient, LLMResponse
from prompts import PromptTemplates


class ReflectionStatus(Enum):
    """反思状态"""
    OK = "ok"
    DEVIATION = "deviation"
    FAILURE = "failure"


class FailureType(Enum):
    """失败类型"""
    NONE = "none"
    EXECUTION_ERROR = "execution_error"
    BAD_PLAN = "bad_plan"
    MISSING_SKILL = "missing_skill"
    RISK_MISCALIBRATION = "risk_miscalibration"


@dataclass
class MemoryCandidate:
    """记忆候选"""
    summary: str
    lesson: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "lesson": self.lesson,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCandidate":
        return cls(
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            tags=data.get("tags", [])
        )


@dataclass
class ReflectionResult:
    """反思结果"""
    status: str
    failure_type: str
    cause: str
    immediate_fix: str
    should_store_memory: bool
    memory_candidate: Optional[MemoryCandidate]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionResult":
        memory_data = data.get("memory_candidate")
        memory_candidate = MemoryCandidate.from_dict(memory_data) if memory_data else None
        
        return cls(
            status=data.get("status", "ok"),
            failure_type=data.get("failure_type", "none"),
            cause=data.get("cause", ""),
            immediate_fix=data.get("immediate_fix", ""),
            should_store_memory=data.get("should_store_memory", False),
            memory_candidate=memory_candidate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "failure_type": self.failure_type,
            "cause": self.cause,
            "immediate_fix": self.immediate_fix,
            "should_store_memory": self.should_store_memory,
            "memory_candidate": self.memory_candidate.to_dict() if self.memory_candidate else None
        }


@dataclass
class SuccessPattern:
    """成功模式"""
    pattern: str
    evidence: str
    reusability: str  # low/medium/high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "evidence": self.evidence,
            "reusability": self.reusability
        }


@dataclass
class FailurePattern:
    """失败模式"""
    pattern: str
    root_cause: str
    severity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "root_cause": self.root_cause,
            "severity": self.severity
        }


@dataclass
class LongReflectionResult:
    """长期复盘结果"""
    episode_summary: str
    success_patterns: List[SuccessPattern]
    failure_patterns: List[FailurePattern]
    new_rules: List[Dict[str, Any]]
    revise_rules: List[Dict[str, Any]]
    new_skills: List[Dict[str, Any]]
    delete_or_deprioritize_skills: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongReflectionResult":
        success_patterns = [
            SuccessPattern(
                pattern=p.get("pattern", ""),
                evidence=p.get("evidence", ""),
                reusability=p.get("reusability", "medium")
            )
            for p in data.get("success_patterns", [])
        ]
        
        failure_patterns = [
            FailurePattern(
                pattern=p.get("pattern", ""),
                root_cause=p.get("root_cause", ""),
                severity=p.get("severity", 0.5)
            )
            for p in data.get("failure_patterns", [])
        ]
        
        return cls(
            episode_summary=data.get("episode_summary", ""),
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            new_rules=data.get("new_rules", []),
            revise_rules=data.get("revise_rules", []),
            new_skills=data.get("new_skills", []),
            delete_or_deprioritize_skills=data.get("delete_or_deprioritize_skills", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_summary": self.episode_summary,
            "success_patterns": [p.to_dict() for p in self.success_patterns],
            "failure_patterns": [p.to_dict() for p in self.failure_patterns],
            "new_rules": self.new_rules,
            "revise_rules": self.revise_rules,
            "new_skills": self.new_skills,
            "delete_or_deprioritize_skills": self.delete_or_deprioritize_skills
        }


class QuickReflection:
    """局部反思模块"""
    
    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        
        self.config = config or DEFAULT_CONFIG
        self.llm = llm_client
        self.memory = memory_manager
        
        # 反思历史
        self.reflection_history: List[ReflectionResult] = []
    
    def _analyze_deviation(self,
                           plan: Dict[str, Any],
                           execution_result: Dict[str, Any],
                           current_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析偏差"""
        deviation_info = {
            "has_deviation": False,
            "deviation_type": "none",
            "severity": 0.0
        }
        
        # 检查目标是否达成
        expected_goal = plan.get("goal", "")
        actual_outcome = execution_result.get("outcome", "")
        
        if execution_result.get("success", True):
            return deviation_info
        
        deviation_info["has_deviation"] = True
        
        # 判断偏差类型
        error_type = execution_result.get("error_type", "")
        
        if error_type == "action_failed":
            deviation_info["deviation_type"] = "execution_error"
            deviation_info["severity"] = 0.5
        elif error_type == "precondition_not_met":
            deviation_info["deviation_type"] = "bad_plan"
            deviation_info["severity"] = 0.6
        elif error_type == "skill_not_found":
            deviation_info["deviation_type"] = "missing_skill"
            deviation_info["severity"] = 0.4
        elif error_type == "unexpected_danger":
            deviation_info["deviation_type"] = "risk_miscalibration"
            deviation_info["severity"] = 0.8
        else:
            deviation_info["deviation_type"] = "unknown"
            deviation_info["severity"] = 0.5
        
        # 根据生命值损失调整严重程度
        health_before = plan.get("initial_health", 20)
        health_after = current_state.get("health", 20)
        health_loss = health_before - health_after
        
        if health_loss > 10:
            deviation_info["severity"] = min(1.0, deviation_info["severity"] + 0.3)
        elif health_loss > 5:
            deviation_info["severity"] = min(1.0, deviation_info["severity"] + 0.15)
        
        return deviation_info
    
    def reflect(self,
                current_state: Dict[str, Any],
                plan: Dict[str, Any],
                execution_result: Dict[str, Any],
                recent_trajectory: List[Dict[str, Any]] = None) -> ReflectionResult:
        """执行局部反思"""
        
        # 分析偏差
        deviation_info = self._analyze_deviation(plan, execution_result, current_state)
        
        # 如果没有偏差，返回 OK
        if not deviation_info["has_deviation"]:
            return ReflectionResult(
                status="ok",
                failure_type="none",
                cause="",
                immediate_fix="",
                should_store_memory=False,
                memory_candidate=None
            )
        
        # 有偏差时，使用 LLM 进行深入分析
        if self.llm:
            system_prompt = PromptTemplates.quick_reflection_prompt()
            
            context = {
                "current_state": current_state,
                "plan": plan,
                "execution_result": execution_result,
                "recent_trajectory": recent_trajectory or [],
                "deviation_analysis": deviation_info
            }
            
            response = self.llm.generate_decision(system_prompt, context)
            
            if response.success and response.parsed:
                result = ReflectionResult.from_dict(response.parsed)
            else:
                result = self._generate_default_reflection(deviation_info)
        else:
            result = self._generate_default_reflection(deviation_info)
        
        # 记录反思历史
        self.reflection_history.append(result)
        if len(self.reflection_history) > 50:
            self.reflection_history = self.reflection_history[-25:]
        
        # 自动存储记忆
        if result.should_store_memory and result.memory_candidate and self.memory:
            self.memory.store_episode(
                summary=result.memory_candidate.summary,
                lesson=result.memory_candidate.lesson,
                tags=result.memory_candidate.tags,
                context=current_state,
                outcome="failure" if result.status == "failure" else "deviation"
            )
        
        return result
    
    def _generate_default_reflection(self, deviation_info: Dict[str, Any]) -> ReflectionResult:
        """生成默认反思结果"""
        status = "failure" if deviation_info["severity"] > 0.6 else "deviation"
        
        fix_map = {
            "execution_error": "重试动作或选择替代方案",
            "bad_plan": "重新规划，考虑当前实际条件",
            "missing_skill": "使用基础动作替代或学习新技能",
            "risk_miscalibration": "提高风险意识，采取更保守策略"
        }
        
        deviation_type = deviation_info.get("deviation_type", "unknown")
        immediate_fix = fix_map.get(deviation_type, "重新评估情况")
        
        return ReflectionResult(
            status=status,
            failure_type=deviation_type,
            cause=f"检测到{deviation_type}类型的偏差",
            immediate_fix=immediate_fix,
            should_store_memory=deviation_info["severity"] > 0.5,
            memory_candidate=MemoryCandidate(
                summary=f"计划执行出现{deviation_type}偏差",
                lesson=immediate_fix,
                tags=[deviation_type, "failure", "auto_generated"]
            ) if deviation_info["severity"] > 0.5 else None
        )
    
    def get_failure_statistics(self) -> Dict[str, int]:
        """获取失败统计"""
        stats = {}
        for result in self.reflection_history:
            if result.status != "ok":
                failure_type = result.failure_type
                stats[failure_type] = stats.get(failure_type, 0) + 1
        return stats


class LongReflection:
    """整局复盘模块"""
    
    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        
        self.config = config or DEFAULT_CONFIG
        self.llm = llm_client
        self.memory = memory_manager
        
        # 复盘历史
        self.reflection_history: List[LongReflectionResult] = []
    
    def reflect(self,
                episode_trajectory: List[Dict[str, Any]],
                episode_outcome: str,
                episode_stats: Dict[str, Any] = None) -> LongReflectionResult:
        """执行整局复盘"""
        
        if self.llm:
            system_prompt = PromptTemplates.long_reflection_prompt()
            
            context = {
                "trajectory": episode_trajectory,
                "outcome": episode_outcome,
                "stats": episode_stats or {},
                "trajectory_length": len(episode_trajectory)
            }
            
            response = self.llm.generate_decision(system_prompt, context)
            
            if response.success and response.parsed:
                result = LongReflectionResult.from_dict(response.parsed)
            else:
                result = self._generate_default_reflection(episode_trajectory, episode_outcome)
        else:
            result = self._generate_default_reflection(episode_trajectory, episode_outcome)
        
        # 记录复盘历史
        self.reflection_history.append(result)
        
        # 应用反思结果
        self._apply_reflection(result)
        
        return result
    
    def _generate_default_reflection(self,
                                     trajectory: List[Dict[str, Any]],
                                     outcome: str) -> LongReflectionResult:
        """生成默认复盘结果"""
        
        # 分析轨迹
        action_counts = {}
        failure_count = 0
        
        for step in trajectory:
            action = step.get("action", {}).get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
            
            result = step.get("result", {})
            if not result.get("success", True):
                failure_count += 1
        
        # 生成基本摘要
        most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "unknown"
        
        return LongReflectionResult(
            episode_summary=f"本局共执行{len(trajectory)}步，最常用动作为{most_common_action}，"
                           f"失败{failure_count}次，最终结果：{outcome}",
            success_patterns=[],
            failure_patterns=[
                FailurePattern(
                    pattern="执行失败",
                    root_cause="具体原因待分析",
                    severity=failure_count / max(len(trajectory), 1)
                )
            ] if failure_count > 0 else [],
            new_rules=[],
            revise_rules=[],
            new_skills=[],
            delete_or_deprioritize_skills=[]
        )
    
    def _apply_reflection(self, result: LongReflectionResult):
        """应用反思结果到记忆系统"""
        if not self.memory:
            return
        
        # 添加新规则
        for rule_data in result.new_rules:
            if rule_data.get("rule") and rule_data.get("confidence", 0) > 0.5:
                self.memory.store_rule(
                    rule=rule_data["rule"],
                    confidence=rule_data["confidence"],
                    conditions=rule_data.get("conditions", [])
                )
        
        # 修订规则
        for revision in result.revise_rules:
            old_rule = revision.get("old_rule", "")
            new_rule = revision.get("new_rule", "")
            if old_rule and new_rule:
                # 降低旧规则置信度并添加新规则
                # 这里简化处理，直接添加新规则
                self.memory.store_rule(
                    rule=new_rule,
                    confidence=0.6,
                    conditions=[]
                )
        
        # 添加新技能
        for skill_data in result.new_skills:
            if skill_data.get("name") and skill_data.get("steps"):
                self.memory.store_skill(
                    name=skill_data["name"],
                    purpose=skill_data.get("purpose", ""),
                    trigger_conditions=skill_data.get("trigger", []),
                    steps=skill_data["steps"],
                    failure_recovery=skill_data.get("failure_recovery", [])
                )
        
        # 存储成功模式为情景记忆
        for pattern in result.success_patterns:
            if pattern.reusability in ["medium", "high"]:
                self.memory.store_episode(
                    summary=pattern.pattern,
                    lesson=f"成功经验: {pattern.evidence}",
                    tags=["success", "pattern", pattern.reusability],
                    outcome="success"
                )
        
        # 存储失败模式为情景记忆
        for pattern in result.failure_patterns:
            if pattern.severity > 0.3:
                self.memory.store_episode(
                    summary=pattern.pattern,
                    lesson=f"失败原因: {pattern.root_cause} (严重度={pattern.severity:.2f})",
                    tags=["failure", "pattern"],
                    outcome="failure"
                )
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """获取洞察摘要"""
        total_success_patterns = sum(len(r.success_patterns) for r in self.reflection_history)
        total_failure_patterns = sum(len(r.failure_patterns) for r in self.reflection_history)
        total_new_rules = sum(len(r.new_rules) for r in self.reflection_history)
        total_new_skills = sum(len(r.new_skills) for r in self.reflection_history)
        
        return {
            "total_reflections": len(self.reflection_history),
            "total_success_patterns": total_success_patterns,
            "total_failure_patterns": total_failure_patterns,
            "total_new_rules": total_new_rules,
            "total_new_skills": total_new_skills
        }
