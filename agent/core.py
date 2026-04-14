"""
Agent Core - 智能体核心决策系统（多层流水线版）

作者: Jiangsheng Yu
许可证: MIT License

责任：
  - 感知环境状态并更新内部状态
  - 规则预判优先级 → 构建丰富上下文 → 调用 LLM 决策流水线
  - 跟踪近期失败模式，防止重复低效行为
  - 管理计划执行流程和重新规划触发
  - 风险评估与保守模式切换
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import deque
import json
import logging

from config import AgentConfig, DEFAULT_CONFIG
from memory import MemoryManager
from utils import LLMClient, LLMResponse
from prompts import PromptTemplates

logger = logging.getLogger("AgentCore")


class AgentMode(Enum):
    """智能体模式"""
    SURVIVE = "survive"
    GATHER = "gather"
    CRAFT = "craft"
    EXPLORE = "explore"
    COMBAT = "combat"
    RETREAT = "retreat"
    BUILD = "build"


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AgentState:
    """智能体内部状态"""
    current_mode: AgentMode = AgentMode.GATHER
    current_goal: str = ""
    current_subgoal: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    conservative_mode: bool = False
    steps_since_reflection: int = 0
    total_steps: int = 0
    episode_count: int = 0
    priority_issue: str = ""          # 规则系统预判的优先问题
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode.value,
            "current_goal": self.current_goal,
            "current_subgoal": self.current_subgoal,
            "risk_level": self.risk_level.value,
            "conservative_mode": self.conservative_mode,
            "priority_issue": self.priority_issue,
            "steps_since_reflection": self.steps_since_reflection,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count
        }


@dataclass
class AgentDecision:
    """智能体决策"""
    mode: str
    goal: str
    subgoal: str
    reason: str
    risk_level: str
    memory_references: List[Dict[str, Any]]
    action_plan: List[Dict[str, Any]]
    replan_trigger: str
    reflection_needed: bool
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentDecision":
        return cls(
            mode=data.get("mode", "gather"),
            goal=data.get("goal", ""),
            subgoal=data.get("subgoal", ""),
            reason=data.get("reason", ""),
            risk_level=data.get("risk_level", "low"),
            memory_references=data.get("memory_references", data.get("use_memory", [])),
            action_plan=data.get("action_plan", []),
            replan_trigger=data.get("replan_trigger", data.get("stop_condition", "")),
            reflection_needed=data.get("reflection_needed", data.get("reflection_flag", "none") != "none"),
            raw_response=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "goal": self.goal,
            "subgoal": self.subgoal,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "memory_references": self.memory_references,
            "action_plan": self.action_plan,
            "replan_trigger": self.replan_trigger,
            "reflection_needed": self.reflection_needed
        }


class AgentCore:
    """智能体核心 — 多层决策流水线"""
    
    # 近期失败缓冲区最大容量
    MAX_RECENT_FAILURES = 10
    # 同一动作连续失败阈值（超过此值将被标记到上下文）
    CONSECUTIVE_FAIL_THRESHOLD = 2
    
    def __init__(self, 
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        
        self.config = config or DEFAULT_CONFIG
        
        # 初始化 LLM 客户端
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = LLMClient(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                api_base=self.config.llm.api_base,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        
        # 初始化记忆管理器
        if memory_manager:
            self.memory = memory_manager
        else:
            self.memory = MemoryManager(
                episodic_max_size=self.config.memory.episodic_max_size,
                semantic_max_rules=self.config.memory.semantic_max_rules,
                skills_max_count=self.config.memory.skills_max_count,
                trajectory_window=self.config.memory.trajectory_window,
                retrieval_weights=self.config.memory.retrieval_weights
            )
        
        # 智能体状态
        self.state = AgentState()
        
        # 当前环境状态
        self.current_env_state: Dict[str, Any] = {}
        
        # 当前计划
        self.current_plan: Optional[AgentDecision] = None
        self.plan_step_index: int = 0
        
        # ── 用户自定义任务 ──
        self.user_task: Optional[str] = None
        
        # ── 失败跟踪（防止重复低效行为）──
        self.recent_failures: deque = deque(maxlen=self.MAX_RECENT_FAILURES)
        self._consecutive_fail_action: str = ""
        self._consecutive_fail_count: int = 0
        self._last_position: Optional[tuple] = None
        self._stuck_steps: int = 0  # 位置不变的连续步数
    
    def _should_enter_conservative_mode(self, env_state: Dict[str, Any]) -> bool:
        """判断是否应该进入保守模式"""
        thresholds = self.config.risk_thresholds
        
        # 检查各种危险条件
        conditions = []
        
        # 时间
        if env_state.get("time") == "night":
            conditions.append("night")
        
        # 生命值
        health = env_state.get("health", 20)
        if health < thresholds.health_low:
            conditions.append("low_health")
        
        # 饥饿值
        hunger = env_state.get("hunger", 20)
        if hunger < thresholds.hunger_low:
            conditions.append("low_hunger")
        
        # 敌对生物
        nearby = env_state.get("nearby_entities", [])
        if any(e.get("hostile", False) for e in nearby if isinstance(e, dict)):
            conditions.append("hostile_nearby")
        
        # 武器
        inventory = env_state.get("inventory", {})
        if not any(weapon in inventory for weapon in ["wooden_sword", "stone_sword", "iron_sword"]):
            conditions.append("no_weapon")
        
        # 检查是否触发保守模式
        triggers = self.config.planner.conservative_mode_triggers
        return any(c in triggers for c in conditions)
    
    def perceive(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境状态并更新内部状态"""
        self.current_env_state = env_state
        
        # 检查是否需要进入保守模式
        self.state.conservative_mode = self._should_enter_conservative_mode(env_state)
        
        # 评估风险等级
        self.state.risk_level = self._assess_risk(env_state)
        
        # 规则预判优先问题（来自 Planner 逻辑）
        self.state.priority_issue = self._identify_priority_issue(env_state)
        
        # 检测位置卡住
        self._detect_stuck(env_state)
        
        # 获取相关记忆
        memories = self.memory.to_prompt_context(env_state)
        
        return {
            "env_state": env_state,
            "agent_state": self.state.to_dict(),
            "memories": memories
        }
    
    def _assess_risk(self, env_state: Dict[str, Any]) -> RiskLevel:
        """评估当前风险等级"""
        risk_score = 0.0
        thresholds = self.config.risk_thresholds
        
        # 生命值风险
        health = env_state.get("health", 20)
        if health < thresholds.health_critical:
            risk_score += 0.5
        elif health < thresholds.health_low:
            risk_score += 0.3
        
        # 饥饿值风险
        hunger = env_state.get("hunger", 20)
        if hunger < thresholds.hunger_critical:
            risk_score += 0.3
        elif hunger < thresholds.hunger_low:
            risk_score += 0.15
        
        # 夜晚风险
        if env_state.get("time") == "night":
            if not env_state.get("has_shelter", False):
                risk_score += 0.4
        
        # 敌对生物风险
        nearby = env_state.get("nearby_entities", [])
        hostile_count = sum(1 for e in nearby if isinstance(e, dict) and e.get("hostile", False))
        risk_score += min(0.3, hostile_count * 0.1)
        
        # 判断风险等级
        if risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    # ── 优先问题识别（借鉴 Planner 的逻辑，用于增强 LLM 上下文）──

    def _identify_priority_issue(self, state: Dict[str, Any]) -> str:
        """规则预判当前最优先的问题"""
        thresholds = self.config.risk_thresholds
        
        health = state.get("health", 20)
        hunger = state.get("hunger", 20)
        
        if health < thresholds.health_critical:
            return "critical_health"
        if hunger < thresholds.hunger_critical:
            return "critical_hunger"
        if state.get("time") == "night" and not state.get("has_shelter", False):
            return "night_exposure"
        
        nearby = state.get("nearby_entities", [])
        has_hostile = any(e.get("hostile", False) for e in nearby if isinstance(e, dict))
        if has_hostile and health < thresholds.health_low:
            return "hostile_low_health"
        if has_hostile:
            return "hostile_nearby"
        if health < thresholds.health_low:
            return "low_health"
        if hunger < thresholds.hunger_low:
            return "low_hunger"
        
        # 工具链
        inventory = state.get("inventory", {})
        has_any_tool = any(k for k in inventory
                          if "pickaxe" in k or "axe" in k or "sword" in k)
        if not has_any_tool:
            return "no_tools"
        
        # 如果位置卡住不动
        if self._stuck_steps >= 5:
            return "stuck_no_progress"
        
        return "normal_exploration"

    # ── 位置卡住检测 ──

    def _detect_stuck(self, env_state: Dict[str, Any]):
        """检测玩家是否长时间停留在同一位置"""
        pos = env_state.get("position", {})
        if isinstance(pos, dict):
            current_pos = (
                round(pos.get("x", 0), 0),
                round(pos.get("y", 0), 0),
                round(pos.get("z", 0), 0)
            )
        else:
            current_pos = None
        
        if current_pos and current_pos == self._last_position:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        
        self._last_position = current_pos

    # ── 失败跟踪 ──

    def record_failure(self, action: str, reason: str):
        """记录一次失败动作"""
        self.recent_failures.append({"action": action, "reason": reason})
        
        if action == self._consecutive_fail_action:
            self._consecutive_fail_count += 1
        else:
            self._consecutive_fail_action = action
            self._consecutive_fail_count = 1

    def get_recent_failures_for_prompt(self) -> List[Dict[str, str]]:
        """获取需要告知 LLM 的近期失败列表"""
        failures = list(self.recent_failures)
        
        # 如果位置卡住，追加提示
        if self._stuck_steps >= 3:
            failures.append({
                "action": "position_stuck",
                "reason": f"位置已连续 {self._stuck_steps} 步未变化，请使用 explore 或 move_to 改变位置"
            })
        
        return failures

    # ════════════════════════════════════════════════════════════
    #  决策主入口 — 使用 decision_pipeline_prompt
    # ════════════════════════════════════════════════════════════
    
    def decide(self, env_state: Dict[str, Any] = None) -> AgentDecision:
        """做出决策（多层流水线）"""
        if env_state:
            perception = self.perceive(env_state)
        else:
            perception = {
                "env_state": self.current_env_state,
                "agent_state": self.state.to_dict(),
                "memories": self.memory.to_prompt_context(self.current_env_state)
            }
        
        # 构建系统 prompt（使用新的决策流水线 Prompt）
        system_prompt = PromptTemplates.decision_pipeline_prompt()
        
        # 构建丰富上下文（使用 PromptTemplates 的新工具方法）
        context = PromptTemplates.build_decision_context(
            current_state=perception["env_state"],
            memories=perception["memories"],
            agent_internal_state=perception["agent_state"],
            recent_failures=self.get_recent_failures_for_prompt(),
            priority_issue=self.state.priority_issue
        )
        
        # 如果处于保守模式，添加提示
        if self.state.conservative_mode:
            context["conservative_mode_active"] = True
            context["conservative_reason"] = "当前处于高风险状态，请采取保守策略"
        
        # 注入用户自定义任务
        if self.user_task:
            context["user_task"] = self.user_task
            context["user_task_instruction"] = (
                "用户设定了明确任务目标，请在保证生存安全的前提下优先完成此任务。"
                "如果当前存在生命威胁，仍应优先处理生存问题，之后继续推进用户任务。"
            )
        
        # 调用 LLM
        response = self.llm.generate_decision(system_prompt, context)
        
        if response.success and response.parsed:
            decision = AgentDecision.from_dict(response.parsed)
        else:
            # 生成默认保守决策
            decision = self._generate_fallback_decision()
        
        # 更新状态
        self.current_plan = decision
        self.plan_step_index = 0
        
        try:
            self.state.current_mode = AgentMode(decision.mode)
        except ValueError:
            self.state.current_mode = AgentMode.SURVIVE
        
        self.state.current_goal = decision.goal
        self.state.current_subgoal = decision.subgoal
        
        try:
            self.state.risk_level = RiskLevel(decision.risk_level)
        except ValueError:
            self.state.risk_level = RiskLevel.MEDIUM
        
        return decision
    
    def _generate_fallback_decision(self) -> AgentDecision:
        """生成默认保守决策（使用有效的 Luanti 动作名）"""
        # 根据优先问题选择不同的回退策略
        issue = self.state.priority_issue
        
        if issue == "critical_health":
            actions = [{"action": "eat_food", "args": {}}, {"action": "retreat", "args": {}}]
            goal = "生命危急，立即恢复"
        elif issue == "critical_hunger":
            actions = [{"action": "find_resource", "args": {"resource_type": "food"}}, {"action": "eat_food", "args": {}}]
            goal = "饥饿危急，寻找食物"
        elif issue == "night_exposure":
            actions = [{"action": "build_shelter", "args": {}}, {"action": "light_area", "args": {}}]
            goal = "夜晚无庇护，建造庇护所"
        elif issue == "hostile_nearby" or issue == "hostile_low_health":
            actions = [{"action": "retreat", "args": {}}, {"action": "flee_from", "args": {}}]
            goal = "敌对威胁，紧急撤退"
        elif issue == "stuck_no_progress":
            actions = [{"action": "explore", "args": {"speed": 5}}, {"action": "find_resource", "args": {"resource_type": "tree"}}]
            goal = "位置卡住，积极探索新区域"
        elif issue == "no_tools":
            actions = [{"action": "gather_wood", "args": {}}, {"action": "craft_tool", "args": {"tool_type": "default:pick_wood"}}]
            goal = "没有工具，采集木材制作工具"
        else:
            actions = [{"action": "explore", "args": {"speed": 4}}, {"action": "find_resource", "args": {"resource_type": "tree"}}, {"action": "gather_wood", "args": {}}]
            goal = "探索环境，采集资源"
        
        return AgentDecision(
            mode="survive",
            goal=goal,
            subgoal="LLM 响应异常，执行规则回退策略",
            reason=f"LLM 响应失败，基于优先问题 [{issue}] 采用规则回退策略",
            risk_level="high",
            memory_references=[],
            action_plan=actions,
            replan_trigger="situation_assessed",
            reflection_needed=True
        )
    
    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """获取下一个待执行的动作"""
        if not self.current_plan:
            return None
        
        actions = self.current_plan.action_plan
        if self.plan_step_index >= len(actions):
            return None
        
        action = actions[self.plan_step_index]
        return action
    
    def step(self, action_result: Dict[str, Any]) -> bool:
        """执行一步并更新状态
        
        Returns:
            bool: 是否需要重新规划
        """
        if not self.current_plan:
            return True
        
        # 记录轨迹
        current_action = self.get_next_action() or {"action": "unknown", "args": {}}
        self.memory.add_trajectory(
            state=self.current_env_state,
            action=current_action,
            result=action_result
        )
        
        # 跟踪失败
        if not action_result.get("success", True):
            action_name = current_action.get("action", "unknown")
            reason = action_result.get("outcome", action_result.get("error_type", "unknown"))
            self.record_failure(action_name, reason)
            logger.debug(f"记录失败: {action_name} -> {reason}")
        else:
            # 成功时重置连续失败计数
            self._consecutive_fail_count = 0
            self._consecutive_fail_action = ""
        
        self.plan_step_index += 1
        self.state.total_steps += 1
        self.state.steps_since_reflection += 1
        
        # 检查是否需要重新规划
        replan_needed = False
        
        # 计划执行完毕
        if self.plan_step_index >= len(self.current_plan.action_plan):
            replan_needed = True
        
        # 连续失败 → 强制重新规划
        if self._consecutive_fail_count >= self.CONSECUTIVE_FAIL_THRESHOLD:
            logger.info(f"动作 [{self._consecutive_fail_action}] 连续失败 {self._consecutive_fail_count} 次，强制重新规划")
            replan_needed = True
        
        # 检查重新规划触发条件
        if self.current_plan.replan_trigger:
            trigger = self.current_plan.replan_trigger.lower()
            if action_result.get("trigger_condition_met", False):
                replan_needed = True
            if "danger" in trigger and action_result.get("danger_detected", False):
                replan_needed = True
            if "full" in trigger and action_result.get("inventory_full", False):
                replan_needed = True
        
        # 检查是否需要反思
        if self.current_plan.reflection_needed:
            self.state.steps_since_reflection = 0
        
        return replan_needed
    
    def start_new_episode(self):
        """开始新的一局"""
        self.state.episode_count += 1
        self.state.total_steps = 0
        self.state.steps_since_reflection = 0
        self.state.priority_issue = ""
        self.memory.clear_trajectory()
        self.current_plan = None
        self.plan_step_index = 0
        # 重置失败跟踪
        self.recent_failures.clear()
        self._consecutive_fail_action = ""
        self._consecutive_fail_count = 0
        self._last_position = None
        self._stuck_steps = 0
    
    def end_episode(self, outcome: str = "unknown") -> Dict[str, Any]:
        """结束当前局"""
        trajectory = self.memory.get_full_trajectory()
        
        episode_data = {
            "episode_id": self.state.episode_count,
            "total_steps": self.state.total_steps,
            "outcome": outcome,
            "trajectory": trajectory,
            "final_state": self.current_env_state
        }
        
        return episode_data
    
    def save_state(self, path: str):
        """保存智能体状态"""
        self.memory.save(path)
    
    def load_state(self, path: str):
        """加载智能体状态"""
        self.memory.load(path)
