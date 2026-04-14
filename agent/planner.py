"""
Planner - 智能体规划器

作者: Jiangsheng Yu
许可证: MIT License

责任：
  - 识别当前最优先的问题（生存威胁 > 资源需求 > 探索）
  - 根据优先级和可用技能选择最佳行动计划
  - 生成最多 5 步的原子动作序列
  - 记录失败模式以避免重复同样的失败策略
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json

from config import AgentConfig, DEFAULT_CONFIG
from memory import MemoryManager
from utils import LLMClient, LLMResponse
from prompts import PromptTemplates


@dataclass
class Plan:
    """行动计划"""
    priority_issue: str                    # 当前最优先的问题
    selected_skill: str                    # 选择的技能
    goal: str                              # 目标
    subgoal: str                           # 子目标
    expected_reward: str                   # 预期收益
    main_risk: str                         # 主要风险
    action_plan: List[Dict[str, Any]]      # 动作列表
    replan_trigger: str                    # 重新规划触发条件
    confidence: float = 0.5                # 计划置信度
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            priority_issue=data.get("priority_issue", ""),
            selected_skill=data.get("selected_skill", ""),
            goal=data.get("goal", ""),
            subgoal=data.get("subgoal", ""),
            expected_reward=data.get("expected_reward", ""),
            main_risk=data.get("main_risk", ""),
            action_plan=data.get("action_plan", []),
            replan_trigger=data.get("replan_trigger", ""),
            confidence=data.get("confidence", 0.5)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority_issue": self.priority_issue,
            "selected_skill": self.selected_skill,
            "goal": self.goal,
            "subgoal": self.subgoal,
            "expected_reward": self.expected_reward,
            "main_risk": self.main_risk,
            "action_plan": self.action_plan,
            "replan_trigger": self.replan_trigger,
            "confidence": self.confidence
        }


class Planner:
    """规划器"""
    
    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        
        self.config = config or DEFAULT_CONFIG
        
        # 使用共享的 LLM 客户端和记忆管理器
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = LLMClient(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature
            )
        
        self.memory = memory_manager
        
        # 规划历史
        self.plan_history: List[Plan] = []
        
        # 失败模式记录
        self.failure_patterns: Dict[str, int] = {}  # pattern -> count
    
    def _identify_priority_issue(self, state: Dict[str, Any]) -> str:
        """识别当前最优先的问题"""
        thresholds = self.config.risk_thresholds
        
        # 按优先级检查
        health = state.get("health", 20)
        if health < thresholds.health_critical:
            return "critical_health"
        
        hunger = state.get("hunger", 20)
        if hunger < thresholds.hunger_critical:
            return "critical_hunger"
        
        # 夜晚无庇护
        if state.get("time") == "night" and not state.get("has_shelter", False):
            return "night_exposure"
        
        # 敌对生物
        nearby = state.get("nearby_entities", [])
        if any(e.get("hostile", False) for e in nearby if isinstance(e, dict)):
            if health < thresholds.health_low:
                return "hostile_low_health"
            return "hostile_nearby"
        
        if health < thresholds.health_low:
            return "low_health"
        
        if hunger < thresholds.hunger_low:
            return "low_hunger"
        
        # 工具升级
        inventory = state.get("inventory", {})
        if not any(tool in inventory for tool in ["wooden_pickaxe", "stone_pickaxe"]):
            return "no_tools"
        
        if "wooden_pickaxe" in inventory and "stone_pickaxe" not in inventory:
            nearby_blocks = state.get("nearby_blocks", [])
            if "stone" in nearby_blocks:
                return "tool_upgrade_available"
        
        return "exploration"
    
    def _select_skill(self, 
                      priority_issue: str, 
                      state: Dict[str, Any],
                      available_skills: List[Dict[str, Any]]) -> Optional[str]:
        """根据优先问题选择技能"""
        
        # 问题到技能的映射
        issue_skill_map = {
            "critical_health": ["retreat_to_safety", "emergency_shelter"],
            "critical_hunger": ["hunt_food", "gather_food"],
            "night_exposure": ["emergency_shelter", "retreat_to_safety"],
            "hostile_low_health": ["retreat_to_safety"],
            "hostile_nearby": ["retreat_to_safety", "combat"],
            "low_health": ["hunt_food", "rest"],
            "low_hunger": ["hunt_food", "gather_food"],
            "no_tools": ["gather_wood", "craft_basic_tools"],
            "tool_upgrade_available": ["mine_stone", "craft_stone_tools"],
            "exploration": ["exploration", "gather_wood"]
        }
        
        preferred_skills = issue_skill_map.get(priority_issue, [])
        available_skill_names = [s.get("name", "") for s in available_skills]
        
        # 检查是否有失败模式需要避免
        for skill_name in preferred_skills:
            pattern_key = f"{priority_issue}:{skill_name}"
            if self.failure_patterns.get(pattern_key, 0) >= 3:
                continue  # 跳过多次失败的组合
            
            if skill_name in available_skill_names:
                return skill_name
        
        # 如果没有精确匹配，返回第一个可用技能
        if available_skills:
            return available_skills[0].get("name", "")
        
        return None
    
    def _generate_fallback_plan(self, priority_issue: str) -> Plan:
        """生成默认保守计划"""
        fallback_plans = {
            "critical_health": Plan(
                priority_issue=priority_issue,
                selected_skill="retreat_to_safety",
                goal="保护生命",
                subgoal="找到安全位置",
                expected_reward="避免死亡",
                main_risk="可能被敌人追击",
                action_plan=[
                    {"action": "find_safe_spot", "args": {}},
                    {"action": "hide", "args": {}}
                ],
                replan_trigger="health_recovered",
                confidence=0.7
            ),
            "critical_hunger": Plan(
                priority_issue=priority_issue,
                selected_skill="hunt_food",
                goal="获取食物",
                subgoal="找到食物来源",
                expected_reward="恢复饥饿值",
                main_risk="可能遭遇危险",
                action_plan=[
                    {"action": "find_food_source", "args": {}},
                    {"action": "gather_or_hunt", "args": {}}
                ],
                replan_trigger="food_obtained",
                confidence=0.6
            ),
            "night_exposure": Plan(
                priority_issue=priority_issue,
                selected_skill="emergency_shelter",
                goal="建造庇护所",
                subgoal="挖掘临时避难所",
                expected_reward="安全度过夜晚",
                main_risk="可能被敌人发现",
                action_plan=[
                    {"action": "dig_hole", "args": {"depth": 3}},
                    {"action": "place_block_overhead", "args": {}},
                    {"action": "wait", "args": {"until": "dawn"}}
                ],
                replan_trigger="dawn",
                confidence=0.8
            ),
            "stuck_no_progress": Plan(
                priority_issue=priority_issue,
                selected_skill="escape_stuck",
                goal="脱困",
                subgoal="跳跃并移动到新位置",
                expected_reward="恢复移动能力",
                main_risk="可能消耗方块资源",
                action_plan=[
                    {"action": "jump", "args": {}},
                    {"action": "move", "args": {"speed": 5}},
                    {"action": "tower_up", "args": {"height": 2}}
                ],
                replan_trigger="position_changed",
                confidence=0.7
            )
        }
        
        return fallback_plans.get(priority_issue, Plan(
            priority_issue=priority_issue,
            selected_skill="wait",
            goal="评估情况",
            subgoal="观察环境",
            expected_reward="获取更多信息",
            main_risk="未知",
            action_plan=[{"action": "observe", "args": {}}],
            replan_trigger="new_information",
            confidence=0.4
        ))
    
    def plan(self, 
             current_state: Dict[str, Any],
             meta_goal: str = None) -> Plan:
        """生成行动计划"""
        
        # 识别优先问题
        priority_issue = self._identify_priority_issue(current_state)
        
        # 获取相关记忆
        memories = {}
        if self.memory:
            memories = self.memory.to_prompt_context(current_state)
        
        # 获取可用技能
        available_skills = memories.get("skills", [])
        
        # 选择技能
        selected_skill = self._select_skill(priority_issue, current_state, available_skills)
        
        # 构建 prompt
        system_prompt = PromptTemplates.planner_prompt()
        
        context = {
            "current_state": current_state,
            "recent_trajectory": memories.get("recent_trajectory", []),
            "episodic_memories": memories.get("episodic_memories", []),
            "semantic_rules": memories.get("semantic_rules", []),
            "skills": available_skills,
            "current_meta_goal": meta_goal or "长期生存与发展",
            "identified_priority": priority_issue,
            "suggested_skill": selected_skill
        }
        
        # 调用 LLM
        response = self.llm.generate_decision(system_prompt, context)
        
        if response.success and response.parsed:
            plan = Plan.from_dict(response.parsed)
            
            # 验证计划
            plan = self._validate_plan(plan, current_state)
        else:
            # 使用默认计划
            plan = self._generate_fallback_plan(priority_issue)
        
        # 记录计划历史
        self.plan_history.append(plan)
        if len(self.plan_history) > 100:
            self.plan_history = self.plan_history[-50:]
        
        return plan
    
    def _validate_plan(self, plan: Plan, state: Dict[str, Any]) -> Plan:
        """验证和调整计划"""
        thresholds = self.config.risk_thresholds
        max_actions = self.config.planner.max_actions_per_plan
        
        # 限制动作数量
        if len(plan.action_plan) > max_actions:
            plan.action_plan = plan.action_plan[:max_actions]
        
        # 如果生命值低，不允许战斗计划
        health = state.get("health", 20)
        if health < thresholds.combat_avoid_health:
            combat_actions = ["attack", "fight", "combat"]
            plan.action_plan = [
                a for a in plan.action_plan 
                if a.get("action", "").lower() not in combat_actions
            ]
            
            if not plan.action_plan:
                plan.action_plan = [{"action": "retreat", "args": {}}]
        
        return plan
    
    def record_failure(self, plan: Plan, failure_reason: str):
        """记录计划失败"""
        pattern_key = f"{plan.priority_issue}:{plan.selected_skill}"
        self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
    
    def record_success(self, plan: Plan):
        """记录计划成功"""
        pattern_key = f"{plan.priority_issue}:{plan.selected_skill}"
        # 成功时减少失败计数
        if pattern_key in self.failure_patterns:
            self.failure_patterns[pattern_key] = max(0, self.failure_patterns[pattern_key] - 1)
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return {
            "total_plans": len(self.plan_history),
            "failure_patterns": self.failure_patterns,
            "recent_plans": [p.to_dict() for p in self.plan_history[-5:]]
        }
