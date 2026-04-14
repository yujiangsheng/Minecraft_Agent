"""
技能库 (Skill Library)

作者: Jiangsheng Yu
许可证: MIT License

存储可复用的技能，每个技能包含触发条件、执行步骤和失败恢复。
预置 6 个基础技能，可通过 SkillBuilder 自动生成更多技能。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class Skill:
    """单个技能"""
    id: str
    name: str                                    # 技能名称
    purpose: str                                 # 技能目的
    trigger_conditions: List[str]                # 触发条件
    preconditions: List[str]                     # 前置条件
    steps: List[Dict[str, Any]]                  # 执行步骤
    stop_conditions: List[str]                   # 停止条件
    failure_recovery: List[Dict[str, Any]]       # 失败恢复策略
    metrics: Dict[str, str] = field(default_factory=dict)  # 成功/风险信号
    priority: float = 0.5                        # 优先级
    created_at: str = ""
    updated_at: str = ""
    usage_count: int = 0                         # 使用次数
    success_count: int = 0                       # 成功次数
    source: str = "manual"                       # 来源: manual/learned/evolved
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "trigger_conditions": self.trigger_conditions,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "stop_conditions": self.stop_conditions,
            "failure_recovery": self.failure_recovery,
            "metrics": self.metrics,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            purpose=data.get("purpose", ""),
            trigger_conditions=data.get("trigger_conditions", []),
            preconditions=data.get("preconditions", []),
            steps=data.get("steps", []),
            stop_conditions=data.get("stop_conditions", []),
            failure_recovery=data.get("failure_recovery", []),
            metrics=data.get("metrics", {}),
            priority=data.get("priority", 0.5),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            source=data.get("source", "manual")
        )


class SkillLibrary:
    """技能库管理器"""
    
    # 预置的基础技能
    DEFAULT_SKILLS = [
        {
            "name": "emergency_shelter",
            "purpose": "在危险情况下快速建造紧急庇护所",
            "trigger_conditions": ["night", "no_shelter", "hostile_nearby"],
            "preconditions": ["has_blocks"],
            "steps": [
                {"action": "find_safe_spot", "args": {}},
                {"action": "dig_hole", "args": {"depth": 3}},
                {"action": "place_block_overhead", "args": {}},
                {"action": "wait", "args": {"until": "dawn"}}
            ],
            "stop_conditions": ["dawn", "shelter_found"],
            "failure_recovery": [
                {"condition": "no_blocks", "action": "punch_dirt", "args": {}},
                {"condition": "enemy_attacking", "action": "run_away", "args": {"direction": "random"}}
            ],
            "metrics": {"success_signal": "survived_night", "risk_signal": "took_damage"},
            "priority": 0.95
        },
        {
            "name": "gather_wood",
            "purpose": "采集木头资源",
            "trigger_conditions": ["need_wood", "trees_nearby"],
            "preconditions": [],
            "steps": [
                {"action": "find_tree", "args": {}},
                {"action": "break_log", "args": {"count": 5}},
                {"action": "collect_drops", "args": {}}
            ],
            "stop_conditions": ["inventory_full", "no_trees", "danger_detected"],
            "failure_recovery": [
                {"condition": "danger_detected", "action": "retreat", "args": {}},
                {"condition": "no_trees", "action": "explore", "args": {"radius": 50}}
            ],
            "metrics": {"success_signal": "wood_in_inventory", "risk_signal": "hostile_nearby"},
            "priority": 0.8
        },
        {
            "name": "craft_basic_tools",
            "purpose": "制作基础木制工具",
            "trigger_conditions": ["has_wood", "no_tools"],
            "preconditions": ["has_wood_logs"],
            "steps": [
                {"action": "craft", "args": {"item": "crafting_table", "count": 1}},
                {"action": "craft", "args": {"item": "wood_planks", "count": 8}},
                {"action": "craft", "args": {"item": "sticks", "count": 4}},
                {"action": "craft", "args": {"item": "wooden_pickaxe", "count": 1}},
                {"action": "craft", "args": {"item": "wooden_sword", "count": 1}}
            ],
            "stop_conditions": ["has_basic_tools"],
            "failure_recovery": [
                {"condition": "not_enough_wood", "action": "gather_wood", "args": {}}
            ],
            "metrics": {"success_signal": "tools_crafted", "risk_signal": "none"},
            "priority": 0.85
        },
        {
            "name": "hunt_food",
            "purpose": "猎杀动物获取食物",
            "trigger_conditions": ["hunger_low", "animals_nearby"],
            "preconditions": ["has_weapon"],
            "steps": [
                {"action": "find_animal", "args": {"type": "passive"}},
                {"action": "approach_carefully", "args": {}},
                {"action": "attack", "args": {}},
                {"action": "collect_drops", "args": {}},
                {"action": "cook_food", "args": {"if": "has_furnace"}}
            ],
            "stop_conditions": ["food_sufficient", "no_animals", "danger_detected"],
            "failure_recovery": [
                {"condition": "no_weapon", "action": "craft_weapon", "args": {}},
                {"condition": "no_animals", "action": "explore", "args": {"radius": 100}}
            ],
            "metrics": {"success_signal": "food_obtained", "risk_signal": "health_lost"},
            "priority": 0.85
        },
        {
            "name": "retreat_to_safety",
            "purpose": "遇到危险时撤退到安全位置",
            "trigger_conditions": ["danger_detected", "health_low"],
            "preconditions": [],
            "steps": [
                {"action": "assess_threat", "args": {}},
                {"action": "find_escape_route", "args": {}},
                {"action": "run", "args": {"direction": "away_from_threat"}},
                {"action": "hide_or_defend", "args": {}}
            ],
            "stop_conditions": ["safe", "threat_eliminated"],
            "failure_recovery": [
                {"condition": "cornered", "action": "fight", "args": {}},
                {"condition": "surrounded", "action": "dig_down", "args": {}}
            ],
            "metrics": {"success_signal": "health_stable", "risk_signal": "health_decreasing"},
            "priority": 0.95
        },
        {
            "name": "mine_stone",
            "purpose": "开采石头升级工具",
            "trigger_conditions": ["has_wooden_pickaxe", "stone_nearby"],
            "preconditions": ["has_pickaxe"],
            "steps": [
                {"action": "find_stone", "args": {}},
                {"action": "mine_cobblestone", "args": {"count": 10}},
                {"action": "craft", "args": {"item": "stone_pickaxe", "count": 1}},
                {"action": "craft", "args": {"item": "stone_sword", "count": 1}}
            ],
            "stop_conditions": ["has_stone_tools", "danger_detected"],
            "failure_recovery": [
                {"condition": "pickaxe_broken", "action": "craft_pickaxe", "args": {}},
                {"condition": "danger_detected", "action": "retreat", "args": {}}
            ],
            "metrics": {"success_signal": "stone_tools_crafted", "risk_signal": "cave_danger"},
            "priority": 0.75
        }
    ]
    
    def __init__(self, max_skills: int = 30):
        self.max_skills = max_skills
        self.skills: Dict[str, Skill] = {}
        self._trigger_index: Dict[str, List[str]] = {}  # trigger -> skill_ids
        
        # 初始化默认技能
        self._init_default_skills()
    
    def _init_default_skills(self):
        """初始化默认技能"""
        for skill_data in self.DEFAULT_SKILLS:
            self.add(
                name=skill_data["name"],
                purpose=skill_data["purpose"],
                trigger_conditions=skill_data["trigger_conditions"],
                preconditions=skill_data.get("preconditions", []),
                steps=skill_data["steps"],
                stop_conditions=skill_data.get("stop_conditions", []),
                failure_recovery=skill_data.get("failure_recovery", []),
                metrics=skill_data.get("metrics", {}),
                priority=skill_data.get("priority", 0.5),
                source="manual"
            )
    
    def _generate_id(self, name: str) -> str:
        """生成技能 ID"""
        return hashlib.md5(name.encode()).hexdigest()[:12]
    
    def add(self,
            name: str,
            purpose: str,
            trigger_conditions: List[str],
            preconditions: List[str] = None,
            steps: List[Dict[str, Any]] = None,
            stop_conditions: List[str] = None,
            failure_recovery: List[Dict[str, Any]] = None,
            metrics: Dict[str, str] = None,
            priority: float = 0.5,
            source: str = "learned") -> Skill:
        """添加新技能"""
        skill_id = self._generate_id(name)
        
        # 如果技能已存在，更新
        if skill_id in self.skills:
            existing = self.skills[skill_id]
            existing.steps = steps or existing.steps
            existing.updated_at = datetime.now().isoformat()
            return existing
        
        # 如果超过最大数量，删除使用最少的学习技能
        if len(self.skills) >= self.max_skills:
            self._evict_one()
        
        timestamp = datetime.now().isoformat()
        skill = Skill(
            id=skill_id,
            name=name,
            purpose=purpose,
            trigger_conditions=trigger_conditions,
            preconditions=preconditions or [],
            steps=steps or [],
            stop_conditions=stop_conditions or [],
            failure_recovery=failure_recovery or [],
            metrics=metrics or {},
            priority=priority,
            created_at=timestamp,
            updated_at=timestamp,
            source=source
        )
        
        self.skills[skill_id] = skill
        
        # 更新触发条件索引
        for trigger in trigger_conditions:
            if trigger not in self._trigger_index:
                self._trigger_index[trigger] = []
            self._trigger_index[trigger].append(skill_id)
        
        return skill
    
    def _evict_one(self):
        """驱逐一个技能"""
        learned_skills = [s for s in self.skills.values() if s.source == "learned"]
        
        if learned_skills:
            # 按使用次数和成功率排序
            sorted_skills = sorted(
                learned_skills,
                key=lambda s: s.usage_count * s.success_rate
            )
            victim = sorted_skills[0]
        else:
            # 删除优先级最低的
            sorted_skills = sorted(
                self.skills.values(),
                key=lambda s: s.priority
            )
            victim = sorted_skills[0]
        
        self.remove(victim.id)
    
    def remove(self, skill_id: str):
        """删除技能"""
        if skill_id not in self.skills:
            return
        
        skill = self.skills[skill_id]
        
        # 从触发索引中移除
        for trigger in skill.trigger_conditions:
            if trigger in self._trigger_index and skill_id in self._trigger_index[trigger]:
                self._trigger_index[trigger].remove(skill_id)
        
        del self.skills[skill_id]
    
    def get(self, skill_id: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(skill_id)
    
    def get_by_name(self, name: str) -> Optional[Skill]:
        """按名称获取技能"""
        skill_id = self._generate_id(name)
        return self.get(skill_id)
    
    def record_usage(self, skill_id: str, success: bool):
        """记录技能使用结果"""
        skill = self.skills.get(skill_id)
        if not skill:
            return
        
        skill.usage_count += 1
        if success:
            skill.success_count += 1
        
        skill.updated_at = datetime.now().isoformat()
    
    def search_by_triggers(self, current_conditions: List[str]) -> List[Skill]:
        """按触发条件搜索适用的技能"""
        matched_skills = []
        
        for skill in self.skills.values():
            # 检查是否满足任一触发条件
            triggered = any(
                any(self._condition_matches(tc, cc) for cc in current_conditions)
                for tc in skill.trigger_conditions
            )
            
            if triggered:
                # 检查前置条件
                preconditions_met = all(
                    any(self._condition_matches(pc, cc) for cc in current_conditions)
                    for pc in skill.preconditions
                ) if skill.preconditions else True
                
                if preconditions_met:
                    matched_skills.append(skill)
        
        # 按优先级和成功率排序
        matched_skills.sort(
            key=lambda s: s.priority * (0.5 + 0.5 * s.success_rate),
            reverse=True
        )
        
        return matched_skills
    
    def _condition_matches(self, skill_condition: str, current_condition: str) -> bool:
        """检查条件是否匹配"""
        return skill_condition.lower() in current_condition.lower() or \
               current_condition.lower() in skill_condition.lower()
    
    def get_all(self) -> List[Skill]:
        """获取所有技能"""
        return list(self.skills.values())
    
    def get_by_priority(self, min_priority: float = 0.5) -> List[Skill]:
        """按优先级获取技能"""
        skills = [s for s in self.skills.values() if s.priority >= min_priority]
        skills.sort(key=lambda s: s.priority, reverse=True)
        return skills
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            "skills": [s.to_dict() for s in self.skills.values()],
            "max_skills": self.max_skills
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.max_skills = data.get("max_skills", 30)
        self.skills.clear()
        self._trigger_index.clear()
        
        for skill_data in data.get("skills", []):
            skill = Skill.from_dict(skill_data)
            self.skills[skill.id] = skill
            
            for trigger in skill.trigger_conditions:
                if trigger not in self._trigger_index:
                    self._trigger_index[trigger] = []
                self._trigger_index[trigger].append(skill.id)
    
    def to_prompt_format(self, skills: List[Skill] = None) -> List[Dict[str, Any]]:
        """转换为 prompt 格式"""
        if skills is None:
            skills = self.get_by_priority()
        
        return [
            {
                "name": s.name,
                "purpose": s.purpose,
                "trigger_conditions": s.trigger_conditions,
                "steps": [step.get("action", str(step)) for step in s.steps],
                "success_rate": round(s.success_rate, 2)
            }
            for s in skills
        ]
