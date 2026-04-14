"""
记忆管理器 (Memory Manager)

作者: Jiangsheng Yu
许可证: MIT License

统一管理三类记忆 + 轨迹窗口：
  - EpisodicMemory:  情景记忆（具体场景下的经验）
  - SemanticMemory:  语义规则（通用知识和规则）
  - SkillLibrary:    技能库（可复用的动作序列）
  - Trajectory:      轨迹窗口（近期 state-action-result 序列）
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import json
import os

from memory.episodic import EpisodicMemory, Episode
from memory.semantic import SemanticMemory, Rule
from memory.skills import SkillLibrary, Skill


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"       # 情景记忆
    SEMANTIC = "semantic"       # 语义规则
    SKILL = "skill"             # 技能
    TRAJECTORY = "trajectory"   # 轨迹


@dataclass
class TrajectoryItem:
    """轨迹项"""
    step: int
    state: Dict[str, Any]
    action: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "action": self.action,
            "result": self.result,
            "timestamp": self.timestamp
        }


class MemoryManager:
    """统一记忆管理器"""
    
    def __init__(self, 
                 episodic_max_size: int = 100,
                 semantic_max_rules: int = 50,
                 skills_max_count: int = 30,
                 trajectory_window: int = 20,
                 retrieval_weights: Dict[str, float] = None):
        
        # 初始化各类型记忆
        self.episodic = EpisodicMemory(max_size=episodic_max_size)
        self.semantic = SemanticMemory(max_rules=semantic_max_rules)
        self.skills = SkillLibrary(max_skills=skills_max_count)
        
        # 轨迹记忆（有限窗口）
        self.trajectory_window = trajectory_window
        self.trajectory: deque = deque(maxlen=trajectory_window)
        self.step_counter = 0
        
        # 检索权重
        self.retrieval_weights = retrieval_weights or {
            "episodic": 0.4,
            "semantic": 0.3,
            "skill": 0.3
        }
        
        # 存储路径
        self.storage_path: Optional[str] = None
    
    def set_storage_path(self, path: str):
        """设置存储路径"""
        self.storage_path = path
        os.makedirs(path, exist_ok=True)
    
    # ========== 轨迹管理 ==========
    
    def add_trajectory(self, 
                       state: Dict[str, Any],
                       action: Dict[str, Any],
                       result: Dict[str, Any]) -> TrajectoryItem:
        """添加轨迹项"""
        from datetime import datetime
        
        self.step_counter += 1
        item = TrajectoryItem(
            step=self.step_counter,
            state=state,
            action=action,
            result=result,
            timestamp=datetime.now().isoformat()
        )
        self.trajectory.append(item)
        return item
    
    def get_recent_trajectory(self, n: int = None) -> List[Dict[str, Any]]:
        """获取最近的轨迹"""
        if n is None:
            n = len(self.trajectory)
        
        recent = list(self.trajectory)[-n:]
        return [item.to_dict() for item in recent]
    
    def clear_trajectory(self):
        """清空轨迹"""
        self.trajectory.clear()
        self.step_counter = 0
    
    def get_full_trajectory(self) -> List[Dict[str, Any]]:
        """获取完整轨迹（用于复盘）"""
        return [item.to_dict() for item in self.trajectory]
    
    # ========== 统一检索 ==========
    
    def retrieve(self, 
                 current_state: Dict[str, Any],
                 current_conditions: List[str] = None,
                 tags: List[str] = None,
                 top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """统一检索所有类型的相关记忆"""
        
        conditions = current_conditions or self._extract_conditions(current_state)
        
        result = {
            "episodic_memories": [],
            "semantic_rules": [],
            "skills": [],
            "recent_trajectory": self.get_recent_trajectory(10)
        }
        
        # 检索情景记忆
        if tags:
            episodes = self.episodic.search_by_tags(tags, top_k=top_k)
        else:
            episodes = self.episodic.search_by_context(current_state, top_k=top_k)
        result["episodic_memories"] = self.episodic.to_prompt_format(episodes)
        
        # 检索语义规则
        rules = self.semantic.search_by_conditions(conditions)
        result["semantic_rules"] = self.semantic.to_prompt_format(rules[:top_k])
        
        # 检索技能
        skills = self.skills.search_by_triggers(conditions)
        result["skills"] = self.skills.to_prompt_format(skills[:top_k])
        
        return result
    
    def _extract_conditions(self, state: Dict[str, Any]) -> List[str]:
        """从状态中提取条件"""
        conditions = []
        
        # 时间相关
        if state.get("time") == "night":
            conditions.append("night")
        elif state.get("time") == "day":
            conditions.append("day")
        
        # 生命值相关
        health = state.get("health", 20)
        if health < 5:
            conditions.extend(["health_critical", "health_low", "danger_detected"])
        elif health < 8:
            conditions.append("health_low")
        
        # 饥饿值相关
        hunger = state.get("hunger", 20)
        if hunger < 3:
            conditions.extend(["hunger_critical", "hunger_low"])
        elif hunger < 6:
            conditions.append("hunger_low")
        
        # 库存相关
        inventory = state.get("inventory", {})
        if inventory.get("wood", 0) > 0:
            conditions.extend(["has_wood", "has_wood_logs"])
        if inventory.get("cobblestone", 0) > 0:
            conditions.append("has_stone")
        if any(tool in inventory for tool in ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"]):
            conditions.append("has_pickaxe")
        if "wooden_pickaxe" in inventory:
            conditions.append("has_wooden_pickaxe")
        if any(tool in inventory for tool in ["wooden_sword", "stone_sword", "iron_sword"]):
            conditions.append("has_weapon")
        if not any(tool in inventory for tool in ["wooden_pickaxe", "wooden_sword"]):
            conditions.append("no_tools")
        
        # 环境相关
        nearby = state.get("nearby_entities", [])
        if any(e.get("hostile", False) for e in nearby if isinstance(e, dict)):
            conditions.append("hostile_nearby")
        if any(e.get("type") == "creeper" for e in nearby if isinstance(e, dict)):
            conditions.append("creeper_nearby")
        if any(e.get("type") in ["cow", "pig", "sheep", "chicken"] for e in nearby if isinstance(e, dict)):
            conditions.append("animals_nearby")
        
        nearby_blocks = state.get("nearby_blocks", [])
        if "tree" in nearby_blocks or "log" in nearby_blocks:
            conditions.append("trees_nearby")
        if "stone" in nearby_blocks or "cobblestone" in nearby_blocks:
            conditions.extend(["stone_nearby", "stone_available"])
        
        # 庇护所相关
        if not state.get("has_shelter", False):
            conditions.append("no_shelter")
        
        return conditions
    
    # ========== 记忆存储 ==========
    
    def store_episode(self, 
                      summary: str,
                      lesson: str,
                      tags: List[str],
                      context: Dict[str, Any] = None,
                      outcome: str = "unknown") -> Episode:
        """存储情景记忆"""
        return self.episodic.add(
            summary=summary,
            lesson=lesson,
            tags=tags,
            context=context,
            outcome=outcome
        )
    
    def store_rule(self,
                   rule: str,
                   confidence: float = 0.5,
                   conditions: List[str] = None) -> Rule:
        """存储语义规则"""
        return self.semantic.add(
            rule=rule,
            confidence=confidence,
            conditions=conditions,
            source="learned"
        )
    
    def store_skill(self,
                    name: str,
                    purpose: str,
                    trigger_conditions: List[str],
                    steps: List[Dict[str, Any]],
                    **kwargs) -> Skill:
        """存储技能"""
        return self.skills.add(
            name=name,
            purpose=purpose,
            trigger_conditions=trigger_conditions,
            steps=steps,
            source="learned",
            **kwargs
        )
    
    # ========== 持久化 ==========
    
    def save(self, path: str = None):
        """保存所有记忆到文件"""
        save_path = path or self.storage_path
        if not save_path:
            raise ValueError("未设置存储路径")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.episodic.save(os.path.join(save_path, "episodic.json"))
        self.semantic.save(os.path.join(save_path, "semantic.json"))
        self.skills.save(os.path.join(save_path, "skills.json"))
        
        # 保存轨迹
        trajectory_data = {
            "step_counter": self.step_counter,
            "trajectory": self.get_full_trajectory()
        }
        with open(os.path.join(save_path, "trajectory.json"), 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str = None):
        """从文件加载所有记忆"""
        load_path = path or self.storage_path
        if not load_path:
            raise ValueError("未设置存储路径")
        
        episodic_file = os.path.join(load_path, "episodic.json")
        if os.path.exists(episodic_file):
            self.episodic.load(episodic_file)
        
        semantic_file = os.path.join(load_path, "semantic.json")
        if os.path.exists(semantic_file):
            self.semantic.load(semantic_file)
        
        skills_file = os.path.join(load_path, "skills.json")
        if os.path.exists(skills_file):
            self.skills.load(skills_file)
        
        trajectory_file = os.path.join(load_path, "trajectory.json")
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.step_counter = data.get("step_counter", 0)
    
    # ========== 统计信息 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "episodic_count": len(self.episodic.episodes),
            "semantic_rules_count": len(self.semantic.rules),
            "skills_count": len(self.skills.skills),
            "trajectory_length": len(self.trajectory),
            "total_steps": self.step_counter
        }
    
    def to_prompt_context(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 prompt 上下文格式"""
        memories = self.retrieve(current_state)
        return {
            "recent_trajectory": memories["recent_trajectory"],
            "episodic_memories": memories["episodic_memories"],
            "semantic_rules": memories["semantic_rules"],
            "skills": memories["skills"]
        }
