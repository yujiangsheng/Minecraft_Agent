"""
语义记忆 (Semantic Memory)

作者: Jiangsheng Yu
许可证: MIT License

存储通用规则和知识，支持条件检索和置信度更新。
预置 8 条基础生存规则，可通过反思自动新增/修改。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class Rule:
    """单条语义规则"""
    id: str
    rule: str                         # 规则内容
    confidence: float                 # 置信度 (0-1)
    source: str                       # 来源: manual/learned/evolved
    conditions: List[str] = field(default_factory=list)  # 适用条件
    created_at: str = ""
    updated_at: str = ""
    usage_count: int = 0              # 使用次数
    success_rate: float = 0.5         # 成功率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule": self.rule,
            "confidence": self.confidence,
            "source": self.source,
            "conditions": self.conditions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(
            id=data.get("id", ""),
            rule=data.get("rule", ""),
            confidence=data.get("confidence", 0.5),
            source=data.get("source", "manual"),
            conditions=data.get("conditions", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.5)
        )


class SemanticMemory:
    """语义记忆管理器"""
    
    # 预置的基础规则
    DEFAULT_RULES = [
        {
            "rule": "夜晚时如果没有安全庇护所，优先寻找或建造庇护所",
            "confidence": 0.95,
            "conditions": ["time=night", "no_shelter"]
        },
        {
            "rule": "生命值低于8时，避免主动进入战斗",
            "confidence": 0.9,
            "conditions": ["health<8"]
        },
        {
            "rule": "饥饿值低于6时，优先获取食物",
            "confidence": 0.9,
            "conditions": ["hunger<6"]
        },
        {
            "rule": "有木制工具且附近可获取石头时，优先升级到石制工具",
            "confidence": 0.85,
            "conditions": ["has_wooden_tools", "stone_available"]
        },
        {
            "rule": "敌对生物靠近且装备不足时，优先撤退而非战斗",
            "confidence": 0.9,
            "conditions": ["hostile_nearby", "low_equipment"]
        },
        {
            "rule": "采集资源前先确认撤退路线",
            "confidence": 0.8,
            "conditions": ["gathering_mode"]
        },
        {
            "rule": "下矿前准备足够的火把和食物",
            "confidence": 0.85,
            "conditions": ["before_mining"]
        },
        {
            "rule": "遇到苦力怕时保持距离，不要在狭窄空间战斗",
            "confidence": 0.95,
            "conditions": ["creeper_nearby"]
        }
    ]
    
    def __init__(self, max_rules: int = 50):
        self.max_rules = max_rules
        self.rules: Dict[str, Rule] = {}
        self._condition_index: Dict[str, List[str]] = {}  # condition -> rule_ids
        
        # 初始化默认规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        for rule_data in self.DEFAULT_RULES:
            self.add(
                rule=rule_data["rule"],
                confidence=rule_data["confidence"],
                conditions=rule_data.get("conditions", []),
                source="manual"
            )
    
    def _generate_id(self, rule: str) -> str:
        """生成规则 ID"""
        return hashlib.md5(rule.encode()).hexdigest()[:12]
    
    def add(self,
            rule: str,
            confidence: float = 0.5,
            conditions: List[str] = None,
            source: str = "learned") -> Rule:
        """添加新规则"""
        rule_id = self._generate_id(rule)
        
        # 如果规则已存在，更新置信度
        if rule_id in self.rules:
            existing = self.rules[rule_id]
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = datetime.now().isoformat()
            return existing
        
        # 如果超过最大数量，删除置信度最低的学习规则
        if len(self.rules) >= self.max_rules:
            self._evict_one()
        
        timestamp = datetime.now().isoformat()
        rule_obj = Rule(
            id=rule_id,
            rule=rule,
            confidence=confidence,
            source=source,
            conditions=conditions or [],
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self.rules[rule_id] = rule_obj
        
        # 更新条件索引
        for condition in rule_obj.conditions:
            if condition not in self._condition_index:
                self._condition_index[condition] = []
            self._condition_index[condition].append(rule_id)
        
        return rule_obj
    
    def _evict_one(self):
        """驱逐一条规则（优先驱逐学习的低置信度规则）"""
        learned_rules = [r for r in self.rules.values() if r.source == "learned"]
        
        if learned_rules:
            # 按置信度和成功率排序
            sorted_rules = sorted(
                learned_rules,
                key=lambda r: r.confidence * r.success_rate
            )
            victim = sorted_rules[0]
        else:
            # 如果没有学习的规则，删除置信度最低的
            sorted_rules = sorted(
                self.rules.values(),
                key=lambda r: r.confidence
            )
            victim = sorted_rules[0]
        
        self.remove(victim.id)
    
    def remove(self, rule_id: str):
        """删除规则"""
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        
        # 从条件索引中移除
        for condition in rule.conditions:
            if condition in self._condition_index and rule_id in self._condition_index[condition]:
                self._condition_index[condition].remove(rule_id)
        
        del self.rules[rule_id]
    
    def get(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def update_confidence(self, rule_id: str, success: bool):
        """根据使用结果更新置信度"""
        rule = self.rules.get(rule_id)
        if not rule:
            return
        
        rule.usage_count += 1
        
        # 更新成功率
        old_rate = rule.success_rate
        alpha = 0.1  # 学习率
        
        if success:
            rule.success_rate = old_rate + alpha * (1 - old_rate)
            rule.confidence = min(1.0, rule.confidence + 0.02)
        else:
            rule.success_rate = old_rate - alpha * old_rate
            rule.confidence = max(0.1, rule.confidence - 0.05)
        
        rule.updated_at = datetime.now().isoformat()
    
    def search_by_conditions(self, 
                             current_conditions: List[str],
                             min_confidence: float = 0.5) -> List[Rule]:
        """按条件搜索适用的规则"""
        matching_rules = []
        
        for rule in self.rules.values():
            if rule.confidence < min_confidence:
                continue
            
            # 检查规则条件是否匹配当前条件
            if not rule.conditions:
                # 无条件规则总是适用
                matching_rules.append(rule)
            else:
                # 检查是否所有条件都满足
                matched = all(
                    any(self._condition_matches(rc, cc) for cc in current_conditions)
                    for rc in rule.conditions
                )
                if matched:
                    matching_rules.append(rule)
        
        # 按置信度排序
        matching_rules.sort(key=lambda r: r.confidence, reverse=True)
        return matching_rules
    
    def _condition_matches(self, rule_condition: str, current_condition: str) -> bool:
        """检查条件是否匹配"""
        # 简单的字符串匹配
        return rule_condition.lower() in current_condition.lower() or \
               current_condition.lower() in rule_condition.lower()
    
    def get_all(self) -> List[Rule]:
        """获取所有规则"""
        return list(self.rules.values())
    
    def get_high_confidence(self, min_confidence: float = 0.7) -> List[Rule]:
        """获取高置信度规则"""
        return [r for r in self.rules.values() if r.confidence >= min_confidence]
    
    def revise_rule(self, old_rule_id: str, new_rule: str, reason: str = ""):
        """修订规则"""
        old = self.rules.get(old_rule_id)
        if not old:
            return None
        
        # 降低旧规则置信度
        old.confidence *= 0.5
        
        # 添加新规则
        new_rule_obj = self.add(
            rule=new_rule,
            confidence=max(0.6, old.confidence),
            conditions=old.conditions,
            source="evolved"
        )
        
        return new_rule_obj
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            "rules": [r.to_dict() for r in self.rules.values()],
            "max_rules": self.max_rules
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.max_rules = data.get("max_rules", 50)
        self.rules.clear()
        self._condition_index.clear()
        
        for rule_data in data.get("rules", []):
            rule = Rule.from_dict(rule_data)
            self.rules[rule.id] = rule
            
            for condition in rule.conditions:
                if condition not in self._condition_index:
                    self._condition_index[condition] = []
                self._condition_index[condition].append(rule.id)
    
    def to_prompt_format(self, rules: List[Rule] = None) -> List[Dict[str, Any]]:
        """转换为 prompt 格式"""
        if rules is None:
            rules = self.get_high_confidence()
        
        return [
            {
                "id": r.id,
                "rule": r.rule,
                "confidence": r.confidence,
                "conditions": r.conditions
            }
            for r in rules
        ]
