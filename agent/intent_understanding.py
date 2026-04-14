"""
意图理解 (Intent Understanding)

作者: Jiangsheng Yu
许可证: MIT License

将用户的自然语言任务（如 "垒墙"）分解为可执行的动作序列，
并将分解结果作为操作语义存入技能库和语义记忆，供后续复用。

核心机制 — 前置条件驱动的分阶段执行：
  1. 接收自然语言意图 → 分解为结构化步骤 + 结构化前置条件
  2. 每个决策周期，根据当前 env_state 评估前置条件：
     - 条件未满足 → 生成 "准备阶段" 动作计划（去补齐缺失资源/工具）
     - 条件全部满足 → 生成 "执行阶段" 动作计划（开始完成主任务）
  3. 将分阶段执行计划注入 LLM 上下文，引导智能体自动先补后做
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import json
import hashlib
import logging
import re

from config import AgentConfig, DEFAULT_CONFIG
from memory import MemoryManager, Skill
from utils import LLMClient, LLMResponse
from prompts import ACTION_REGISTRY

logger = logging.getLogger("IntentUnderstanding")


@dataclass
class IntentDecomposition:
    """意图分解结果"""
    intent: str                                   # 原始意图文本
    summary: str                                  # 一句话概要
    preconditions: List[str]                      # 前置条件
    steps: List[Dict[str, Any]]                   # 动作步骤
    parameters: Dict[str, Any]                    # 可调参数（如数量、方向）
    stop_conditions: List[str]                    # 完成/终止条件
    failure_recovery: List[Dict[str, Any]]        # 失败恢复策略
    sub_intents: List[str] = field(default_factory=list)  # 递归子意图

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "summary": self.summary,
            "preconditions": self.preconditions,
            "steps": [s for s in self.steps],
            "parameters": self.parameters,
            "stop_conditions": self.stop_conditions,
            "failure_recovery": [r for r in self.failure_recovery],
            "sub_intents": self.sub_intents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentDecomposition":
        return cls(
            intent=data.get("intent", ""),
            summary=data.get("summary", ""),
            preconditions=data.get("preconditions", []),
            steps=data.get("steps", []),
            parameters=data.get("parameters", {}),
            stop_conditions=data.get("stop_conditions", []),
            failure_recovery=data.get("failure_recovery", []),
            sub_intents=data.get("sub_intents", []),
        )


# ════════════════════════════════════════════════════════════════
#  结构化前置条件 & 执行计划
# ════════════════════════════════════════════════════════════════

@dataclass
class StructuredPrecondition:
    """一条可自动检查的前置条件"""
    description: str                              # 人类可读描述
    check_type: str                               # inventory_min / has_tool / nearby_block
    item: str = ""                                # 物品/方块名称（支持 | 分隔的多选项）
    min_count: int = 1                            # inventory_min 的最低数量
    # 当条件不满足时的补救动作序列
    resolution: List[Dict[str, Any]] = field(default_factory=list)

    def evaluate(self, env_state: Dict[str, Any]) -> Tuple[bool, str]:
        """根据 env_state 评估此条件是否满足

        Returns:
            (satisfied, detail)  detail 描述当前值
        """
        inventory = env_state.get("inventory", {})
        nearby_blocks = env_state.get("nearby_blocks", [])

        if self.check_type == "inventory_min":
            alternatives = [a.strip() for a in self.item.split("|")]
            total = sum(inventory.get(alt, 0) for alt in alternatives)
            if total >= self.min_count:
                return True, f"已有 {total}/{self.min_count}"
            return False, f"缺少 {self.item}（当前 {total}/{self.min_count}）"

        elif self.check_type == "has_tool":
            alternatives = [a.strip() for a in self.item.split("|")]
            for alt in alternatives:
                if alt in inventory:
                    return True, f"拥有 {alt}"
            return False, f"缺少工具 {self.item}"

        elif self.check_type == "nearby_block":
            alternatives = [a.strip() for a in self.item.split("|")]
            for alt in alternatives:
                if alt in nearby_blocks:
                    return True, f"附近有 {alt}"
            return False, f"附近没有 {self.item}"

        return True, "未知检查类型，默认满足"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "check_type": self.check_type,
            "item": self.item,
            "min_count": self.min_count,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredPrecondition":
        return cls(
            description=data.get("description", ""),
            check_type=data.get("check_type", "inventory_min"),
            item=data.get("item", ""),
            min_count=data.get("min_count", 1),
            resolution=data.get("resolution", []),
        )


@dataclass
class TaskExecutionPlan:
    """分阶段执行计划 — 准备阶段 + 执行阶段"""
    intent: str
    summary: str
    phase: str                                    # "preparing" | "ready" | "executing"
    all_conditions: List[Dict[str, Any]]          # 所有前置条件及其检查结果
    unmet_conditions: List[Dict[str, Any]]        # 未满足的条件
    prerequisite_actions: List[Dict[str, Any]]    # 准备阶段的动作序列
    main_task_actions: List[Dict[str, Any]]       # 主任务的动作序列
    progress_hint: str                            # 人类可读的进度提示

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "summary": self.summary,
            "phase": self.phase,
            "all_conditions": self.all_conditions,
            "unmet_conditions": self.unmet_conditions,
            "prerequisite_actions": self.prerequisite_actions,
            "main_task_actions": self.main_task_actions,
            "progress_hint": self.progress_hint,
        }


# ════════════════════════════════════════════════════════════════
#  意图理解 Prompt
# ════════════════════════════════════════════════════════════════

INTENT_UNDERSTANDING_PROMPT = f"""你是 Luanti（类 Minecraft 沙盒游戏）中的任务分解专家。

你的职责是将用户的自然语言意图拆解为具体的、可执行的动作序列。
拆解必须考虑游戏的物理规则和资源限制。

【可用动作列表】
{ACTION_REGISTRY}

【分解原则】
1. 先识别意图的核心目标（做什么？为什么？）。
2. 列出完成目标所需的前置条件（材料、工具、环境）。
3. 将目标拆解为有序的原子动作步骤，每一步必须对应上方动作列表中的一个动作。
4. 如果某一步本身是复合操作（如 "收集 20 块石头"），递归拆解为子意图。
5. 注明可调参数（数量、方向、材料类型等）及其默认值。
6. 给出停止条件和失败恢复策略。

【示例】
意图: "垒一堵墙"
分解:
- 核心: 在地面上用方块摆成一条直线，然后逐层向上堆叠。
- 前置条件: 库存中有足够方块（如 cobble ≥ 20）。
- 步骤:
  1. 确定墙的起点和长度方向
  2. 沿直线逐格放置方块 (place_block)
  3. 移到已完成行的一端，向上搭一格 (tower_up 或 jump + place_block)
  4. 在上层重复放置直线
  5. 重复到目标高度

【输出】严格 JSON，无其他文字：
{{
  "intent": "原始意图",
  "summary": "一句话描述操作语义",
  "preconditions": ["前置条件1", "前置条件2"],
  "steps": [
    {{"step": 1, "description": "描述", "action": "动作名", "args": {{}}, "note": "可选备注"}},
    ...
  ],
  "parameters": {{
    "参数名": {{"default": "默认值", "description": "说明"}}
  }},
  "stop_conditions": ["完成条件1"],
  "failure_recovery": [
    {{"condition": "失败情况", "action": "恢复动作", "args": {{}}}}
  ],
  "sub_intents": ["需要递归分解的子任务（如有）"]
}}"""


class IntentUnderstanding:
    """意图理解引擎"""

    # 预置的常见意图操作语义（即使没有 LLM 也可以工作）
    # 每个意图额外包含 structured_preconditions 字段，用于自动检查和补齐
    BUILTIN_INTENTS: Dict[str, Dict[str, Any]] = {
        "垒墙": {
            "summary": "用方块沿一条直线逐层向上堆叠，形成一面墙",
            "preconditions": ["库存中有方块（cobble/wood ≥ 20）"],
            "structured_preconditions": [
                {
                    "description": "建筑方块 ≥ 20",
                    "check_type": "inventory_min",
                    "item": "cobble|wood",
                    "min_count": 20,
                    "resolution": [
                        {"step": 1, "description": "寻找石头资源", "action": "find_resource", "args": {"resource": "stone"}},
                        {"step": 2, "description": "采集石块", "action": "dig", "args": {"target_type": "stone"}},
                        {"step": 3, "description": "拾取掉落物", "action": "pickup_item", "args": {}},
                    ],
                },
            ],
            "steps": [
                {"step": 1, "description": "确定墙体起点，面向建造方向", "action": "look_at", "args": {"target": "build_direction"}},
                {"step": 2, "description": "沿直线放置第一层方块", "action": "place_block", "args": {"node": "cobble"}, "note": "重复 length 次，每次前移一格"},
                {"step": 3, "description": "回到起点，向上跳一格", "action": "jump", "args": {}},
                {"step": 4, "description": "放置第二层方块", "action": "place_block", "args": {"node": "cobble"}, "note": "重复放置"},
                {"step": 5, "description": "重复步骤3-4直到目标高度", "action": "tower_up", "args": {"height": 1}},
            ],
            "parameters": {
                "length": {"default": 5, "description": "墙的长度（方块数）"},
                "height": {"default": 3, "description": "墙的高度（层数）"},
                "material": {"default": "cobble", "description": "建筑材料"},
            },
            "stop_conditions": ["墙体建造完成（达到目标高度和长度）", "材料耗尽"],
            "failure_recovery": [
                {"condition": "材料不足", "action": "find_resource", "args": {"resource": "stone"}},
                {"condition": "位置被阻挡", "action": "dig", "args": {}},
            ],
            "sub_intents": [],
        },
        "建庇护所": {
            "summary": "在当前位置建造一个封闭的3×3×3方块庇护所",
            "preconditions": ["库存中有方块 ≥ 30"],
            "structured_preconditions": [
                {
                    "description": "建筑方块 ≥ 30",
                    "check_type": "inventory_min",
                    "item": "cobble|wood|stone",
                    "min_count": 30,
                    "resolution": [
                        {"step": 1, "description": "寻找石头资源", "action": "find_resource", "args": {"resource": "stone"}},
                        {"step": 2, "description": "采集石块", "action": "dig", "args": {"target_type": "stone"}},
                        {"step": 3, "description": "拾取掉落物", "action": "pickup_item", "args": {}},
                    ],
                },
            ],
            "steps": [
                {"step": 1, "description": "建造庇护所", "action": "build_shelter", "args": {}},
                {"step": 2, "description": "放置火把照明", "action": "light_area", "args": {}},
            ],
            "parameters": {},
            "stop_conditions": ["庇护所建成"],
            "failure_recovery": [
                {"condition": "材料不足", "action": "find_resource", "args": {"resource": "stone"}},
            ],
            "sub_intents": [],
        },
        "挖隧道": {
            "summary": "向前挖掘一条2格高的通道",
            "preconditions": ["有镐（wooden_pickaxe 以上）"],
            "structured_preconditions": [
                {
                    "description": "拥有镐工具",
                    "check_type": "has_tool",
                    "item": "wooden_pickaxe|stone_pickaxe|steel_pickaxe|diamond_pickaxe",
                    "min_count": 1,
                    "resolution": [
                        {"step": 1, "description": "寻找树木", "action": "find_resource", "args": {"resource": "tree"}},
                        {"step": 2, "description": "采集木头", "action": "dig", "args": {"target_type": "tree"}},
                        {"step": 3, "description": "拾取木头", "action": "pickup_item", "args": {}},
                        {"step": 4, "description": "合成木镐", "action": "craft_tool", "args": {"item": "default:pick_wood"}},
                    ],
                },
            ],
            "steps": [
                {"step": 1, "description": "装备镐", "action": "equip", "args": {"item": "stone_pickaxe"}},
                {"step": 2, "description": "向前挖掘通道", "action": "tunnel", "args": {"length": 5}},
                {"step": 3, "description": "拾取挖出的材料", "action": "pickup_item", "args": {}},
                {"step": 4, "description": "放置火把照明", "action": "light_area", "args": {}},
            ],
            "parameters": {
                "length": {"default": 5, "description": "隧道长度"},
            },
            "stop_conditions": ["达到目标长度", "遇到空洞或水"],
            "failure_recovery": [
                {"condition": "镐损坏", "action": "craft_tool", "args": {"item": "default:pick_stone"}},
            ],
            "sub_intents": [],
        },
        "搭桥": {
            "summary": "向前搭建方块桥梁跨越间隙",
            "preconditions": ["库存中有方块 ≥ 10"],
            "structured_preconditions": [
                {
                    "description": "建筑方块 ≥ 10",
                    "check_type": "inventory_min",
                    "item": "cobble|wood",
                    "min_count": 10,
                    "resolution": [
                        {"step": 1, "description": "寻找石头资源", "action": "find_resource", "args": {"resource": "stone"}},
                        {"step": 2, "description": "采集石块", "action": "dig", "args": {"target_type": "stone"}},
                        {"step": 3, "description": "拾取掉落物", "action": "pickup_item", "args": {}},
                    ],
                },
            ],
            "steps": [
                {"step": 1, "description": "开启潜行防止掉落", "action": "sneak", "args": {"enable": True}},
                {"step": 2, "description": "向前搭桥", "action": "bridge", "args": {"length": 5, "block": "cobble"}},
                {"step": 3, "description": "关闭潜行", "action": "sneak", "args": {"enable": False}},
            ],
            "parameters": {
                "length": {"default": 5, "description": "桥的长度"},
                "block": {"default": "cobble", "description": "建筑材料"},
            },
            "stop_conditions": ["到达对面", "材料耗尽"],
            "failure_recovery": [
                {"condition": "掉落", "action": "jump", "args": {}},
            ],
            "sub_intents": [],
        },
    }

    def __init__(self,
                 config: AgentConfig = None,
                 llm_client: LLMClient = None,
                 memory_manager: MemoryManager = None):
        self.config = config or DEFAULT_CONFIG
        self.llm = llm_client
        self.memory = memory_manager

        # 缓存：intent_text → IntentDecomposition
        self._cache: Dict[str, IntentDecomposition] = {}

    # ── 主入口 ──

    def understand(self, intent: str, env_state: Dict[str, Any] = None) -> IntentDecomposition:
        """理解并分解一个自然语言意图

        Args:
            intent: 自然语言任务描述
            env_state: 当前环境状态（可选，帮助 LLM 考虑当前资源）

        Returns:
            IntentDecomposition 结构化分解结果
        """
        intent_key = intent.strip()

        # 1. 检查缓存
        if intent_key in self._cache:
            logger.info(f"[意图缓存命中] {intent_key}")
            return self._cache[intent_key]

        # 2. 在技能库中搜索已有的操作语义
        existing = self._search_existing_knowledge(intent_key)
        if existing:
            logger.info(f"[已有操作语义] {intent_key} → 技能 {existing.name}")
            decomp = self._skill_to_decomposition(intent_key, existing)
            self._cache[intent_key] = decomp
            return decomp

        # 3. 检查内置意图
        builtin = self._match_builtin(intent_key)
        if builtin:
            logger.info(f"[内置意图匹配] {intent_key}")
            # 过滤掉 IntentDecomposition 不接受的字段
            decomp_fields = {k: v for k, v in builtin.items() if k != "structured_preconditions"}
            decomp = IntentDecomposition(intent=intent_key, **decomp_fields)
            self._store_as_knowledge(decomp)
            self._cache[intent_key] = decomp
            return decomp

        # 4. 调用 LLM 分解
        decomp = self._llm_decompose(intent_key, env_state)
        if decomp:
            self._store_as_knowledge(decomp)
            self._cache[intent_key] = decomp
            return decomp

        # 5. 回退：生成一个最小可用的分解
        logger.warning(f"[意图分解回退] {intent_key}")
        decomp = self._fallback_decomposition(intent_key)
        self._cache[intent_key] = decomp
        return decomp

    def understand_and_get_actions(self, intent: str, env_state: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """理解意图并直接返回可执行的动作列表

        Args:
            intent: 自然语言任务描述
            env_state: 当前环境状态

        Returns:
            动作列表 [{"action": "...", "args": {...}}, ...]
        """
        decomp = self.understand(intent, env_state)
        return [{"action": s["action"], "args": s.get("args", {})} for s in decomp.steps]

    # ── 知识检索 ──

    def _search_existing_knowledge(self, intent: str) -> Optional[Skill]:
        """在技能库中搜索与意图匹配的已有技能"""
        if not self.memory:
            return None

        # 按名称精确匹配（含 intent_ 前缀的技能）
        intent_skill_name = f"intent_{self._normalize_name(intent)}"
        skill = self.memory.skills.get_by_name(intent_skill_name)
        if skill:
            return skill

        # 按 purpose 模糊匹配
        for s in self.memory.skills.get_all():
            if intent in s.purpose or s.purpose in intent:
                return s

        return None

    def _match_builtin(self, intent: str) -> Optional[Dict[str, Any]]:
        """匹配内置意图（支持模糊匹配）"""
        # 精确匹配
        if intent in self.BUILTIN_INTENTS:
            return self.BUILTIN_INTENTS[intent]

        # 子串匹配
        for key, value in self.BUILTIN_INTENTS.items():
            if key in intent or intent in key:
                return value

        # 核心词匹配：去除常见虚词后比较
        import re
        stop_words = {"一个", "一条", "一堵", "一面", "一些", "一座",
                      "的", "了", "把", "在", "和", "去", "来", "到"}
        def core_chars(text: str) -> str:
            t = text
            for w in stop_words:
                t = t.replace(w, "")
            return re.sub(r'\s+', '', t)

        intent_core = core_chars(intent)
        best_match = None
        best_overlap = 0
        for key, value in self.BUILTIN_INTENTS.items():
            key_core = core_chars(key)
            # 检查核心字符重叠度
            if key_core in intent_core or intent_core in key_core:
                return value
            overlap = sum(1 for c in key_core if c in intent_core)
            ratio = overlap / max(len(key_core), 1)
            if ratio > best_overlap and ratio >= 0.5:
                best_overlap = ratio
                best_match = value

        return best_match

    # ── LLM 分解 ──

    def _llm_decompose(self, intent: str, env_state: Dict[str, Any] = None) -> Optional[IntentDecomposition]:
        """调用 LLM 将意图分解为动作序列"""
        if not self.llm:
            return None

        context: Dict[str, Any] = {
            "intent": intent,
        }

        # 提供当前环境信息以帮助 LLM 考虑资源限制
        if env_state:
            context["current_inventory"] = env_state.get("inventory", {})
            context["nearby_blocks"] = env_state.get("nearby_blocks", [])
            context["current_position"] = env_state.get("position", {})

        # 提供已有技能列表避免重复
        if self.memory:
            context["existing_skills"] = [
                s.name for s in self.memory.skills.get_all()
            ]

        response = self.llm.generate_decision(INTENT_UNDERSTANDING_PROMPT, context)

        if response.success and response.parsed:
            try:
                decomp = IntentDecomposition.from_dict(response.parsed)
                if decomp.steps:
                    logger.info(f"[LLM 意图分解成功] {intent} → {len(decomp.steps)} 步")
                    return decomp
            except Exception as e:
                logger.error(f"[LLM 意图分解解析失败] {e}")

        return None

    # ── 知识存储 ──

    def _store_as_knowledge(self, decomp: IntentDecomposition):
        """将分解结果存储为技能和语义规则"""
        if not self.memory:
            return

        # 1. 存为技能
        skill_name = f"intent_{self._normalize_name(decomp.intent)}"
        trigger_conditions = [f"user_intent={decomp.intent}"]
        for pre in decomp.preconditions:
            trigger_conditions.append(pre)

        action_steps = [
            {"action": s["action"], "args": s.get("args", {}), "description": s.get("description", "")}
            for s in decomp.steps
        ]

        self.memory.store_skill(
            name=skill_name,
            purpose=decomp.summary or decomp.intent,
            trigger_conditions=trigger_conditions,
            steps=action_steps,
            preconditions=decomp.preconditions,
            stop_conditions=decomp.stop_conditions,
            failure_recovery=decomp.failure_recovery,
            metrics={"intent": decomp.intent},
            priority=0.7,
        )
        logger.info(f"[操作语义→技能] {skill_name}")

        # 2. 存为语义规则
        params_desc = ""
        if decomp.parameters:
            params_desc = "（参数: " + ", ".join(
                f"{k}={v.get('default', '?')}" for k, v in decomp.parameters.items()
            ) + "）"

        rule_text = f"「{decomp.intent}」的操作语义: {decomp.summary}{params_desc}"
        self.memory.store_rule(
            rule=rule_text,
            confidence=0.75,
            conditions=[f"user_intent={decomp.intent}"],
        )
        logger.info(f"[操作语义→规则] {rule_text}")

    # ── 工具方法 ──

    def _skill_to_decomposition(self, intent: str, skill: Skill) -> IntentDecomposition:
        """将技能转换回 IntentDecomposition"""
        return IntentDecomposition(
            intent=intent,
            summary=skill.purpose,
            preconditions=skill.preconditions,
            steps=[
                {
                    "step": i + 1,
                    "description": s.get("description", s.get("action", "")),
                    "action": s["action"],
                    "args": s.get("args", {}),
                }
                for i, s in enumerate(skill.steps)
            ],
            parameters=skill.metrics if "intent" not in skill.metrics else {},
            stop_conditions=skill.stop_conditions,
            failure_recovery=skill.failure_recovery,
        )

    def _fallback_decomposition(self, intent: str) -> IntentDecomposition:
        """回退分解：无法理解意图时返回安全的探索计划"""
        return IntentDecomposition(
            intent=intent,
            summary=f"尝试完成: {intent}（未找到操作语义，使用探索策略）",
            preconditions=[],
            steps=[
                {"step": 1, "description": "环顾四周评估环境", "action": "look_around", "args": {}},
                {"step": 2, "description": "探索寻找相关资源", "action": "find_resource", "args": {"resource": "tree"}},
                {"step": 3, "description": "收集基础材料", "action": "dig", "args": {}},
                {"step": 4, "description": "拾取物品", "action": "pickup_item", "args": {}},
            ],
            parameters={},
            stop_conditions=["找到相关资源或线索"],
            failure_recovery=[],
        )

    @staticmethod
    def _normalize_name(text: str) -> str:
        """将意图文本归一化为技能名称"""
        # 去除空白，用下划线连接
        import re
        name = re.sub(r'\s+', '_', text.strip())
        # 截断过长名称
        if len(name) > 40:
            name = name[:40]
        return name

    def get_known_intents(self) -> List[Dict[str, Any]]:
        """获取所有已知的意图操作语义（内置 + 已学习）"""
        intents = []

        # 内置意图
        for key, val in self.BUILTIN_INTENTS.items():
            intents.append({
                "intent": key,
                "summary": val["summary"],
                "source": "builtin",
                "steps_count": len(val["steps"]),
            })

        # 从技能库中查找 intent_ 前缀的技能
        if self.memory:
            for skill in self.memory.skills.get_all():
                if skill.name.startswith("intent_"):
                    original_intent = skill.metrics.get("intent", skill.name.replace("intent_", ""))
                    # 避免与内置重复
                    if original_intent not in self.BUILTIN_INTENTS:
                        intents.append({
                            "intent": original_intent,
                            "summary": skill.purpose,
                            "source": "learned",
                            "steps_count": len(skill.steps),
                            "usage_count": skill.usage_count,
                            "success_rate": skill.success_rate,
                        })

        return intents

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        known = self.get_known_intents()
        return {
            "builtin_intents": len(self.BUILTIN_INTENTS),
            "learned_intents": sum(1 for i in known if i["source"] == "learned"),
            "cached_intents": len(self._cache),
            "total_known": len(known),
        }

    # ════════════════════════════════════════════════════════════════
    #  前置条件评估 & 分阶段执行计划
    # ════════════════════════════════════════════════════════════════

    def evaluate_readiness(
        self, decomp: IntentDecomposition, env_state: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """评估任务分解的所有前置条件是否满足

        Returns:
            (all_conditions, unmet_conditions)
            每个元素 = {"description", "check_type", "item", "satisfied", "detail", "resolution"}
        """
        # 从 BUILTIN_INTENTS 或 LLM 分解中获取结构化前置条件
        structured = self._get_structured_preconditions(decomp)

        all_conds: List[Dict[str, Any]] = []
        unmet: List[Dict[str, Any]] = []

        for sp in structured:
            satisfied, detail = sp.evaluate(env_state)
            entry = {
                "description": sp.description,
                "check_type": sp.check_type,
                "item": sp.item,
                "min_count": sp.min_count,
                "satisfied": satisfied,
                "detail": detail,
                "resolution": sp.resolution,
            }
            all_conds.append(entry)
            if not satisfied:
                unmet.append(entry)

        return all_conds, unmet

    def plan_execution(
        self, intent: str, env_state: Dict[str, Any]
    ) -> TaskExecutionPlan:
        """为意图生成分阶段执行计划

        1. 理解意图 → 分解
        2. 评估前置条件 → 区分满足/未满足
        3. 若有未满足条件 → phase="preparing"，生成补齐动作
        4. 若全部满足 → phase="ready"，直接给出主任务步骤

        Returns:
            TaskExecutionPlan
        """
        decomp = self.understand(intent, env_state)
        all_conds, unmet = self.evaluate_readiness(decomp, env_state)

        main_actions = [
            {"action": s["action"], "args": s.get("args", {}), "description": s.get("description", "")}
            for s in decomp.steps
        ]

        if unmet:
            # ── 准备阶段：合并所有未满足条件的补救动作 ──
            prereq_actions: List[Dict[str, Any]] = []
            hints = []
            for cond in unmet:
                hints.append(cond["detail"])
                for res_step in cond.get("resolution", []):
                    prereq_actions.append({
                        "action": res_step["action"],
                        "args": res_step.get("args", {}),
                        "description": res_step.get("description", ""),
                    })

            progress_hint = "准备阶段 — " + "；".join(hints)
            phase = "preparing"
        else:
            prereq_actions = []
            progress_hint = "条件就绪 — 可以开始执行主任务"
            phase = "ready"

        return TaskExecutionPlan(
            intent=intent,
            summary=decomp.summary,
            phase=phase,
            all_conditions=all_conds,
            unmet_conditions=unmet,
            prerequisite_actions=prereq_actions,
            main_task_actions=main_actions,
            progress_hint=progress_hint,
        )

    def _get_structured_preconditions(
        self, decomp: IntentDecomposition
    ) -> List[StructuredPrecondition]:
        """从分解结果中提取结构化前置条件

        优先使用 BUILTIN_INTENTS 中定义的 structured_preconditions，
        若没有则尝试从文本前置条件中解析。
        """
        # 1. 检查内置意图是否有 structured_preconditions
        builtin = self._match_builtin(decomp.intent)
        if builtin and "structured_preconditions" in builtin:
            return [
                StructuredPrecondition.from_dict(sp)
                for sp in builtin["structured_preconditions"]
            ]

        # 2. 尝试从文本前置条件中启发式解析
        results: List[StructuredPrecondition] = []
        for text in decomp.preconditions:
            sp = self._parse_text_precondition(text)
            if sp:
                results.append(sp)

        return results

    @staticmethod
    def _parse_text_precondition(text: str) -> Optional[StructuredPrecondition]:
        """从文本描述中启发式解析出 StructuredPrecondition

        支持模式:
            "库存中有X ≥ N"   → inventory_min
            "有X（Y以上）"     → has_tool
        """
        # 模式1: 库存数量  "cobble ≥ 20"  "方块 >= 30"
        m = re.search(r'(\w[\w/|]+)\s*[≥>=]+\s*(\d+)', text)
        if m:
            item = m.group(1)
            count = int(m.group(2))
            return StructuredPrecondition(
                description=text,
                check_type="inventory_min",
                item=item,
                min_count=count,
                resolution=[
                    {"step": 1, "description": f"寻找 {item}", "action": "find_resource", "args": {"resource": item}},
                    {"step": 2, "description": "采集", "action": "dig", "args": {}},
                    {"step": 3, "description": "拾取", "action": "pickup_item", "args": {}},
                ],
            )

        # 模式2: 工具  "有镐"  "pickaxe"
        tool_keywords = ["镐", "斧", "剑", "锄", "铲", "pickaxe", "axe", "sword", "shovel", "hoe"]
        for kw in tool_keywords:
            if kw in text:
                return StructuredPrecondition(
                    description=text,
                    check_type="has_tool",
                    item=f"wooden_{kw}|stone_{kw}|steel_{kw}" if kw in ("pickaxe", "axe", "sword", "shovel", "hoe") else kw,
                    min_count=1,
                    resolution=[
                        {"step": 1, "description": "寻找树木", "action": "find_resource", "args": {"resource": "tree"}},
                        {"step": 2, "description": "采集木头", "action": "dig", "args": {"target_type": "tree"}},
                        {"step": 3, "description": "拾取", "action": "pickup_item", "args": {}},
                        {"step": 4, "description": f"合成工具", "action": "craft_tool", "args": {}},
                    ],
                )

        return None
