"""
Minecraft/Luanti 智能体 - 多层 Prompt 流水线系统

作者: Jiangsheng Yu
许可证: MIT License

6 层 Prompt 模板，构成完整的认知流水线：
  ┌─────────────────────────────────────────────────────┐
  │  Layer 1: Agent Core     — 感知评估 → 模式/风险判定  │
  │  Layer 2: Planner        — 优先级 → 技能选择 → 计划  │
  │  Layer 3: Quick Reflect  — 在线偏差检测 → 即时修正    │
  │  Layer 4: Long Reflect   — 整局复盘 → 规则/技能更新  │
  │  Layer 5: Skill Builder  — 从经验中发明可复用技能     │
  │  Layer 6: Evolution      — 多版本配置演化优化         │
  └─────────────────────────────────────────────────────┘

运行时决策使用 decision_pipeline_prompt（融合 Layer 1+2），
在单次 LLM 调用内完成"评估 → 规划 → 输出动作"的三阶段推理。
"""

from typing import Dict, Any, List, Optional
import json


# ════════════════════════════════════════════════════════════════
#  动作注册表（所有 Prompt 共享的动作列表）
# ════════════════════════════════════════════════════════════════

ACTION_REGISTRY = """
  ═══ 基础移动 ═══
  - explore         : 向随机方向移动探索环境
  - move_to         : 移动到指定坐标 {args: {x, y, z}}
  - jump            : 跳跃（翻越障碍或避敌）
  - retreat         : 向后逃跑撤退
  - flee_from       : 远离指定方向高速逃跑
  - swim            : 在水中游泳移动（含上浮分量）
  - sneak           : 开启/关闭潜行模式 {args: {enable: true/false}}
  - sprint          : 开启/关闭冲刺加速 {args: {enable: true/false, speed: 1.5}}

  ═══ 视角控制 ═══
  - look_around     : 原地观察（不移动，刷新感知信息）
  - look_at         : 注视指定位置或方向 {args: {target: {x,y,z}} 或 {yaw, pitch}}

  ═══ 采集/挖掘 ═══
  - gather_wood     : 砍伐附近的树木获取木材（前提：附近有树）
  - mine_stone      : 开采附近的石头（前提：附近有石头）
  - mine_ore        : 开采矿石 {args: {ore_type: "iron_ore|coal|copper|gold|diamond|mese"}}
  - dig             : 挖掘面前的方块或指定类型方块 {args: {target_type: "tree|stone|ore"}}
  - dig_down        : 向下挖掘脚下方块 {args: {depth: 1-5}}
  - dig_up          : 向上挖掘头顶方块 {args: {height: 1-3}}
  - tunnel          : 向前挖掘2格高通道 {args: {length: 1-8}}

  ═══ 放置/建造 ═══
  - place_block     : 放置方块到指定坐标 {args: {node: "cobble|wood|...", target: {x,y,z}}}
  - place_at        : 智能相对放置 {args: {node: "cobble", direction: "front|back|left|right|above|below"}}
  - build_shelter   : 在当前位置建造3×3×3庇护所
  - bridge          : 向前搭桥跨越间隙 {args: {length: 1-10, block: "cobble"}}
  - tower_up        : 垂直向上搭建柱子 {args: {height: 1-10, block: "cobble"}}
  - light_area      : 在当前位置放置火把照明

  ═══ 物品管理 ═══
  - equip           : 装备物品到手持栏 {args: {item: "stone_pickaxe|stone_sword|..."}}
  - drop_item       : 丢弃物品到地面 {args: {item: "dirt", count: 1}}
  - set_hotbar      : 切换手持栏活动槽位 {args: {slot: 1-32}}
  - sort_inventory  : 整理库存（合并堆叠、按名称排序）

  ═══ 合成/冶炼 ═══
  - craft_tool      : 合成工具 {args: {item: "default:pick_wood|default:sword_stone|..."}}
  - craft_item      : 合成物品 {args: {item: "default:stick|default:wood|..."}}
  - smelt           : 使用熔炉冶炼 {args: {item: "default:iron_lump", fuel: "default:coal_lump"}}
  - check_recipe    : 查询物品是否可合成及所需材料 {args: {item: "default:pick_stone"}}

  ═══ 进食/战斗 ═══
  - eat_food        : 进食恢复生命值（自动搜索库存中食物）
  - attack          : 攻击最近的敌对实体
  - attack_entity   : 攻击指定类型实体 {args: {target: "zombie|skeleton|spider|..."}}
  - punch_node      : 敲击/打击一个方块 {args: {target: {x,y,z}}}

  ═══ 容器交互 ═══
  - deposit_item    : 将物品放入附近箱子 {args: {item: "default:cobble"}}
  - take_from_container : 从附近箱子取出物品 {args: {item: "default:iron_lump", count: 1}}
  - use_node        : 使用/交互节点（开门、操作熔炉、打开箱子等） {args: {target: {x,y,z}}}

  ═══ 搜索/感知 ═══
  - find_resource   : 搜索附近特定资源并移向目标 {args: {resource: "tree|stone|iron_ore|coal|water|food"}}
  - pickup_item     : 移向并拾取附近掉落物 {args: {item: "可选过滤名"}}

  ═══ 农业 ═══
  - farm_plant      : 在耕地上种植种子 {args: {seed: "farming:seed_wheat", count: 1}}
  - farm_harvest    : 收获附近成熟的农作物 {args: {radius: 5}}

  ═══ 等待 ═══
  - wait            : 原地等待一段时间 {args: {duration: 秒}}
"""


class PromptTemplates:
    """多层 Prompt 流水线管理器"""

    # ────────────────────────────────────────────────────
    #  Layer 1: Agent Core — 感知 + 风险 + 模式决定
    # ────────────────────────────────────────────────────
    @staticmethod
    def agent_core_prompt() -> str:
        return """你是一个在 Minecraft 生存模式中行动的自主智能体。

你的总目标不是盲目执行当前动作，而是长期生存、持续积累资源、提升科技树、构建安全基地，并通过经验不断改进自己的策略。

你必须同时遵守以下原则：

1. 生存优先于探索；探索优先于无意义停留。
2. 遇到风险时，优先评估撤退、掩体、补给，而不是硬拼。
3. 你必须使用记忆：
   - recent_trajectory 代表近期上下文
   - episodic_memories 代表相似场景下的过去经验
   - semantic_rules 代表通用规则
   - skills 代表可调用技能
4. 你不能重复已经被证明低效或危险的行为，除非当前上下文明确不同。
5. 你的决策必须解释"为什么现在做这件事"。
6. 如果信息不足，不要假装确定；请选择最稳健的行动。
7. 如果夜晚、低生命值、食物不足、武器不足、附近存在敌对生物，则默认进入保守模式。
8. 你要追求长期改进：若当前失败可能暴露策略缺陷，请显式标记以便后续反思。

请基于输入输出严格的 JSON，不要输出额外文字。

输出格式：
{
  "mode": "survive|gather|craft|explore|combat|retreat|build",
  "goal": "当前主目标",
  "subgoal": "下一子目标",
  "reason": "做出该决策的简洁原因",
  "risk_level": "low|medium|high",
  "use_memory": [
    {"memory_id": "...", "why_relevant": "..."}
  ],
  "action_plan": [
    {"action": "动作名", "args": {...}}
  ],
  "stop_condition": "何时停止当前计划并重新规划",
  "reflection_flag": "none|minor|major"
}"""

    # ────────────────────────────────────────────────────
    #  Layer 2: Planner — 优先级识别 + 计划生成
    # ────────────────────────────────────────────────────
    @staticmethod
    def planner_prompt() -> str:
        return """你现在扮演 Minecraft 智能体的规划器。

请综合以下信息：
- current_state
- recent_trajectory
- episodic_memories
- semantic_rules
- skills
- current_meta_goal

你的任务：
1. 判断当前最优先的问题是什么
2. 决定当前阶段目标
3. 选择一个最稳妥且收益高的技能或动作链
4. 输出最多 5 个原子动作，避免过长计划
5. 若当前计划风险过高，改为更保守方案

规划规则：
- 如果 time=night 且没有安全庇护所，优先 shelter
- 如果 health < 8，避免主动战斗
- 如果 hunger < 6，优先获取食物
- 如果已有木制工具且周围可获取石头，优先升级到石制工具
- 如果已多次在类似场景失败，禁止重复同样策略

请严格输出 JSON：
{
  "priority_issue": "",
  "selected_skill": "",
  "goal": "",
  "subgoal": "",
  "expected_reward": "",
  "main_risk": "",
  "action_plan": [
    {"action": "", "args": {}}
  ],
  "replan_trigger": ""
}"""

    # ────────────────────────────────────────────────────
    #  Layer 3: Quick Reflection — 在线偏差检测
    # ────────────────────────────────────────────────────
    @staticmethod
    def quick_reflection_prompt() -> str:
        return """你是一个在线反思模块。你的输入是：
- current_state
- plan
- execution_result
- recent_trajectory

请判断当前偏差是：
- 正常波动
- 执行失误
- 规则失效
- 技能缺失
- 风险误判

输出：
{
  "status": "ok|deviation|failure",
  "failure_type": "none|execution_error|bad_plan|missing_skill|risk_miscalibration",
  "cause": "",
  "immediate_fix": "",
  "should_store_memory": true,
  "memory_candidate": {
    "summary": "",
    "lesson": "",
    "tags": []
  }
}"""

    # ────────────────────────────────────────────────────
    #  Layer 4: Long Reflection — 整局复盘
    # ────────────────────────────────────────────────────
    @staticmethod
    def long_reflection_prompt() -> str:
        return """你是 Minecraft 智能体的赛后复盘器。

请根据整局 trajectory，总结：
1. 本局成功的关键策略
2. 本局失败的根本原因
3. 哪些经验只适用于局部场景
4. 哪些经验可以抽象成通用规则
5. 是否应该新增或修改技能

请输出严格 JSON：
{
  "episode_summary": "",
  "success_patterns": [
    {"pattern": "", "evidence": "", "reusability": "low|medium|high"}
  ],
  "failure_patterns": [
    {"pattern": "", "root_cause": "", "severity": 0.0}
  ],
  "new_rules": [
    {"rule": "", "confidence": 0.0}
  ],
  "revise_rules": [
    {"old_rule": "", "problem": "", "new_rule": ""}
  ],
  "new_skills": [
    {
      "name": "",
      "trigger": "",
      "steps": [],
      "failure_recovery": []
    }
  ],
  "delete_or_deprioritize_skills": [
    {"skill": "", "reason": ""}
  ]
}"""

    # ────────────────────────────────────────────────────
    #  Layer 5: Skill Builder — 技能发明
    # ────────────────────────────────────────────────────
    @staticmethod
    def skill_builder_prompt() -> str:
        return """你是一个技能发明器。请基于过去失败案例和成功经验，为 Minecraft 智能体发明一个"可复用的技能"。

要求：
1. 技能必须有明确触发条件
2. 技能必须解决一个高频问题
3. 技能步骤必须短小、稳定、可执行
4. 必须包含失败恢复策略
5. 不要发明过于宽泛的技能

输出格式：
{
  "skill_name": "",
  "purpose": "",
  "trigger_conditions": [],
  "preconditions": [],
  "steps": [],
  "stop_conditions": [],
  "failure_recovery": [],
  "metrics": {
    "success_signal": "",
    "risk_signal": ""
  }
}"""

    # ────────────────────────────────────────────────────
    #  Layer 6: Evolution — 策略演化
    # ────────────────────────────────────────────────────
    @staticmethod
    def evolution_prompt() -> str:
        return """你是一个策略演化器。你将看到多个版本的 agent 配置及其评测结果。

你的目标：
1. 找出高分版本真正有效的原因
2. 找出低分版本失败的主要机制
3. 生成一个新的候选版本
4. 变异幅度不能过大，必须是局部、可解释、可验证的改动

候选配置包括：
- planner_prompt
- reflection_prompt
- risk_thresholds
- memory_retrieval_weights
- skill_priority

输出格式：
{
  "elite_insights": [
    {"version": "", "why_good": ""}
  ],
  "bad_insights": [
    {"version": "", "why_bad": ""}
  ],
  "child_candidate": {
    "planner_delta": "",
    "reflection_delta": "",
    "risk_thresholds": {},
    "memory_retrieval_weights": {},
    "skill_priority_changes": []
  },
  "hypothesis": "这个新版本预计为什么会更强"
}"""

    # ════════════════════════════════════════════════════════════
    #  运行时主 Prompt: Decision Pipeline（融合 Layer 1 + 2）
    #  在单次 LLM 调用内完成三阶段推理：
    #    Phase A  评估（Agent Core）—— 分析状态、判定风险
    #    Phase B  规划（Planner）   —— 确定优先问题、选择技能、生成计划
    #    Phase C  输出动作
    # ════════════════════════════════════════════════════════════
    @staticmethod
    def decision_pipeline_prompt() -> str:
        return f"""你是 Luanti 生存智能体。分析状态，输出目标导向的动作计划（JSON）。

【体能系统】每个动作消耗体能，体能值 0~100：
  重型动作（tunnel/bridge/build_shelter/tower_up）→ 消耗 8~12
  中型动作（dig/attack/farm/swim）→ 消耗 4~7
  轻型动作（move/craft/find_resource）→ 消耗 2~3
  感知与整理（look_around/equip/sort_inventory）→ 消耗 1
  恢复动作：eat +15, wait +5, 日间自动 +1/步
  体能 < 10 → 只能 wait（强制休息）
  体能耗尽 → 持续惩罚，必须立刻恢复

【奖励机制】每个动作必须推进目标，得分规则：
  +3 采集到新资源（wood/stone/ore）
  +5 合成新工具或物品
  +5 建造庇护所
  +2 成功进食恢复生命/饥饿
  +1 探索到新区域（位置变化）
  -2 重复失败的动作
  -3 原地不动（position_stuck）
  -5 死亡
  额外：体能越低惩罚越大，体能=0 每步 -1.0

【决策流程】
  1. 读 current_state：health, hunger, time, nearby_blocks, inventory, nearby_entities, stamina
  2. 读 recent_failures：这些动作刚失败，绝不重复
  3. 评估体能：体能不足时优先选择低消耗动作或恢复
  4. 如有 user_task → 分解为子步骤，优先执行
  5. 选择净收益最高的动作序列（奖励 - 体能消耗），3-5个

【优先级】
  stamina<10 → wait（必须恢复）
  health<5/hunger<3 → eat_food/retreat（保命 + 恢复体能）
  night+无庇护 → build_shelter
  无工具 → find_resource(tree) → gather_wood → craft_tool
  有工具 → 升级（木→石→铁）或采集更多资源
  安全且体能充足 → 探索、建造、扩张
  体能中等(30-50) → 优先轻量动作，穿插 wait 恢复

【关键规则】
  - nearby_blocks 里有 tree/wood 才能 gather_wood；有 stone/cobble 才能 mine_stone
  - 没有目标资源 → 先 find_resource 或 explore 移动
  - position_stuck → 必须 jump 跳跃脱困，再配合 explore/move_to 移动；如多次跳跃仍无法脱困，使用 tower_up 垫高或 tunnel 挖隧道离开
  - 参考 episodic_memories 和 semantic_rules 避免重蹈覆辙
  - 挖掘(dig/gather)后物品会掉落在地上，nearby_entities 中 type="dropped_item" 表示掉落物
  - 发现 dropped_item → 立刻用 pickup_item 拾取，否则物品会消失！
  - 推荐动作链：dig → pickup_item → 检查 inventory

【可用动作】
{ACTION_REGISTRY}
【输出】严格JSON，无其他文字：
{{
  "mode": "survive|gather|craft|explore|combat|retreat|build",
  "goal": "主目标",
  "subgoal": "当前子步骤",
  "reason": "选择原因",
  "risk_level": "low|medium|high",
  "priority_issue": "最紧迫问题",
  "memory_references": [],
  "action_plan": [{{"action": "动作名", "args": {{}}}}],
  "replan_trigger": "重新规划条件",
  "reflection_needed": false
}}"""

    # ────────────────────────────────────────────────────
    #  向后兼容：保留 unified_agent_prompt 别名
    # ────────────────────────────────────────────────────
    @staticmethod
    def unified_agent_prompt() -> str:
        """向后兼容别名 → 转发到 decision_pipeline_prompt"""
        return PromptTemplates.decision_pipeline_prompt()

    # ════════════════════════════════════════════════════════════
    #  上下文构建工具
    # ════════════════════════════════════════════════════════════

    @classmethod
    def format_context(cls,
                       current_state: Dict[str, Any],
                       recent_trajectory: list,
                       episodic_memories: list,
                       semantic_rules: list,
                       skills: list,
                       meta_goal: str = None) -> str:
        """格式化上下文信息为 prompt 输入"""
        context = {
            "current_state": current_state,
            "recent_trajectory": recent_trajectory[-10:] if len(recent_trajectory) > 10 else recent_trajectory,
            "episodic_memories": episodic_memories,
            "semantic_rules": semantic_rules,
            "skills": [s.get("name", s) if isinstance(s, dict) else s for s in skills]
        }

        if meta_goal:
            context["current_meta_goal"] = meta_goal

        return json.dumps(context, ensure_ascii=False, indent=2)

    @classmethod
    def build_decision_context(cls,
                               current_state: Dict[str, Any],
                               memories: Dict[str, Any],
                               agent_internal_state: Dict[str, Any] = None,
                               recent_failures: List[Dict[str, str]] = None,
                               priority_issue: str = None) -> Dict[str, Any]:
        """构建决策流水线的完整上下文

        Args:
            current_state:       环境状态（来自 Luanti）
            memories:            记忆检索结果（来自 MemoryManager.to_prompt_context）
            agent_internal_state: 智能体内部状态（模式、风险等级等）
            recent_failures:     近期失败记录 [{"action": ..., "reason": ...}, ...]
            priority_issue:      规则系统预判的最优先问题

        Returns:
            完整上下文字典，可直接传入 LLMClient.generate_decision()
        """
        context: Dict[str, Any] = {
            "current_state": current_state,
            "recent_trajectory": memories.get("recent_trajectory", []),
            "episodic_memories": memories.get("episodic_memories", []),
            "semantic_rules": memories.get("semantic_rules", []),
            "skills": memories.get("skills", []),
        }

        # 添加记忆摘要，让 LLM 更容易发现可用记忆
        memory_hints = []
        for ep in memories.get("episodic_memories", []):
            if isinstance(ep, dict) and ep.get("lesson"):
                memory_hints.append(ep["lesson"])
        for rule in memories.get("semantic_rules", []):
            if isinstance(rule, dict) and rule.get("rule"):
                memory_hints.append(rule["rule"])
        if memory_hints:
            context["memory_summary"] = "你已经学到的经验：" + "；".join(memory_hints[:5])

        if agent_internal_state:
            context["agent_internal_state"] = agent_internal_state

        if recent_failures:
            context["recent_failures"] = recent_failures

        if priority_issue:
            context["system_priority_hint"] = priority_issue

        return context

    @classmethod
    def build_full_prompt(cls,
                          prompt_type: str,
                          current_state: Dict[str, Any],
                          recent_trajectory: list = None,
                          episodic_memories: list = None,
                          semantic_rules: list = None,
                          skills: list = None,
                          meta_goal: str = None,
                          extra_context: Dict[str, Any] = None) -> str:
        """构建完整的 prompt（向后兼容接口）"""

        prompt_map = {
            "core": cls.agent_core_prompt,
            "planner": cls.planner_prompt,
            "quick_reflection": cls.quick_reflection_prompt,
            "long_reflection": cls.long_reflection_prompt,
            "skill_builder": cls.skill_builder_prompt,
            "evolution": cls.evolution_prompt,
            "unified": cls.unified_agent_prompt,
            "pipeline": cls.decision_pipeline_prompt,
        }

        template_fn = prompt_map.get(prompt_type, cls.decision_pipeline_prompt)
        system_prompt = template_fn()

        context = cls.format_context(
            current_state=current_state or {},
            recent_trajectory=recent_trajectory or [],
            episodic_memories=episodic_memories or [],
            semantic_rules=semantic_rules or [],
            skills=skills or [],
            meta_goal=meta_goal
        )

        if extra_context:
            context += "\n\n额外信息：\n" + json.dumps(extra_context, ensure_ascii=False, indent=2)

        return f"{system_prompt}\n\n当前输入：\n{context}"
