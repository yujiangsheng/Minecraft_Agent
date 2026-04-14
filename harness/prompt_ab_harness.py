"""
Prompt A/B Harness — Prompt 变体对比测试

对同一场景使用不同 prompt 变体调用 LLM N 次，
统计动作分布、模式分布、JSON 解析成功率和延迟，
自动评选最优 prompt。
"""

import copy
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from config import AgentConfig, DEFAULT_CONFIG
from agent import AgentCore
from memory import MemoryManager
from utils import LLMClient
from prompts import PromptTemplates
from harness.scenarios import ScenarioLibrary, Scenario

logger = logging.getLogger("PromptABHarness")


@dataclass
class PromptVariant:
    """Prompt 变体"""
    id: str
    name: str
    description: str
    prompt_fn: Callable[[], str]   # 返回 system prompt 的函数


@dataclass
class VariantStats:
    """单变体在一组场景上的统计"""
    variant_id: str
    variant_name: str
    total_runs: int = 0
    json_parse_success: int = 0
    action_distribution: Dict[str, int] = field(default_factory=dict)
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    risk_distribution: Dict[str, int] = field(default_factory=dict)

    # 与标注的匹配度
    action_correct: int = 0
    mode_correct: int = 0
    safety_violations: int = 0

    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        n = max(self.total_runs, 1)
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "total_runs": self.total_runs,
            "json_parse_rate": round(self.json_parse_success / n, 4),
            "action_accuracy": round(self.action_correct / n, 4),
            "mode_accuracy": round(self.mode_correct / n, 4),
            "safety_violation_rate": round(self.safety_violations / n, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "action_distribution": dict(self.action_distribution),
            "mode_distribution": dict(self.mode_distribution),
        }


@dataclass
class ABReport:
    """A/B 测试报告"""
    scenarios_used: int = 0
    runs_per_variant: int = 0
    variant_stats: List[VariantStats] = field(default_factory=list)
    winner_id: str = ""
    winner_reason: str = ""

    def compute_winner(self):
        """根据综合分数选出最优变体"""
        if not self.variant_stats:
            return

        best_score = -1.0
        for vs in self.variant_stats:
            n = max(vs.total_runs, 1)
            score = (
                (vs.action_correct / n) * 0.35 +
                (vs.json_parse_success / n) * 0.25 +
                (vs.mode_correct / n) * 0.20 +
                (1.0 - vs.safety_violations / n) * 0.20
            )
            if score > best_score:
                best_score = score
                self.winner_id = vs.variant_id
                self.winner_reason = f"综合得分 {score:.3f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios_used": self.scenarios_used,
            "runs_per_variant": self.runs_per_variant,
            "winner_id": self.winner_id,
            "winner_reason": self.winner_reason,
            "variants": [vs.to_dict() for vs in self.variant_stats],
        }

    def summary(self) -> str:
        lines = [
            f"═══ Prompt A/B Test Report ═══",
            f"Scenarios: {self.scenarios_used} | Runs/variant: {self.runs_per_variant}",
            f"Winner: {self.winner_id} ({self.winner_reason})",
            "",
        ]
        for vs in self.variant_stats:
            d = vs.to_dict()
            flag = " ← WINNER" if vs.variant_id == self.winner_id else ""
            lines.append(
                f"  [{vs.variant_id}] {vs.variant_name}{flag}\n"
                f"      action_acc={d['action_accuracy']:.0%}  "
                f"mode_acc={d['mode_accuracy']:.0%}  "
                f"json={d['json_parse_rate']:.0%}  "
                f"safety_viol={d['safety_violation_rate']:.0%}  "
                f"latency={d['avg_latency_ms']:.0f}ms"
            )
            top3_actions = sorted(vs.action_distribution.items(),
                                  key=lambda x: -x[1])[:3]
            lines.append(f"      Top actions: {top3_actions}")
        return "\n".join(lines)


# ════════════════════════════════════════════════
#  内置 Prompt 变体
# ════════════════════════════════════════════════

def _variant_default() -> str:
    """默认决策流水线 prompt"""
    return PromptTemplates.decision_pipeline_prompt()


def _variant_concise() -> str:
    """精简版 prompt — 删除详细动作列表，只保留核心指令"""
    return """你是 Minecraft 生存智能体。根据当前状态输出决策 JSON。

优先级: 生命安全 > 食物 > 庇护所 > 工具 > 探索。
保守模式: 夜晚/低血量/低饥饿/有敌人时启用。

输出格式:
{
  "mode": "survive|gather|craft|explore|combat|retreat|build",
  "goal": "当前主目标",
  "subgoal": "下一子目标",
  "reason": "简洁原因",
  "risk_level": "low|medium|high",
  "memory_references": [],
  "action_plan": [{"action": "动作名", "args": {}}],
  "replan_trigger": "何时重新规划",
  "reflection_needed": false
}

动作: explore, move_to, jump, retreat, flee_from, swim, gather_wood, mine_stone,
mine_ore, dig, dig_down, tunnel, build_shelter, light_area, craft_tool, craft_item,
smelt, eat_food, attack, attack_entity, find_resource, equip, wait, place_block,
bridge, tower_up, farm_plant, farm_harvest, sort_inventory"""


def _variant_chain_of_thought() -> str:
    """思维链版 — 要求 LLM 先分析再决策"""
    base = PromptTemplates.decision_pipeline_prompt()
    cot_prefix = """在输出 JSON 之前，请先在 JSON 内的 "reason" 字段写出你的完整推理过程：
1. 当前最紧急的问题是什么？
2. 我有哪些资源可以利用？
3. 哪些记忆/规则与当前情境相关？
4. 最安全高效的行动方案是什么？

"""
    return cot_prefix + base


BUILTIN_VARIANTS = [
    PromptVariant(
        id="default",
        name="默认流水线",
        description="完整的 decision_pipeline_prompt",
        prompt_fn=_variant_default,
    ),
    PromptVariant(
        id="concise",
        name="精简版",
        description="去除详细动作说明，只保留核心指令",
        prompt_fn=_variant_concise,
    ),
    PromptVariant(
        id="cot",
        name="思维链版",
        description="要求 LLM 先分析推理再输出决策",
        prompt_fn=_variant_chain_of_thought,
    ),
]


# ════════════════════════════════════════════════
#  评测器
# ════════════════════════════════════════════════

class PromptABHarness:
    """Prompt A/B 对比评测器"""

    def __init__(self,
                 llm_provider: str = "mock",
                 llm_model: str = None,
                 config: AgentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""
        self.library = ScenarioLibrary()

    def run(self,
            variants: List[PromptVariant] = None,
            scenarios: List[Scenario] = None,
            runs_per_scenario: int = 1) -> ABReport:
        """
        对每个变体、每个场景跑 runs_per_scenario 次。

        参数:
          variants: prompt 变体列表（默认使用内置 3 种）
          scenarios: 场景子集（默认随机选 10 个覆盖各类别）
          runs_per_scenario: 每场景每变体重复次数
        """
        if variants is None:
            variants = BUILTIN_VARIANTS
        if scenarios is None:
            # 从各类别选代表性场景
            selected = []
            for cat in ("survival", "combat", "resource", "craft", "explore"):
                pool = self.library.filter(category=cat)
                selected.extend(pool[:3])
            if not selected:
                selected = self.library.all()[:10]
            scenarios = selected

        llm = LLMClient(provider=self.llm_provider, model=self.llm_model or None)

        report = ABReport(
            scenarios_used=len(scenarios),
            runs_per_variant=runs_per_scenario,
        )

        for variant in variants:
            stats = self._evaluate_variant(variant, scenarios, runs_per_scenario, llm)
            report.variant_stats.append(stats)

        report.compute_winner()
        return report

    def _evaluate_variant(self,
                          variant: PromptVariant,
                          scenarios: List[Scenario],
                          runs: int,
                          llm: LLMClient) -> VariantStats:
        """评测单个 prompt 变体"""
        stats = VariantStats(variant_id=variant.id, variant_name=variant.name)

        system_prompt = variant.prompt_fn()

        for sc in scenarios:
            for _ in range(runs):
                stats.total_runs += 1

                memory = MemoryManager()
                core = AgentCore(config=self.config, llm_client=llm, memory_manager=memory)

                # 注入自定义 prompt（通过直接调用底层 LLM）
                t0 = time.time()
                try:
                    perception = core.perceive(sc.env_state)
                    context = PromptTemplates.build_decision_context(
                        current_state=perception["env_state"],
                        memories=perception["memories"],
                        agent_internal_state=perception["agent_state"],
                        recent_failures=[],
                        priority_issue=core.state.priority_issue,
                    )
                    response = llm.generate_decision(system_prompt, context)
                except Exception as e:
                    logger.warning(f"变体 {variant.id} 场景 {sc.id} 异常: {e}")
                    stats.total_latency_ms += (time.time() - t0) * 1000
                    continue

                elapsed = (time.time() - t0) * 1000
                stats.total_latency_ms += elapsed

                if not response.success or not response.parsed:
                    continue
                stats.json_parse_success += 1

                parsed = response.parsed
                first_action = ""
                plan = parsed.get("action_plan", [])
                if plan and isinstance(plan[0], dict):
                    first_action = plan[0].get("action", "")
                elif plan and isinstance(plan[0], str):
                    first_action = plan[0]

                mode = parsed.get("mode", "")

                # 统计分布
                if first_action:
                    stats.action_distribution[first_action] = (
                        stats.action_distribution.get(first_action, 0) + 1
                    )
                if mode:
                    stats.mode_distribution[mode] = (
                        stats.mode_distribution.get(mode, 0) + 1
                    )
                risk = parsed.get("risk_level", "")
                if risk:
                    stats.risk_distribution[risk] = (
                        stats.risk_distribution.get(risk, 0) + 1
                    )

                # 与标注比较
                if first_action in sc.acceptable_actions:
                    stats.action_correct += 1
                if mode in sc.acceptable_modes:
                    stats.mode_correct += 1
                if sc.unacceptable_actions:
                    all_actions = [
                        (a.get("action", "") if isinstance(a, dict) else a)
                        for a in plan
                    ]
                    if any(a in sc.unacceptable_actions for a in all_actions):
                        stats.safety_violations += 1

        if stats.total_runs > 0:
            stats.avg_latency_ms = stats.total_latency_ms / stats.total_runs

        return stats

    def save_report(self, report: ABReport, filepath: str):
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
