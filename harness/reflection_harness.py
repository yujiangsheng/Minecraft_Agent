"""
Reflection Harness — 反思校准评测

注入已知失败模式，检查 QuickReflection 和 LongReflection 的诊断准确性：
  - QuickReflection: 给定 plan + 执行结果 → 是否正确识别 failure_type 和 cause
  - LongReflection:  给定一整段轨迹 → 是否正确总结模式、提出规则更新

评测维度:
  - 偏差检测率:    有偏差时识别为非 ok 的比例
  - 失败类型准确率: failure_type 与标注匹配的比例
  - 即时修正合理率: immediate_fix 建议中包含合理动作的比例
  - 长期反思覆盖率: 成功/失败模式是否被识别
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from config import AgentConfig, DEFAULT_CONFIG
from agent import QuickReflection, LongReflection
from memory import MemoryManager
from utils import LLMClient

logger = logging.getLogger("ReflectionHarness")


# ════════════════════════════════════════════════
#  测试用例
# ════════════════════════════════════════════════

@dataclass
class ReflectionTestCase:
    """反思测试用例"""
    id: str
    name: str
    test_type: str               # quick / long

    # Quick Reflection 输入
    current_state: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    recent_trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # Long Reflection 输入
    episode_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    episode_outcome: str = ""
    episode_stats: Dict[str, Any] = field(default_factory=dict)

    # 期望
    expect_deviation: bool = True        # 是否应察觉偏差
    expected_failure_type: str = ""      # 期望的 failure_type
    expected_fix_keywords: List[str] = field(default_factory=list)  # 修正建议应包含的关键词
    expect_memory_store: bool = False    # 是否应建议存储记忆
    expected_pattern_keywords: List[str] = field(default_factory=list)  # 长期反思应识别的模式关键词


@dataclass
class ReflectionTestResult:
    """单测试结果"""
    test_id: str
    test_name: str
    test_type: str
    passed: bool
    sub_scores: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "passed": self.passed,
            "sub_scores": self.sub_scores,
            "details": self.details,
        }


@dataclass
class ReflectionReport:
    """反思评测报告"""
    total_tests: int = 0
    passed: int = 0
    results: List[ReflectionTestResult] = field(default_factory=list)

    # 细分
    quick_deviation_detect_rate: float = 0.0
    quick_failure_type_accuracy: float = 0.0
    quick_fix_relevance: float = 0.0
    long_pattern_coverage: float = 0.0

    def compute(self):
        self.total_tests = len(self.results)
        self.passed = sum(r.passed for r in self.results)

        quick = [r for r in self.results if r.test_type == "quick"]
        if quick:
            self.quick_deviation_detect_rate = (
                sum(r.sub_scores.get("deviation_detected", False) for r in quick) / len(quick)
            )
            self.quick_failure_type_accuracy = (
                sum(r.sub_scores.get("failure_type_correct", False) for r in quick) / len(quick)
            )
            self.quick_fix_relevance = (
                sum(r.sub_scores.get("fix_relevant", False) for r in quick) / len(quick)
            )

        long_tests = [r for r in self.results if r.test_type == "long"]
        if long_tests:
            self.long_pattern_coverage = (
                sum(r.sub_scores.get("patterns_identified", False) for r in long_tests) / len(long_tests)
            )

    def to_dict(self) -> Dict[str, Any]:
        rate = self.passed / self.total_tests if self.total_tests else 0
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "pass_rate": round(rate, 4),
            "quick_deviation_detect_rate": round(self.quick_deviation_detect_rate, 4),
            "quick_failure_type_accuracy": round(self.quick_failure_type_accuracy, 4),
            "quick_fix_relevance": round(self.quick_fix_relevance, 4),
            "long_pattern_coverage": round(self.long_pattern_coverage, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        rate = self.passed / self.total_tests if self.total_tests else 0
        lines = [
            f"═══ Reflection Harness Report ═══",
            f"Total: {self.total_tests} | Passed: {self.passed} | Rate: {rate:.1%}",
            "",
            f"  Quick — 偏差检测率:     {self.quick_deviation_detect_rate:.1%}",
            f"  Quick — 失败类型准确率: {self.quick_failure_type_accuracy:.1%}",
            f"  Quick — 修正建议合理率: {self.quick_fix_relevance:.1%}",
            f"  Long  — 模式覆盖率:     {self.long_pattern_coverage:.1%}",
            "",
        ]
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append("── 失败用例 ──")
            for r in failures:
                lines.append(f"  {r.test_id} {r.test_name}: {r.details.get('reason', '')}")
        return "\n".join(lines)


# ════════════════════════════════════════════════
#  内置测试用例
# ════════════════════════════════════════════════

BUILTIN_QUICK_TESTS = [
    ReflectionTestCase(
        id="qr_01", name="执行失败:挖矿无镐",
        test_type="quick",
        current_state={"health": 15, "hunger": 15, "inventory": {},
                       "nearby_blocks": ["stone"], "time": "day",
                       "nearby_entities": [], "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "采集石头", "action": "mine_stone", "initial_health": 15},
        execution_result={"success": False, "action": "mine_stone",
                          "outcome": "没有镐", "error_type": "precondition_not_met",
                          "resources_gained": 0, "damage_taken": 0},
        expect_deviation=True,
        expected_failure_type="bad_plan",
        expected_fix_keywords=["craft", "tool", "pickaxe", "工具", "合成", "镐"],
    ),
    ReflectionTestCase(
        id="qr_02", name="执行失败:吃食物但无食物",
        test_type="quick",
        current_state={"health": 5, "hunger": 3, "inventory": {},
                       "time": "day", "nearby_entities": [],
                       "nearby_blocks": [], "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "恢复生命值", "action": "eat_food", "initial_health": 5},
        execution_result={"success": False, "action": "eat_food",
                          "outcome": "无食物", "error_type": "precondition_not_met",
                          "resources_gained": 0, "damage_taken": 0},
        expect_deviation=True,
        expected_failure_type="bad_plan",
        expected_fix_keywords=["find", "food", "hunt", "食物", "寻找", "farm"],
    ),
    ReflectionTestCase(
        id="qr_03", name="风险误判:低血量主动战斗受伤",
        test_type="quick",
        current_state={"health": 3, "hunger": 10,
                       "inventory": {"stone_sword": 1},
                       "nearby_entities": [{"type": "zombie", "hostile": True, "distance": 3}],
                       "time": "night", "nearby_blocks": [],
                       "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "战斗", "action": "attack", "initial_health": 6},
        execution_result={"success": True, "action": "attack",
                          "outcome": "击中僵尸但受伤", "error_type": None,
                          "resources_gained": 0, "damage_taken": 3},
        expect_deviation=True,
        expected_failure_type="risk_miscalibration",
        expected_fix_keywords=["retreat", "flee", "eat", "撤退", "逃", "回血"],
    ),
    ReflectionTestCase(
        id="qr_04", name="正常成功:采集木头",
        test_type="quick",
        current_state={"health": 20, "hunger": 20,
                       "inventory": {"wood": 5},
                       "nearby_blocks": ["tree"], "time": "day",
                       "nearby_entities": [], "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "采集木头", "action": "gather_wood", "initial_health": 20},
        execution_result={"success": True, "action": "gather_wood",
                          "outcome": "获得 2 木头", "error_type": None,
                          "resources_gained": 2, "damage_taken": 0},
        expect_deviation=False,
    ),
    ReflectionTestCase(
        id="qr_05", name="技能缺失:尝试冶炼但无熔炉",
        test_type="quick",
        current_state={"health": 18, "hunger": 18,
                       "inventory": {"iron_lump": 3, "coal_lump": 2},
                       "time": "day", "nearby_entities": [],
                       "nearby_blocks": ["stone"], "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "冶炼铁锭", "action": "smelt", "initial_health": 18},
        execution_result={"success": False, "action": "smelt",
                          "outcome": "附近没有熔炉", "error_type": "skill_not_found",
                          "resources_gained": 0, "damage_taken": 0},
        expect_deviation=True,
        expected_failure_type="missing_skill",
        expected_fix_keywords=["furnace", "craft", "build", "熔炉", "合成"],
    ),
    ReflectionTestCase(
        id="qr_06", name="执行失败:重复挖同一位置",
        test_type="quick",
        current_state={"health": 20, "hunger": 15,
                       "inventory": {"wooden_pickaxe": 1},
                       "time": "day", "nearby_entities": [],
                       "nearby_blocks": ["dirt"], "has_shelter": False,
                       "position": {"x": 0, "y": 0, "z": 0}},
        plan={"goal": "采集石头", "action": "mine_stone", "initial_health": 20},
        execution_result={"success": False, "action": "mine_stone",
                          "outcome": "此处没有石头", "error_type": "action_failed",
                          "resources_gained": 0, "damage_taken": 0},
        recent_trajectory=[
            {"action": "mine_stone", "result": {"success": False}},
            {"action": "mine_stone", "result": {"success": False}},
        ],
        expect_deviation=True,
        expected_failure_type="execution_error",
        expected_fix_keywords=["move", "explore", "find", "移动", "寻找", "换位置"],
    ),
]

BUILTIN_LONG_TESTS = [
    ReflectionTestCase(
        id="lr_01", name="整局复盘:反复被夜间敌人杀死",
        test_type="long",
        episode_trajectory=[
            {"step": 1, "action": "explore", "result": {"success": True}},
            {"step": 2, "action": "gather_wood", "result": {"success": True}},
            {"step": 3, "action": "gather_wood", "result": {"success": True}},
            {"step": 4, "action": "explore", "result": {"success": True}},
            {"step": 5, "action": "explore", "result": {"success": True}},
            # 天黑了
            {"step": 6, "action": "explore", "result": {"success": True, "note": "night_fell"}},
            {"step": 7, "action": "attack", "result": {"success": True, "damage_taken": 3}},
            {"step": 8, "action": "attack", "result": {"success": True, "damage_taken": 4}},
            {"step": 9, "action": "retreat", "result": {"success": True}},
            {"step": 10, "action": "retreat", "result": {"success": False, "outcome": "被包围"}},
        ],
        episode_outcome="death",
        episode_stats={"total_steps": 10, "health_min": 0, "resources_gathered": 4},
        expected_pattern_keywords=["night", "shelter", "庇护", "夜晚", "建造"],
    ),
    ReflectionTestCase(
        id="lr_02", name="整局复盘:高效生存成功局",
        test_type="long",
        episode_trajectory=[
            {"step": 1, "action": "gather_wood", "result": {"success": True, "resources_gained": 2}},
            {"step": 2, "action": "gather_wood", "result": {"success": True, "resources_gained": 2}},
            {"step": 3, "action": "craft_tool", "result": {"success": True}},
            {"step": 4, "action": "mine_stone", "result": {"success": True, "resources_gained": 2}},
            {"step": 5, "action": "craft_tool", "result": {"success": True}},
            {"step": 6, "action": "build_shelter", "result": {"success": True}},
            {"step": 7, "action": "light_area", "result": {"success": True}},
            {"step": 8, "action": "wait", "result": {"success": True}},
            {"step": 9, "action": "explore", "result": {"success": True}},
            {"step": 10, "action": "mine_ore", "result": {"success": True, "resources_gained": 1}},
        ],
        episode_outcome="survived",
        episode_stats={"total_steps": 10, "health_min": 18, "resources_gathered": 7},
        expected_pattern_keywords=["gather", "craft", "shelter", "采集", "合成", "庇护"],
    ),
]


# ════════════════════════════════════════════════
#  评测器
# ════════════════════════════════════════════════

class ReflectionHarness:
    """反思校准评测器"""

    def __init__(self,
                 llm_provider: str = "mock",
                 llm_model: str = None,
                 config: AgentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""

    def run(self) -> ReflectionReport:
        """执行所有反思测试"""
        llm = LLMClient(provider=self.llm_provider, model=self.llm_model or None)
        memory = MemoryManager()

        quick_ref = QuickReflection(config=self.config, llm_client=llm, memory_manager=memory)
        long_ref = LongReflection(config=self.config, llm_client=llm, memory_manager=memory)

        report = ReflectionReport()

        for tc in BUILTIN_QUICK_TESTS:
            result = self._test_quick(tc, quick_ref)
            report.results.append(result)

        for tc in BUILTIN_LONG_TESTS:
            result = self._test_long(tc, long_ref)
            report.results.append(result)

        report.compute()
        return report

    def _test_quick(self, tc: ReflectionTestCase, qr: QuickReflection) -> ReflectionTestResult:
        """测试 QuickReflection"""
        sub_scores = {}
        details = {}

        try:
            result = qr.reflect(
                current_state=tc.current_state,
                plan=tc.plan,
                execution_result=tc.execution_result,
                recent_trajectory=tc.recent_trajectory,
            )

            status = result.status if hasattr(result, "status") else "unknown"
            failure_type = result.failure_type if hasattr(result, "failure_type") else ""
            immediate_fix = result.immediate_fix if hasattr(result, "immediate_fix") else ""
            should_store = result.should_store_memory if hasattr(result, "should_store_memory") else False

            # 1. 偏差检测
            detected = status != "ok"
            sub_scores["deviation_detected"] = (detected == tc.expect_deviation)

            # 2. 失败类型准确率
            if tc.expected_failure_type:
                sub_scores["failure_type_correct"] = (failure_type == tc.expected_failure_type)
            else:
                sub_scores["failure_type_correct"] = True

            # 3. 修正建议合理性
            if tc.expected_fix_keywords and immediate_fix:
                fix_lower = immediate_fix.lower()
                sub_scores["fix_relevant"] = any(
                    kw.lower() in fix_lower for kw in tc.expected_fix_keywords
                )
            elif not tc.expected_fix_keywords:
                sub_scores["fix_relevant"] = True
            else:
                # 有期望关键词但无修正建议 → 只在检测到偏差时才算失败
                sub_scores["fix_relevant"] = not tc.expect_deviation

            details = {
                "status": status,
                "failure_type": failure_type,
                "immediate_fix": immediate_fix[:100],
                "should_store": should_store,
            }

        except Exception as e:
            sub_scores = {"deviation_detected": False, "failure_type_correct": False,
                          "fix_relevant": False}
            details = {"reason": f"异常: {e}"}

        passed = all(sub_scores.values())
        return ReflectionTestResult(
            test_id=tc.id, test_name=tc.name, test_type="quick",
            passed=passed, sub_scores=sub_scores, details=details,
        )

    def _test_long(self, tc: ReflectionTestCase, lr: LongReflection) -> ReflectionTestResult:
        """测试 LongReflection"""
        sub_scores = {}
        details = {}

        try:
            result = lr.reflect(
                episode_trajectory=tc.episode_trajectory,
                episode_outcome=tc.episode_outcome,
                episode_stats=tc.episode_stats,
            )

            # 收集所有文本用于关键词匹配
            all_text = ""
            if hasattr(result, "episode_summary"):
                all_text += result.episode_summary + " "
            if hasattr(result, "success_patterns"):
                for p in result.success_patterns:
                    txt = p.pattern if hasattr(p, "pattern") else str(p)
                    all_text += txt + " "
            if hasattr(result, "failure_patterns"):
                for p in result.failure_patterns:
                    txt = p.pattern if hasattr(p, "pattern") else str(p)
                    all_text += txt + " "
            if hasattr(result, "new_rules"):
                for r in result.new_rules:
                    all_text += json.dumps(r, ensure_ascii=False) + " "
            if hasattr(result, "new_skills"):
                for s in result.new_skills:
                    all_text += json.dumps(s, ensure_ascii=False) + " "

            all_text_lower = all_text.lower()

            # 检查模式关键词覆盖
            if tc.expected_pattern_keywords:
                hits = sum(1 for kw in tc.expected_pattern_keywords
                           if kw.lower() in all_text_lower)
                sub_scores["patterns_identified"] = hits >= len(tc.expected_pattern_keywords) / 2
            else:
                sub_scores["patterns_identified"] = True

            # 检查是否有实质性输出
            summary = result.episode_summary if hasattr(result, "episode_summary") else ""
            sub_scores["has_summary"] = len(summary) > 5

            details = {
                "summary_preview": summary[:150],
                "text_length": len(all_text),
                "keyword_hits": [
                    kw for kw in tc.expected_pattern_keywords
                    if kw.lower() in all_text_lower
                ],
            }

        except Exception as e:
            sub_scores = {"patterns_identified": False, "has_summary": False}
            details = {"reason": f"异常: {e}"}

        passed = all(sub_scores.values())
        return ReflectionTestResult(
            test_id=tc.id, test_name=tc.name, test_type="long",
            passed=passed, sub_scores=sub_scores, details=details,
        )

    def save_report(self, report: ReflectionReport, filepath: str):
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
