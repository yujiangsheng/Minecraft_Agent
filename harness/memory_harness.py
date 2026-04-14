"""
Memory Harness — 记忆系统评测

评测三层记忆系统的检索质量：
  - 情景记忆检索召回率（给定状态，检索出的记忆是否匹配预期标签）
  - 语义规则匹配准确率（给定条件，返回的规则是否相关）
  - 技能触发精度（给定条件，触发的技能是否正确）
  - 规则置信度校准（经过 N 次反馈后置信度是否合理收敛）
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from memory import MemoryManager

logger = logging.getLogger("MemoryHarness")


# ════════════════════════════════════════════════
#  测试用例
# ════════════════════════════════════════════════

@dataclass
class MemoryTestCase:
    """记忆系统测试用例"""
    id: str
    name: str
    test_type: str          # episodic_retrieval / semantic_match / skill_trigger / confidence_cal
    description: str

    # 输入
    query_state: Dict[str, Any] = field(default_factory=dict)
    query_conditions: List[str] = field(default_factory=list)
    query_tags: List[str] = field(default_factory=list)

    # 期望
    expected_ids: List[str] = field(default_factory=list)     # 期望返回的记忆 ID
    expected_tags: List[str] = field(default_factory=list)     # 期望返回的记忆包含这些标签
    expected_rules: List[str] = field(default_factory=list)    # 期望匹配的规则关键词
    expected_skills: List[str] = field(default_factory=list)   # 期望触发的技能名
    min_results: int = 1
    max_results: int = 10


@dataclass
class MemoryTestResult:
    """单测试结果"""
    test_id: str
    test_name: str
    test_type: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class MemoryReport:
    """记忆系统评测报告"""
    total_tests: int = 0
    passed: int = 0
    results: List[MemoryTestResult] = field(default_factory=list)

    # 按类型统计
    episodic_pass_rate: float = 0.0
    semantic_pass_rate: float = 0.0
    skill_pass_rate: float = 0.0
    confidence_pass_rate: float = 0.0

    def compute(self):
        self.total_tests = len(self.results)
        self.passed = sum(r.passed for r in self.results)

        def _rate(test_type):
            rs = [r for r in self.results if r.test_type == test_type]
            return sum(r.passed for r in rs) / len(rs) if rs else 0.0

        self.episodic_pass_rate = _rate("episodic_retrieval")
        self.semantic_pass_rate = _rate("semantic_match")
        self.skill_pass_rate = _rate("skill_trigger")
        self.confidence_pass_rate = _rate("confidence_cal")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "pass_rate": round(self.passed / self.total_tests, 4) if self.total_tests else 0,
            "episodic_pass_rate": round(self.episodic_pass_rate, 4),
            "semantic_pass_rate": round(self.semantic_pass_rate, 4),
            "skill_pass_rate": round(self.skill_pass_rate, 4),
            "confidence_pass_rate": round(self.confidence_pass_rate, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        rate = self.passed / self.total_tests if self.total_tests else 0
        lines = [
            f"═══ Memory Harness Report ═══",
            f"Total: {self.total_tests} | Passed: {self.passed} | Rate: {rate:.1%}",
            f"",
            f"  情景记忆检索: {self.episodic_pass_rate:.1%}",
            f"  语义规则匹配: {self.semantic_pass_rate:.1%}",
            f"  技能触发精度: {self.skill_pass_rate:.1%}",
            f"  置信度校准:   {self.confidence_pass_rate:.1%}",
            "",
        ]
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append("── 失败用例 ──")
            for r in failures:
                lines.append(f"  {r.test_id} {r.test_name}: {r.details.get('reason', '')}")
        return "\n".join(lines)


# ════════════════════════════════════════════════
#  评测器
# ════════════════════════════════════════════════

class MemoryHarness:
    """记忆系统评测器"""

    def __init__(self):
        pass

    def run(self) -> MemoryReport:
        """执行全部记忆系统测试"""
        report = MemoryReport()

        # 准备包含预置数据的记忆管理器
        mm = self._prepare_memory()

        # 运行各类测试
        report.results.extend(self._test_episodic_retrieval(mm))
        report.results.extend(self._test_semantic_match(mm))
        report.results.extend(self._test_skill_trigger(mm))
        report.results.extend(self._test_confidence_calibration())

        report.compute()
        return report

    def _prepare_memory(self) -> MemoryManager:
        """准备含预植数据的记忆管理器"""
        mm = MemoryManager()

        # ── 植入情景记忆 ──
        mm.store_episode(
            summary="夜晚没有庇护所被僵尸攻击，掉血严重",
            lesson="夜晚前必须建造庇护所",
            tags=["night", "zombie", "no_shelter", "damage"],
            context={"time": "night", "health": 5, "has_shelter": False},
            outcome="failure",
        )
        mm.store_episode(
            summary="利用石剑击退了两只僵尸",
            lesson="有武器时可以选择战斗而非逃跑",
            tags=["combat", "zombie", "stone_sword", "victory"],
            context={"health": 15, "inventory": {"stone_sword": 1}},
            outcome="success",
        )
        mm.store_episode(
            summary="低血量时吃苹果恢复后继续采矿",
            lesson="保持食物储备以应对紧急回血",
            tags=["low_health", "eat", "recovery", "mining"],
            context={"health": 4, "inventory": {"apple": 2}},
            outcome="success",
        )
        mm.store_episode(
            summary="在水中差点溺死，紧急上浮",
            lesson="水中活动注意氧气，及时上浮",
            tags=["water", "drowning", "swim"],
            context={"breath": 1, "nearby_blocks": ["water"]},
            outcome="partial",
        )
        mm.store_episode(
            summary="收集木头后立即合成了木镐",
            lesson="优先合成工具提高效率",
            tags=["gather", "wood", "craft", "efficiency"],
            context={"inventory": {"wood": 8}},
            outcome="success",
        )
        mm.store_episode(
            summary="尝试无镐采矿失败",
            lesson="采矿需要对应等级的镐",
            tags=["mining", "no_tools", "failure"],
            context={"inventory": {}, "nearby_blocks": ["stone"]},
            outcome="failure",
        )

        return mm

    # ── 情景记忆检索 ──

    def _test_episodic_retrieval(self, mm: MemoryManager) -> List[MemoryTestResult]:
        results = []

        tests = [
            {
                "id": "ep_01", "name": "夜间搜索返回夜晚经验",
                "state": {"time": "night", "has_shelter": False, "health": 10,
                          "hunger": 15, "inventory": {}, "nearby_entities": [],
                          "nearby_blocks": [], "position": {"x": 0, "y": 0, "z": 0}},
                "tags": ["night"],
                "expected_tags": ["night"],
            },
            {
                "id": "ep_02", "name": "低血量搜索返回回血经验",
                "state": {"time": "day", "health": 4, "hunger": 15,
                          "inventory": {"apple": 1}, "nearby_entities": [],
                          "nearby_blocks": [], "has_shelter": False,
                          "position": {"x": 0, "y": 0, "z": 0}},
                "tags": ["low_health"],
                "expected_tags": ["low_health", "recovery", "eat"],
            },
            {
                "id": "ep_03", "name": "水中搜索返回溺水经验",
                "state": {"time": "day", "health": 15, "hunger": 15,
                          "inventory": {}, "nearby_entities": [],
                          "nearby_blocks": ["water"], "has_shelter": False,
                          "breath": 3, "position": {"x": 0, "y": 0, "z": 0}},
                "tags": ["water", "drowning"],
                "expected_tags": ["water", "drowning"],
            },
            {
                "id": "ep_04", "name": "战斗搜索返回战斗经验",
                "state": {"time": "day", "health": 15, "hunger": 15,
                          "inventory": {"stone_sword": 1},
                          "nearby_entities": [{"type": "zombie", "hostile": True, "distance": 5}],
                          "nearby_blocks": [], "has_shelter": False,
                          "position": {"x": 0, "y": 0, "z": 0}},
                "tags": ["combat", "zombie"],
                "expected_tags": ["combat", "zombie"],
            },
            {
                "id": "ep_05", "name": "采矿搜索返回采矿经验",
                "state": {"time": "day", "health": 20, "hunger": 20,
                          "inventory": {"stone_pickaxe": 1},
                          "nearby_entities": [],
                          "nearby_blocks": ["stone", "iron_ore"],
                          "has_shelter": False,
                          "position": {"x": 0, "y": 0, "z": 0}},
                "tags": ["mining"],
                "expected_tags": ["mining"],
            },
        ]

        for t in tests:
            try:
                retrieved = mm.retrieve(t["state"], tags=t.get("tags"))
                episodes = retrieved.get("episodic_memories", [])

                # 检查返回的情景是否包含期望标签
                returned_tags = set()
                for ep in episodes:
                    if isinstance(ep, dict):
                        returned_tags.update(ep.get("tags", []))

                expected = set(t["expected_tags"])
                hit = bool(expected & returned_tags)

                results.append(MemoryTestResult(
                    test_id=t["id"],
                    test_name=t["name"],
                    test_type="episodic_retrieval",
                    passed=hit,
                    details={
                        "expected_tags": t["expected_tags"],
                        "returned_tags": sorted(returned_tags),
                        "num_episodes": len(episodes),
                        "reason": "" if hit else "返回记忆不包含期望标签",
                    },
                ))
            except Exception as e:
                results.append(MemoryTestResult(
                    test_id=t["id"], test_name=t["name"],
                    test_type="episodic_retrieval", passed=False,
                    details={"reason": f"异常: {e}"},
                ))

        return results

    # ── 语义规则匹配 ──

    def _test_semantic_match(self, mm: MemoryManager) -> List[MemoryTestResult]:
        results = []

        tests = [
            {
                "id": "sem_01", "name": "夜间无庇护所触发庇护所规则",
                "conditions": ["night", "no_shelter"],
                "expected_keywords": ["shelter", "庇护所", "庇护"],
            },
            {
                "id": "sem_02", "name": "低血量触发避战规则",
                "conditions": ["health_low", "health<8"],
                "expected_keywords": ["生命值", "战斗", "health", "避免"],
            },
            {
                "id": "sem_03", "name": "低饥饿触发食物规则",
                "conditions": ["hunger_low", "hunger<6"],
                "expected_keywords": ["饥饿", "食物", "hunger", "获取"],
            },
            {
                "id": "sem_04", "name": "敌对生物触发战斗/撤退规则",
                "conditions": ["hostile_nearby", "low_equipment"],
                "expected_keywords": ["敌对", "撤退", "生物", "装备", "战斗"],
            },
        ]

        for t in tests:
            try:
                rules = mm.semantic.search_by_conditions(t["conditions"])
                rule_texts = " ".join(r.rule if hasattr(r, "rule") else str(r) for r in rules).lower()

                hit = any(kw.lower() in rule_texts for kw in t["expected_keywords"])

                results.append(MemoryTestResult(
                    test_id=t["id"],
                    test_name=t["name"],
                    test_type="semantic_match",
                    passed=hit,
                    details={
                        "conditions": t["conditions"],
                        "num_rules": len(rules),
                        "rule_preview": rule_texts[:200],
                        "reason": "" if hit else "规则文本不包含期望关键词",
                    },
                ))
            except Exception as e:
                results.append(MemoryTestResult(
                    test_id=t["id"], test_name=t["name"],
                    test_type="semantic_match", passed=False,
                    details={"reason": f"异常: {e}"},
                ))

        return results

    # ── 技能触发 ──

    def _test_skill_trigger(self, mm: MemoryManager) -> List[MemoryTestResult]:
        results = []

        tests = [
            {
                "id": "skill_01", "name": "夜间无庇护所触发建造技能",
                "conditions": ["night", "no_shelter", "has_blocks"],
                "expected_skills": ["emergency_shelter"],
            },
            {
                "id": "skill_02", "name": "需要木材时触发伐木技能",
                "conditions": ["need_wood", "trees_nearby"],
                "expected_skills": ["gather_wood"],
            },
            {
                "id": "skill_03", "name": "低血量敌人靠近触发撤退技能",
                "conditions": ["health_low", "danger_detected"],
                "expected_skills": ["retreat_to_safety"],
            },
            {
                "id": "skill_04", "name": "有材料时触发工具合成技能",
                "conditions": ["has_wood", "no_tools", "has_wood_logs"],
                "expected_skills": ["craft_basic_tools"],
            },
        ]

        for t in tests:
            try:
                skills = mm.skills.search_by_triggers(t["conditions"])
                skill_names = [sk.name if hasattr(sk, "name") else str(sk) for sk in skills]

                hit = any(exp in skill_names for exp in t["expected_skills"])

                results.append(MemoryTestResult(
                    test_id=t["id"],
                    test_name=t["name"],
                    test_type="skill_trigger",
                    passed=hit,
                    details={
                        "conditions": t["conditions"],
                        "triggered_skills": skill_names,
                        "expected": t["expected_skills"],
                        "reason": "" if hit else "未触发期望技能",
                    },
                ))
            except Exception as e:
                results.append(MemoryTestResult(
                    test_id=t["id"], test_name=t["name"],
                    test_type="skill_trigger", passed=False,
                    details={"reason": f"异常: {e}"},
                ))

        return results

    # ── 置信度校准 ──

    def _test_confidence_calibration(self) -> List[MemoryTestResult]:
        """测试规则置信度是否在反馈后合理收敛"""
        results = []

        mm = MemoryManager()

        # 添加测试规则
        rule = mm.store_rule(
            rule="夜晚在户外很危险",
            confidence=0.5,
            conditions=["night"],
        )
        rule_id = rule.id if hasattr(rule, "id") else None

        if rule_id is None:
            results.append(MemoryTestResult(
                test_id="conf_01", test_name="正反馈置信度上升",
                test_type="confidence_cal", passed=False,
                details={"reason": "无法获取规则 ID"},
            ))
            return results

        # 测试 1: 连续正反馈应提高置信度
        initial_conf = 0.5
        for _ in range(5):
            mm.semantic.update_confidence(rule_id, success=True)

        rule_after = mm.semantic.get(rule_id) if hasattr(mm.semantic, "get") else None
        if rule_after:
            final_conf = rule_after.confidence if hasattr(rule_after, "confidence") else 0.5
            passed = final_conf > initial_conf
            results.append(MemoryTestResult(
                test_id="conf_01", test_name="正反馈置信度上升",
                test_type="confidence_cal",
                passed=passed,
                details={"initial": initial_conf, "after_5_success": final_conf,
                         "reason": "" if passed else "置信度未上升"},
            ))
        else:
            results.append(MemoryTestResult(
                test_id="conf_01", test_name="正反馈置信度上升",
                test_type="confidence_cal", passed=False,
                details={"reason": "无法读取规则置信度"},
            ))

        # 测试 2: 连续负反馈应降低置信度
        mm2 = MemoryManager()
        rule2 = mm2.store_rule(
            rule="石头可以空手采集",
            confidence=0.7,
            conditions=["stone_nearby"],
        )
        rule2_id = rule2.id if hasattr(rule2, "id") else None

        if rule2_id:
            initial_conf2 = 0.7
            for _ in range(5):
                mm2.semantic.update_confidence(rule2_id, success=False)

            rule2_after = mm2.semantic.get(rule2_id) if hasattr(mm2.semantic, "get") else None
            if rule2_after:
                final_conf2 = rule2_after.confidence if hasattr(rule2_after, "confidence") else 0.7
                passed2 = final_conf2 < initial_conf2
                results.append(MemoryTestResult(
                    test_id="conf_02", test_name="负反馈置信度下降",
                    test_type="confidence_cal",
                    passed=passed2,
                    details={"initial": initial_conf2, "after_5_failure": final_conf2,
                             "reason": "" if passed2 else "置信度未下降"},
                ))
            else:
                results.append(MemoryTestResult(
                    test_id="conf_02", test_name="负反馈置信度下降",
                    test_type="confidence_cal", passed=False,
                    details={"reason": "无法读取规则置信度"},
                ))

        # 测试 3: 置信度应在 [0, 1] 内
        mm3 = MemoryManager()
        rule3 = mm3.store_rule(rule="测试规则", confidence=0.99, conditions=[])
        rule3_id = rule3.id if hasattr(rule3, "id") else None
        if rule3_id:
            for _ in range(20):
                mm3.semantic.update_confidence(rule3_id, success=True)
            rule3_after = mm3.semantic.get(rule3_id) if hasattr(mm3.semantic, "get") else None
            if rule3_after:
                conf3 = rule3_after.confidence if hasattr(rule3_after, "confidence") else 0
                passed3 = 0.0 <= conf3 <= 1.0
                results.append(MemoryTestResult(
                    test_id="conf_03", test_name="置信度范围 [0,1]",
                    test_type="confidence_cal",
                    passed=passed3,
                    details={"confidence": conf3,
                             "reason": "" if passed3 else f"置信度越界: {conf3}"},
                ))

        return results

    def save_report(self, report: MemoryReport, filepath: str):
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
