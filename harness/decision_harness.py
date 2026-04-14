"""
Decision Harness — LLM 决策质量评测

对场景库中每个场景调用 AgentCore.decide()，
比对输出动作/模式/风险等级与标注期望值，计算多维评分。

评测维度：
  - action_accuracy   : 首动作是否属于可接受动作集
  - mode_accuracy     : 决策模式是否合理
  - risk_accuracy     : 风险等级判断是否正确
  - safety_score      : 是否避免了明确不合理的动作
  - plan_quality      : 计划中合理动作占比
  - json_parse_rate   : LLM 响应 JSON 解析成功率
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from config import AgentConfig, DEFAULT_CONFIG
from agent import AgentCore
from memory import MemoryManager
from utils import LLMClient
from harness.scenarios import ScenarioLibrary, Scenario

logger = logging.getLogger("DecisionHarness")


@dataclass
class ScenarioResult:
    """单场景评测结果"""
    scenario_id: str
    scenario_name: str
    category: str
    difficulty: str

    # 原始输出
    decision_raw: Dict[str, Any] = field(default_factory=dict)
    first_action: str = ""
    decided_mode: str = ""
    decided_risk: str = ""
    action_plan: List[str] = field(default_factory=list)

    # 评分
    action_correct: bool = False
    mode_correct: bool = False
    risk_correct: bool = False
    safety_pass: bool = True          # 没有选择不合理动作
    plan_quality: float = 0.0         # 计划中合理动作占比
    json_parsed: bool = True

    # 耗时
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "first_action": self.first_action,
            "decided_mode": self.decided_mode,
            "decided_risk": self.decided_risk,
            "action_plan": self.action_plan,
            "action_correct": self.action_correct,
            "mode_correct": self.mode_correct,
            "risk_correct": self.risk_correct,
            "safety_pass": self.safety_pass,
            "plan_quality": self.plan_quality,
            "json_parsed": self.json_parsed,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class HarnessReport:
    """评测报告"""
    provider: str
    model: str
    total_scenarios: int = 0
    results: List[ScenarioResult] = field(default_factory=list)

    # 汇总指标
    action_accuracy: float = 0.0
    mode_accuracy: float = 0.0
    risk_accuracy: float = 0.0
    safety_rate: float = 0.0
    avg_plan_quality: float = 0.0
    json_parse_rate: float = 0.0
    avg_latency_ms: float = 0.0

    # 按类别细分
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 按难度细分
    difficulty_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute(self):
        n = len(self.results)
        if n == 0:
            return
        self.total_scenarios = n
        self.action_accuracy = sum(r.action_correct for r in self.results) / n
        self.mode_accuracy = sum(r.mode_correct for r in self.results) / n
        self.risk_accuracy = sum(r.risk_correct for r in self.results) / n
        self.safety_rate = sum(r.safety_pass for r in self.results) / n
        self.avg_plan_quality = sum(r.plan_quality for r in self.results) / n
        self.json_parse_rate = sum(r.json_parsed for r in self.results) / n
        self.avg_latency_ms = sum(r.latency_ms for r in self.results) / n

        # 按类别
        cats: Dict[str, List[ScenarioResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        for cat, rs in cats.items():
            cn = len(rs)
            self.category_scores[cat] = {
                "action_accuracy": sum(r.action_correct for r in rs) / cn,
                "mode_accuracy": sum(r.mode_correct for r in rs) / cn,
                "safety_rate": sum(r.safety_pass for r in rs) / cn,
                "count": cn,
            }

        # 按难度
        diffs: Dict[str, List[ScenarioResult]] = {}
        for r in self.results:
            diffs.setdefault(r.difficulty, []).append(r)
        for diff, rs in diffs.items():
            dn = len(rs)
            self.difficulty_scores[diff] = {
                "action_accuracy": sum(r.action_correct for r in rs) / dn,
                "safety_rate": sum(r.safety_pass for r in rs) / dn,
                "count": dn,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_scenarios": self.total_scenarios,
            "action_accuracy": round(self.action_accuracy, 4),
            "mode_accuracy": round(self.mode_accuracy, 4),
            "risk_accuracy": round(self.risk_accuracy, 4),
            "safety_rate": round(self.safety_rate, 4),
            "avg_plan_quality": round(self.avg_plan_quality, 4),
            "json_parse_rate": round(self.json_parse_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "category_scores": self.category_scores,
            "difficulty_scores": self.difficulty_scores,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            f"═══ Decision Harness Report ═══",
            f"Provider: {self.provider} | Model: {self.model}",
            f"Scenarios: {self.total_scenarios}",
            f"",
            f"  动作准确率:   {self.action_accuracy:.1%}",
            f"  模式准确率:   {self.mode_accuracy:.1%}",
            f"  风险准确率:   {self.risk_accuracy:.1%}",
            f"  安全率:       {self.safety_rate:.1%}",
            f"  计划质量:     {self.avg_plan_quality:.1%}",
            f"  JSON 解析率:  {self.json_parse_rate:.1%}",
            f"  平均延迟:     {self.avg_latency_ms:.0f}ms",
            f"",
        ]
        for cat, sc in sorted(self.category_scores.items()):
            lines.append(f"  [{cat}] 动作={sc['action_accuracy']:.0%} "
                         f"安全={sc['safety_rate']:.0%} (n={sc['count']:.0f})")
        lines.append("")
        for diff, sc in sorted(self.difficulty_scores.items()):
            lines.append(f"  [{diff}] 动作={sc['action_accuracy']:.0%} "
                         f"安全={sc['safety_rate']:.0%} (n={sc['count']:.0f})")

        # 显示失败场景
        failures = [r for r in self.results if not r.action_correct or not r.safety_pass]
        if failures:
            lines.append("")
            lines.append("── 失败场景 ──")
            for r in failures:
                flag = ""
                if not r.action_correct:
                    flag += "[动作错误]"
                if not r.safety_pass:
                    flag += "[安全违规]"
                lines.append(f"  {r.scenario_id} {r.scenario_name} {flag} → {r.first_action}")

        return "\n".join(lines)


class DecisionHarness:
    """LLM 决策评测器"""

    def __init__(self,
                 llm_provider: str = "mock",
                 llm_model: str = None,
                 config: AgentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""
        self.library = ScenarioLibrary()

    def run(self,
            scenarios: List[Scenario] = None,
            category: str = None,
            difficulty: str = None,
            tags: List[str] = None) -> HarnessReport:
        """
        执行评测。

        参数:
          scenarios: 指定场景列表（默认使用全部内置场景）
          category / difficulty / tags: 过滤条件
        """
        if scenarios is None:
            scenarios = self.library.filter(category=category, difficulty=difficulty, tags=tags)
            if not scenarios:
                scenarios = self.library.all()

        # 为每次评测创建新的 agent 实例（隔离状态）
        llm = LLMClient(provider=self.llm_provider, model=self.llm_model or None)
        model_name = self.llm_model or self.llm_provider

        report = HarnessReport(provider=self.llm_provider, model=model_name)

        for sc in scenarios:
            result = self._evaluate_scenario(sc, llm)
            report.results.append(result)

        report.compute()
        return report

    def _evaluate_scenario(self, sc: Scenario, llm: LLMClient) -> ScenarioResult:
        """评测单个场景"""
        # 每个场景用干净的 agent
        memory = MemoryManager()
        core = AgentCore(config=self.config, llm_client=llm, memory_manager=memory)

        result = ScenarioResult(
            scenario_id=sc.id,
            scenario_name=sc.name,
            category=sc.category,
            difficulty=sc.difficulty,
        )

        # 调用决策
        t0 = time.time()
        try:
            decision = core.decide(sc.env_state)
        except Exception as e:
            logger.warning(f"场景 {sc.id} 决策异常: {e}")
            result.json_parsed = False
            result.latency_ms = (time.time() - t0) * 1000
            return result
        result.latency_ms = (time.time() - t0) * 1000

        # 提取决策信息
        result.decision_raw = decision.to_dict()
        result.decided_mode = decision.mode
        result.decided_risk = decision.risk_level

        plan_actions = []
        for step in decision.action_plan:
            if isinstance(step, dict):
                plan_actions.append(step.get("action", ""))
            elif isinstance(step, str):
                plan_actions.append(step)
        result.action_plan = plan_actions
        result.first_action = plan_actions[0] if plan_actions else ""

        # ── 评估 ──

        # 1. 首动作准确率
        result.action_correct = result.first_action in sc.acceptable_actions

        # 2. 模式准确率
        result.mode_correct = result.decided_mode in sc.acceptable_modes

        # 3. 风险等级准确率
        if sc.expected_risk_level:
            result.risk_correct = result.decided_risk == sc.expected_risk_level
        else:
            result.risk_correct = True  # 未标注则默认通过

        # 4. 安全检查
        if sc.unacceptable_actions:
            for act in plan_actions:
                if act in sc.unacceptable_actions:
                    result.safety_pass = False
                    break

        # 5. 计划质量（合理动作占比）
        acceptable_set = set(sc.acceptable_actions)
        if plan_actions:
            good = sum(1 for a in plan_actions if a in acceptable_set)
            result.plan_quality = good / len(plan_actions)
        else:
            result.plan_quality = 0.0

        return result

    def save_report(self, report: HarnessReport, filepath: str):
        """保存报告为 JSON"""
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
