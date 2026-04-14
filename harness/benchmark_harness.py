"""
Benchmark Harness — 端到端生存基准测试

在模拟环境中运行完整的 Agent 循环（感知→决策→执行→反思），
衡量存活步数、资源效率、任务完成率等。

基准套件:
  - survive_N        : 存活 N 步，衡量 health 保持率
  - gather_resource   : 从零开始收集指定资源，衡量步效率
  - build_shelter     : 天黑前完成庇护所建造
  - combat_survival   : 面对 N 波敌人的存活率
  - tool_progression  : 科技树推进效率（木→石→铁）
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from config import AgentConfig, DEFAULT_CONFIG
from agent import AgentCore
from memory import MemoryManager
from utils import LLMClient

logger = logging.getLogger("BenchmarkHarness")


# ════════════════════════════════════════════════
#  模拟环境（无需 Luanti）
# ════════════════════════════════════════════════

class SimulatedEnvironment:
    """轻量级模拟环境，根据动作更新状态"""

    def __init__(self, initial_state: Dict[str, Any]):
        self.state = copy.deepcopy(initial_state)
        self.step_count = 0
        self.events: List[Dict[str, Any]] = []

    def get_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """模拟执行动作，返回 action_result"""
        self.step_count += 1
        act = action.get("action", "")
        args = action.get("args", {})

        result = {
            "success": True,
            "action": act,
            "outcome": "",
            "error_type": None,
            "resources_gained": 0,
            "damage_taken": 0,
            "trigger_condition_met": False,
        }

        # ── 模拟各动作效果 ──

        if act in ("gather_wood", "dig") and args.get("target_type", "") in ("tree", ""):
            if "tree" in self.state.get("nearby_blocks", []):
                inv = self.state.setdefault("inventory", {})
                inv["wood"] = inv.get("wood", 0) + 2
                result["outcome"] = "获得 2 木头"
                result["resources_gained"] = 2
            else:
                result["success"] = False
                result["outcome"] = "附近无树木"
                result["error_type"] = "precondition_not_met"

        elif act in ("mine_stone", "dig") and args.get("target_type", "") == "stone":
            if "stone" in self.state.get("nearby_blocks", []):
                inv = self.state.setdefault("inventory", {})
                inv["cobble"] = inv.get("cobble", 0) + 2
                result["outcome"] = "获得 2 圆石"
                result["resources_gained"] = 2
            else:
                result["success"] = False
                result["outcome"] = "附近无石头"
                result["error_type"] = "precondition_not_met"

        elif act == "mine_ore":
            ore = args.get("ore_type", "iron_ore")
            if ore in self.state.get("nearby_blocks", []) or "iron_ore" in self.state.get("nearby_blocks", []):
                inv = self.state.setdefault("inventory", {})
                inv["iron_lump"] = inv.get("iron_lump", 0) + 1
                result["outcome"] = f"获得 1 {ore}"
                result["resources_gained"] = 1
            else:
                result["success"] = False
                result["outcome"] = "附近无矿石"

        elif act == "craft_tool":
            inv = self.state.setdefault("inventory", {})
            item = args.get("item", args.get("tool_type", ""))
            if inv.get("wood", 0) >= 3 and inv.get("stick", 0) >= 2:
                inv["wood"] -= 3
                inv["stick"] -= 2
                tool_name = "wooden_pickaxe" if "pick" in item else "wooden_sword"
                inv[tool_name] = inv.get(tool_name, 0) + 1
                result["outcome"] = f"合成 {tool_name}"
            elif inv.get("cobble", 0) >= 3 and inv.get("stick", 0) >= 2:
                inv["cobble"] -= 3
                inv["stick"] -= 2
                tool_name = "stone_pickaxe" if "pick" in item else "stone_sword"
                inv[tool_name] = inv.get(tool_name, 0) + 1
                result["outcome"] = f"合成 {tool_name}"
            else:
                result["success"] = False
                result["outcome"] = "材料不足"
                result["error_type"] = "precondition_not_met"

        elif act == "craft_item":
            inv = self.state.setdefault("inventory", {})
            if "stick" in args.get("item", ""):
                if inv.get("wood", 0) >= 1:
                    inv["wood"] -= 1
                    inv["stick"] = inv.get("stick", 0) + 4
                    result["outcome"] = "合成 4 木棍"
                else:
                    result["success"] = False
                    result["outcome"] = "无木头"

        elif act == "eat_food":
            inv = self.state.setdefault("inventory", {})
            foods = [k for k in inv if k in ("apple", "bread", "cooked_meat") and inv[k] > 0]
            if foods:
                food = foods[0]
                inv[food] -= 1
                if inv[food] <= 0:
                    del inv[food]
                self.state["health"] = min(20, self.state.get("health", 20) + 4)
                self.state["hunger"] = min(20, self.state.get("hunger", 20) + 4)
                result["outcome"] = f"吃了 {food}，恢复生命和饥饿"
            else:
                result["success"] = False
                result["outcome"] = "无食物可吃"

        elif act == "build_shelter":
            inv = self.state.setdefault("inventory", {})
            blocks = inv.get("cobble", 0) + inv.get("wood", 0)
            if blocks >= 10:
                # 消耗材料
                used = 0
                for mat in ("cobble", "wood"):
                    take = min(inv.get(mat, 0), 10 - used)
                    inv[mat] = inv.get(mat, 0) - take
                    used += take
                    if inv[mat] <= 0:
                        inv.pop(mat, None)
                    if used >= 10:
                        break
                self.state["has_shelter"] = True
                result["outcome"] = "建造了庇护所"
            else:
                result["success"] = False
                result["outcome"] = "建材不足"

        elif act == "light_area":
            inv = self.state.setdefault("inventory", {})
            if inv.get("torch", 0) > 0:
                inv["torch"] -= 1
                result["outcome"] = "放置了火把"
            else:
                result["outcome"] = "放置了简易照明"

        elif act in ("explore", "move_to"):
            pos = self.state.get("position", {"x": 0, "y": 0, "z": 0})
            pos["x"] += 5
            self.state["position"] = pos
            result["outcome"] = "移动到新位置"

        elif act in ("retreat", "flee_from"):
            pos = self.state.get("position", {"x": 0, "y": 0, "z": 0})
            pos["x"] -= 10
            self.state["position"] = pos
            # 远离敌人
            entities = self.state.get("nearby_entities", [])
            self.state["nearby_entities"] = [
                e for e in entities
                if not (isinstance(e, dict) and e.get("hostile", False) and e.get("distance", 0) < 5)
            ]
            for e in self.state.get("nearby_entities", []):
                if isinstance(e, dict):
                    e["distance"] = e.get("distance", 0) + 8
            result["outcome"] = "撤退远离敌人"

        elif act == "attack" or act == "attack_entity":
            entities = self.state.get("nearby_entities", [])
            if entities:
                target = entities[0]
                if isinstance(target, dict):
                    entities.pop(0)
                    result["outcome"] = f"击杀 {target.get('type', 'entity')}"
                    # 战斗可能受伤
                    if target.get("hostile", False):
                        dmg = 2
                        self.state["health"] = max(0, self.state.get("health", 20) - dmg)
                        result["damage_taken"] = dmg
                    # 被动生物掉落食物
                    if not target.get("hostile", False):
                        inv = self.state.setdefault("inventory", {})
                        inv["cooked_meat"] = inv.get("cooked_meat", 0) + 1
                        result["resources_gained"] = 1
            else:
                result["success"] = False
                result["outcome"] = "附近无实体"

        elif act == "find_resource":
            resource = args.get("resource", args.get("resource_type", "tree"))
            if resource in ("tree", "wood"):
                blocks = self.state.setdefault("nearby_blocks", [])
                if "tree" not in blocks:
                    blocks.append("tree")
            result["outcome"] = f"定位到 {resource}"

        elif act in ("swim", "jump"):
            self.state["breath"] = min(10, self.state.get("breath", 10) + 3)
            result["outcome"] = "浮出水面"

        elif act == "wait":
            self.state["hunger"] = max(0, self.state.get("hunger", 20) - 0.5)
            result["outcome"] = "等待中"

        elif act == "smelt":
            inv = self.state.setdefault("inventory", {})
            if inv.get("iron_lump", 0) > 0 and inv.get("coal_lump", 0) > 0:
                inv["iron_lump"] -= 1
                inv["coal_lump"] -= 1
                inv["iron_ingot"] = inv.get("iron_ingot", 0) + 1
                result["outcome"] = "冶炼了铁锭"
            else:
                result["success"] = False
                result["outcome"] = "材料不足"

        elif act == "farm_harvest":
            inv = self.state.setdefault("inventory", {})
            inv["wheat"] = inv.get("wheat", 0) + 2
            inv["wheat_seed"] = inv.get("wheat_seed", 0) + 1
            result["outcome"] = "收获农作物"

        else:
            result["outcome"] = f"执行了 {act}"

        # ── 全局环境衰减 ──

        # 饥饿缓慢下降
        if self.step_count % 3 == 0:
            self.state["hunger"] = max(0, self.state.get("hunger", 20) - 1)
        # 饥饿过低开始掉血
        if self.state.get("hunger", 20) <= 0:
            self.state["health"] = max(0, self.state.get("health", 20) - 1)

        # 夜间每 5 步可能刷怪
        if self.state.get("time") == "night" and self.step_count % 5 == 0:
            if not self.state.get("has_shelter", False):
                entities = self.state.setdefault("nearby_entities", [])
                entities.append({"type": "zombie", "hostile": True, "distance": 8})

        self.events.append({"step": self.step_count, "action": act, "result": result})
        return result


# ════════════════════════════════════════════════
#  基准定义
# ════════════════════════════════════════════════

@dataclass
class BenchmarkDef:
    """基准测试定义"""
    id: str
    name: str
    description: str
    initial_state: Dict[str, Any]
    max_steps: int
    success_condition: Callable[[Dict[str, Any], int], bool]
    scoring: Callable[[Dict[str, Any], int, List], Dict[str, float]]


def _default_initial_state() -> Dict[str, Any]:
    return {
        "time": "day",
        "health": 20,
        "hunger": 20,
        "position": {"x": 0, "y": 10, "z": 0},
        "inventory": {},
        "nearby_entities": [],
        "nearby_blocks": ["tree", "dirt", "stone"],
        "has_shelter": False,
    }


# ═══ 基准 1: 存活 N 步 ═══

def _survive_success(state, steps):
    return state.get("health", 0) > 0 and steps >= 50

def _survive_score(state, steps, events):
    health_kept = state.get("health", 0) / 20.0
    hunger_kept = state.get("hunger", 0) / 20.0
    return {
        "survived_steps": steps,
        "health_retention": health_kept,
        "hunger_retention": hunger_kept,
        "overall": (steps / 50.0) * 0.5 + health_kept * 0.3 + hunger_kept * 0.2,
    }


# ═══ 基准 2: 资源采集效率 ═══

def _gather_success(state, steps):
    inv = state.get("inventory", {})
    return inv.get("wood", 0) >= 10 and inv.get("cobble", 0) >= 5

def _gather_score(state, steps, events):
    inv = state.get("inventory", {})
    wood = min(inv.get("wood", 0), 10) / 10.0
    stone = min(inv.get("cobble", 0), 5) / 5.0
    efficiency = 1.0 - min(steps, 30) / 30.0  # 越快越好
    return {
        "wood_collected": inv.get("wood", 0),
        "stone_collected": inv.get("cobble", 0),
        "steps_used": steps,
        "efficiency": efficiency,
        "overall": wood * 0.35 + stone * 0.35 + efficiency * 0.3,
    }


# ═══ 基准 3: 庇护所建造 ═══

def _shelter_success(state, steps):
    return state.get("has_shelter", False)

def _shelter_score(state, steps, events):
    built = 1.0 if state.get("has_shelter", False) else 0.0
    speed = 1.0 - min(steps, 20) / 20.0
    return {
        "shelter_built": state.get("has_shelter", False),
        "steps_used": steps,
        "overall": built * 0.7 + speed * 0.3,
    }


# ═══ 基准 4: 战斗存活 ═══

def _combat_success(state, steps):
    return state.get("health", 0) > 0

def _combat_score(state, steps, events):
    kills = sum(1 for e in events if "击杀" in e.get("result", {}).get("outcome", ""))
    health_kept = state.get("health", 0) / 20.0
    return {
        "kills": kills,
        "health_retention": health_kept,
        "survived_steps": steps,
        "overall": health_kept * 0.5 + min(kills, 5) / 5.0 * 0.5,
    }


# ═══ 基准 5: 科技树推进 ═══

def _tech_success(state, steps):
    inv = state.get("inventory", {})
    return any(k for k in inv if "stone" in k and ("pickaxe" in k or "sword" in k))

def _tech_score(state, steps, events):
    inv = state.get("inventory", {})
    has_wooden = any(k for k in inv if "wooden" in k)
    has_stone = any(k for k in inv if "stone" in k and ("pickaxe" in k or "sword" in k))
    has_iron = any(k for k in inv if "iron" in k and ("pickaxe" in k or "sword" in k))
    tier = 0
    if has_wooden:
        tier = 1
    if has_stone:
        tier = 2
    if has_iron:
        tier = 3
    speed = 1.0 - min(steps, 40) / 40.0
    return {
        "tech_tier": tier,
        "steps_used": steps,
        "overall": tier / 3.0 * 0.7 + speed * 0.3,
    }


# ════════════════════════════════════════════════
#  基准注册表
# ════════════════════════════════════════════════

BUILTIN_BENCHMARKS = [
    BenchmarkDef(
        id="survive_50",
        name="存活 50 步",
        description="在标准环境中存活 50 步，衡量生存决策质量",
        initial_state=_default_initial_state(),
        max_steps=50,
        success_condition=_survive_success,
        scoring=_survive_score,
    ),
    BenchmarkDef(
        id="gather_basic",
        name="基础资源采集",
        description="采集 10 木头 + 5 圆石（30 步内完成最佳）",
        initial_state=_default_initial_state(),
        max_steps=30,
        success_condition=_gather_success,
        scoring=_gather_score,
    ),
    BenchmarkDef(
        id="build_shelter",
        name="建造庇护所",
        description="使用采集的资源建造庇护所（20 步内最佳）",
        initial_state={
            **_default_initial_state(),
            "time": "day",
            "inventory": {"wood": 5},
        },
        max_steps=20,
        success_condition=_shelter_success,
        scoring=_shelter_score,
    ),
    BenchmarkDef(
        id="combat_survival",
        name="战斗存活",
        description="面对多波敌人存活，衡量战斗与撤退决策",
        initial_state={
            **_default_initial_state(),
            "time": "night",
            "inventory": {"stone_sword": 1, "apple": 3, "cobble": 10},
            "nearby_entities": [
                {"type": "zombie", "hostile": True, "distance": 6},
                {"type": "zombie", "hostile": True, "distance": 10},
            ],
        },
        max_steps=30,
        success_condition=_combat_success,
        scoring=_combat_score,
    ),
    BenchmarkDef(
        id="tech_progression",
        name="科技树推进",
        description="从空手开始推进到石制工具（40 步内最佳）",
        initial_state=_default_initial_state(),
        max_steps=40,
        success_condition=_tech_success,
        scoring=_tech_score,
    ),
]


# ════════════════════════════════════════════════
#  评测结果
# ════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """单个基准结果"""
    benchmark_id: str
    benchmark_name: str
    success: bool
    steps_used: int
    final_state: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "success": self.success,
            "steps_used": self.steps_used,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class BenchmarkReport:
    """基准报告"""
    provider: str
    model: str
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_score: float = 0.0

    def compute(self):
        if self.results:
            self.overall_score = (
                sum(r.scores.get("overall", 0) for r in self.results) / len(self.results)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "overall_score": round(self.overall_score, 4),
            "benchmarks": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            f"═══ Benchmark Report ═══",
            f"Provider: {self.provider} | Model: {self.model}",
            f"Overall Score: {self.overall_score:.1%}",
            "",
        ]
        for r in self.results:
            status = "✓" if r.success else "✗"
            overall = r.scores.get("overall", 0)
            lines.append(f"  {status} {r.benchmark_name:<20} score={overall:.2f}  "
                         f"steps={r.steps_used}  {r.latency_ms:.0f}ms")
            for k, v in r.scores.items():
                if k != "overall":
                    lines.append(f"      {k}: {v}")
        return "\n".join(lines)


# ════════════════════════════════════════════════
#  评测器
# ════════════════════════════════════════════════

class BenchmarkHarness:
    """端到端基准评测器"""

    def __init__(self,
                 llm_provider: str = "mock",
                 llm_model: str = None,
                 config: AgentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""

    def run(self, benchmark_ids: List[str] = None) -> BenchmarkReport:
        """执行基准测试"""
        benchmarks = BUILTIN_BENCHMARKS
        if benchmark_ids:
            benchmarks = [b for b in benchmarks if b.id in benchmark_ids]

        llm = LLMClient(provider=self.llm_provider, model=self.llm_model or None)
        model_name = self.llm_model or self.llm_provider

        report = BenchmarkReport(provider=self.llm_provider, model=model_name)

        for bm in benchmarks:
            result = self._run_benchmark(bm, llm)
            report.results.append(result)

        report.compute()
        return report

    def _run_benchmark(self, bm: BenchmarkDef, llm: LLMClient) -> BenchmarkResult:
        """运行单个基准"""
        env = SimulatedEnvironment(bm.initial_state)
        memory = MemoryManager()
        core = AgentCore(config=self.config, llm_client=llm, memory_manager=memory)

        t0 = time.time()
        steps = 0

        for _ in range(bm.max_steps):
            state = env.get_state()

            # 检查死亡
            if state.get("health", 0) <= 0:
                break

            # 决策
            try:
                decision = core.decide(state)
            except Exception as e:
                logger.warning(f"基准 {bm.id} 决策异常: {e}")
                break

            # 逐步执行计划
            for action in decision.action_plan:
                if steps >= bm.max_steps:
                    break
                action_result = env.execute_action(action)
                core.step(action_result)
                steps += 1

                # 中途检查成功条件
                if bm.success_condition(env.get_state(), steps):
                    break

            if bm.success_condition(env.get_state(), steps):
                break

        elapsed = (time.time() - t0) * 1000
        final_state = env.get_state()
        success = bm.success_condition(final_state, steps)
        scores = bm.scoring(final_state, steps, env.events)

        return BenchmarkResult(
            benchmark_id=bm.id,
            benchmark_name=bm.name,
            success=success,
            steps_used=steps,
            final_state=final_state,
            scores=scores,
            events=env.events,
            latency_ms=elapsed,
        )

    def save_report(self, report: BenchmarkReport, filepath: str):
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
