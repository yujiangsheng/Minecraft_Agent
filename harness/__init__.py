"""
Harness Engineering — 智能体评测与校准框架

作者: Jiangsheng Yu
许可证: MIT License

五大评测模块：
  1. DecisionHarness    — LLM 决策质量评测（场景库 → 期望动作 → 自动评分）
  2. BenchmarkHarness   — 端到端生存基准测试（存活率、资源效率、任务完成）
  3. MemoryHarness      — 记忆系统评测（检索召回率、规则置信度、技能触发精度）
  4. PromptABHarness    — Prompt A/B 对比测试（同场景多 prompt 变体统计比较）
  5. ReflectionHarness  — 反思校准评测（注入已知失败 → 检查诊断准确性）
"""

from harness.scenarios import ScenarioLibrary, Scenario
from harness.decision_harness import DecisionHarness
from harness.benchmark_harness import BenchmarkHarness
from harness.memory_harness import MemoryHarness
from harness.prompt_ab_harness import PromptABHarness
from harness.reflection_harness import ReflectionHarness

__all__ = [
    "ScenarioLibrary", "Scenario",
    "DecisionHarness",
    "BenchmarkHarness",
    "MemoryHarness",
    "PromptABHarness",
    "ReflectionHarness",
]
