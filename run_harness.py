#!/usr/bin/env python3
"""
Harness Engineering — 统一评测入口

用法:
  # 运行全部评测（mock 模式，无需 GPU）
  python run_harness.py

  # 仅运行决策评测
  python run_harness.py --suite decision

  # 用真实 LLM 运行
  python run_harness.py --llm local --model qwen2.5:7b-instruct

  # 仅运行特定类别场景
  python run_harness.py --suite decision --category survival

  # 运行基准测试
  python run_harness.py --suite benchmark

  # 运行全部并保存报告
  python run_harness.py --save

可用套件:
  decision    — LLM 决策质量评测（50+ 场景）
  benchmark   — 端到端生存基准（5 项基准）
  memory      — 记忆系统评测（检索/规则/技能/置信度）
  prompt_ab   — Prompt A/B 对比测试
  reflection  — 反思校准评测
  all         — 全部运行
"""

import argparse
import json
import logging
import os
import sys
import time

from config import DEFAULT_CONFIG
from harness.decision_harness import DecisionHarness
from harness.benchmark_harness import BenchmarkHarness
from harness.memory_harness import MemoryHarness
from harness.prompt_ab_harness import PromptABHarness
from harness.reflection_harness import ReflectionHarness


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_decision(args) -> dict:
    print("\n" + "=" * 60)
    print("  Decision Harness — LLM 决策质量评测")
    print("=" * 60)

    h = DecisionHarness(
        llm_provider=args.llm,
        llm_model=args.model,
    )
    report = h.run(
        category=args.category,
        difficulty=args.difficulty,
    )
    print(report.summary())

    if args.save:
        path = os.path.join(args.output, "decision_report.json")
        h.save_report(report, path)
        print(f"\n报告已保存: {path}")

    return report.to_dict()


def run_benchmark(args) -> dict:
    print("\n" + "=" * 60)
    print("  Benchmark Harness — 端到端生存基准")
    print("=" * 60)

    h = BenchmarkHarness(
        llm_provider=args.llm,
        llm_model=args.model,
    )
    report = h.run()
    print(report.summary())

    if args.save:
        path = os.path.join(args.output, "benchmark_report.json")
        h.save_report(report, path)
        print(f"\n报告已保存: {path}")

    return report.to_dict()


def run_memory(args) -> dict:
    print("\n" + "=" * 60)
    print("  Memory Harness — 记忆系统评测")
    print("=" * 60)

    h = MemoryHarness()
    report = h.run()
    print(report.summary())

    if args.save:
        path = os.path.join(args.output, "memory_report.json")
        h.save_report(report, path)
        print(f"\n报告已保存: {path}")

    return report.to_dict()


def run_prompt_ab(args) -> dict:
    print("\n" + "=" * 60)
    print("  Prompt A/B Harness — Prompt 变体对比")
    print("=" * 60)

    h = PromptABHarness(
        llm_provider=args.llm,
        llm_model=args.model,
    )
    report = h.run(runs_per_scenario=args.ab_runs)
    print(report.summary())

    if args.save:
        path = os.path.join(args.output, "prompt_ab_report.json")
        h.save_report(report, path)
        print(f"\n报告已保存: {path}")

    return report.to_dict()


def run_reflection(args) -> dict:
    print("\n" + "=" * 60)
    print("  Reflection Harness — 反思校准评测")
    print("=" * 60)

    h = ReflectionHarness(
        llm_provider=args.llm,
        llm_model=args.model,
    )
    report = h.run()
    print(report.summary())

    if args.save:
        path = os.path.join(args.output, "reflection_report.json")
        h.save_report(report, path)
        print(f"\n报告已保存: {path}")

    return report.to_dict()


SUITE_MAP = {
    "decision": run_decision,
    "benchmark": run_benchmark,
    "memory": run_memory,
    "prompt_ab": run_prompt_ab,
    "reflection": run_reflection,
}


def main():
    parser = argparse.ArgumentParser(
        description="Luanti 智能体 — Harness Engineering 评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_harness.py                              # 全部评测 (mock)
  python run_harness.py --suite decision             # 仅决策评测
  python run_harness.py --suite benchmark --llm local  # 基准 (Ollama)
  python run_harness.py --suite all --save           # 全部 + 保存报告
        """,
    )

    parser.add_argument(
        "--suite", type=str, default="all",
        choices=["all", "decision", "benchmark", "memory", "prompt_ab", "reflection"],
        help="要运行的评测套件 (默认: all)",
    )
    parser.add_argument(
        "--llm", type=str, default="mock",
        choices=["mock", "local", "openai", "anthropic"],
        help="LLM 提供者 (默认: mock — 无需 GPU)",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM 模型名")
    parser.add_argument("--category", type=str, default=None,
                        help="决策评测: 过滤场景类别 (survival/combat/resource/craft/explore)")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="决策评测: 过滤难度 (easy/medium/hard)")
    parser.add_argument("--ab-runs", type=int, default=1,
                        help="Prompt A/B: 每场景每变体重复次数 (默认: 1)")
    parser.add_argument("--save", action="store_true",
                        help="保存评测报告到 --output 目录")
    parser.add_argument("--output", type=str, default="./logs/harness",
                        help="报告输出目录 (默认: ./logs/harness)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细日志输出")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.save:
        os.makedirs(args.output, exist_ok=True)

    print("╔══════════════════════════════════════════════╗")
    print("║   Luanti Agent — Harness Engineering 评测   ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  LLM: {args.llm}" + (f" ({args.model})" if args.model else ""))

    t0 = time.time()
    results = {}

    if args.suite == "all":
        for name, fn in SUITE_MAP.items():
            try:
                results[name] = fn(args)
            except Exception as e:
                print(f"\n⚠ {name} 评测失败: {e}")
                logging.exception(e)
    else:
        fn = SUITE_MAP[args.suite]
        try:
            results[args.suite] = fn(args)
        except Exception as e:
            print(f"\n⚠ {args.suite} 评测失败: {e}")
            logging.exception(e)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"  全部评测完成  耗时: {elapsed:.1f}s")
    print("=" * 60)

    # 保存汇总报告
    if args.save:
        summary_path = os.path.join(args.output, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "llm": args.llm,
                "model": args.model,
                "elapsed_seconds": round(elapsed, 1),
                "suites": list(results.keys()),
            }, f, ensure_ascii=False, indent=2)
        print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
