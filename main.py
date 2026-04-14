#!/usr/bin/env python3
"""
Luanti 自主智能体 — 主程序入口

作者: Jiangsheng Yu
许可证: MIT License

运行模式:
  python main.py --mode demo                  # 模拟环境演示
  python main.py --mode luanti                # 连接 Luanti 游戏（默认 Ollama LLM）
  python main.py --mode luanti --llm mock     # 连接 Luanti（mock 模式测试）
  python main.py --mode interactive           # 交互模式（手动输入状态 JSON）
  python main.py --mode evolve                # 遗传算法演化模式
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Any, Optional, List

from config import AgentConfig, DEFAULT_CONFIG
from agent import AgentCore, Planner, QuickReflection, LongReflection, SkillBuilder, IntentUnderstanding, EvolutionOperator
from memory import MemoryManager
from utils import LLMClient
from luanti_env import LuantiEnvironment, ActionTranslator
from web_dashboard import WebDashboard


class MinecraftAgent:
    """Minecraft 智能体主类"""
    
    def __init__(self, 
                 config: AgentConfig = None,
                 llm_provider: str = "mock",
                 storage_path: str = "./data"):
        
        self.config = config or DEFAULT_CONFIG
        
        # 初始化 LLM 客户端
        self.llm = LLMClient(
            provider=llm_provider,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature
        )
        
        # 初始化记忆管理器
        self.memory = MemoryManager(
            episodic_max_size=self.config.memory.episodic_max_size,
            semantic_max_rules=self.config.memory.semantic_max_rules,
            skills_max_count=self.config.memory.skills_max_count,
            trajectory_window=self.config.memory.trajectory_window
        )
        self.memory.set_storage_path(storage_path)
        
        # 初始化核心模块
        self.core = AgentCore(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.planner = Planner(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.quick_reflection = QuickReflection(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.long_reflection = LongReflection(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.skill_builder = SkillBuilder(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.intent_understanding = IntentUnderstanding(
            config=self.config,
            llm_client=self.llm,
            memory_manager=self.memory
        )
        
        self.evolution = EvolutionOperator(
            config=self.config,
            llm_client=self.llm
        )
        
        # 存储路径
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def step(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步决策
        
        Args:
            env_state: 环境状态
            
        Returns:
            决策结果，包含动作计划
        """
        # 1. 感知并决策
        decision = self.core.decide(env_state)
        
        return {
            "decision": decision.to_dict(),
            "next_action": self.core.get_next_action(),
            "agent_state": self.core.state.to_dict()
        }
    
    def receive_result(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """接收动作执行结果
        
        Args:
            action_result: 动作执行结果
            
        Returns:
            处理结果，包含是否需要重新规划
        """
        # 更新状态
        replan_needed = self.core.step(action_result)
        
        # 如果需要，执行快速反思
        reflection_result = None
        if self.config.reflection.quick_reflection_enabled:
            current_plan = self.core.current_plan.to_dict() if self.core.current_plan else {}
            reflection_result = self.quick_reflection.reflect(
                current_state=self.core.current_env_state,
                plan=current_plan,
                execution_result=action_result,
                recent_trajectory=self.memory.get_recent_trajectory(5)
            )
            
            if reflection_result.status == "failure":
                replan_needed = True
        
        return {
            "replan_needed": replan_needed,
            "reflection": reflection_result.to_dict() if reflection_result else None,
            "next_action": self.core.get_next_action() if not replan_needed else None
        }
    
    def end_episode(self, outcome: str = "unknown") -> Dict[str, Any]:
        """结束当前局
        
        Args:
            outcome: 结局描述 (success/failure/timeout)
            
        Returns:
            复盘结果
        """
        # 获取完整轨迹
        episode_data = self.core.end_episode(outcome)
        
        # 执行长期复盘
        reflection_result = None
        if self.config.reflection.long_reflection_enabled:
            reflection_result = self.long_reflection.reflect(
                episode_trajectory=episode_data["trajectory"],
                episode_outcome=outcome,
                episode_stats={
                    "total_steps": episode_data["total_steps"],
                    "episode_id": episode_data["episode_id"]
                }
            )
        
        # 尝试自动生成技能
        new_skills = self.skill_builder.auto_generate_skills(min_frequency=2)
        for skill in new_skills:
            self.skill_builder.commit_skill(skill)
        
        return {
            "episode_data": episode_data,
            "reflection": reflection_result.to_dict() if reflection_result else None,
            "new_skills_generated": len(new_skills)
        }
    
    def start_new_episode(self):
        """开始新的一局"""
        self.core.start_new_episode()
    
    def switch_llm_provider(self, provider: str, model: str = None, api_base: str = None):
        """运行时切换 LLM 提供者

        Args:
            provider: 提供者名称 (local/openai/anthropic/mock)
            model: 模型名称（可选，使用默认值）
            api_base: API 地址（可选，仅 local 模式需要）
        """
        new_llm = LLMClient(
            provider=provider,
            model=model or self.config.llm.model,
            temperature=self.config.llm.temperature,
            api_base=api_base,
        )
        self.llm = new_llm
        self.core.llm = new_llm
        self.planner.llm = new_llm
        self.quick_reflection.llm = new_llm
        self.long_reflection.llm = new_llm
        self.skill_builder.llm = new_llm
        self.intent_understanding.llm = new_llm
        self.evolution.llm = new_llm

    def save(self):
        """保存状态"""
        self.memory.save()
        self.evolution.save(os.path.join(self.storage_path, "evolution.json"))
    
    def load(self):
        """加载状态"""
        try:
            self.memory.load()
        except FileNotFoundError:
            pass
        
        evolution_path = os.path.join(self.storage_path, "evolution.json")
        if os.path.exists(evolution_path):
            self.evolution.load(evolution_path)
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_state": self.core.state.to_dict(),
            "memory_stats": self.memory.get_stats(),
            "reflection_stats": {
                "quick": self.quick_reflection.get_failure_statistics(),
                "long": self.long_reflection.get_insights_summary()
            },
            "evolution_stats": self.evolution.get_evolution_summary()
        }


def run_luanti(llm_provider: str = "local", port: int = 8765, data_path: str = "./luanti_data",
               web_port: int = 8080):
    """连接 Luanti 游戏运行智能体"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logger = logging.getLogger("LuantiAgent")

    print("=" * 60)
    print("Luanti 自主智能体 - 实时游戏模式")
    print("=" * 60)

    # 创建核心组件
    agent = MinecraftAgent(llm_provider=llm_provider, storage_path=data_path)
    agent.start_new_episode()
    env = LuantiEnvironment(host="localhost", port=port)
    dashboard = WebDashboard(host="0.0.0.0", port=web_port)
    dashboard.state.update_memory_stats(agent.memory.get_stats())
    dashboard.state.set_llm_provider(llm_provider)

    print(f"\n[初始化完成] LLM: {llm_provider}")
    print(f"记忆状态: {agent.memory.get_stats()}")

    # 创建会话管理器并绑定回调
    session = LuantiSession(agent, env, dashboard, logger, port)
    session.bind_callbacks()

    # 初始化演化统计
    dashboard.state.update_evolution_stats(agent.evolution.get_evolution_summary())

    # 启动服务器
    env.start()
    dashboard.start()

    dashboard.state.add_log({
        "time": time.strftime("%H:%M:%S"),
        "type": "system",
        "message": f"智能体已启动 | LLM: {llm_provider} | 桥接端口: {port}"
    })

    print(f"\n[HTTP 服务器已启动] http://localhost:{port}")
    print(f"[Web 控制面板] http://localhost:{web_port}")
    print("\n请在 Web 控制面板中点击「启动 Luanti」，或手动启动 Luanti 并进入游戏。")
    print("等待 Luanti 连接...\n")

    # 主循环
    try:
        session.run_main_loop()
    except KeyboardInterrupt:
        print("\n\n[停止智能体]")
        try:
            agent.end_episode("manual_stop")
            agent.save()
            print("[状态已保存]")
        except Exception as e:
            print(f"保存出错: {e}")
    finally:
        dashboard.stop()
        env.stop()
        print("[服务器已关闭]")


class LuantiSession:
    """Luanti 游戏会话管理器 — 封装所有回调逻辑和状态追踪"""

    # 值得积累经验的重要动作类型
    NOTABLE_ACTIONS = {"craft", "build_shelter", "smelt", "dig", "eat", "light_area", "farm_plant", "farm_harvest"}

    def __init__(self, agent: MinecraftAgent, env: LuantiEnvironment,
                 dashboard: WebDashboard, logger, port: int):
        self.agent = agent
        self.env = env
        self.dashboard = dashboard
        self.logger = logger
        self.port = port

        self.step_count = 0
        self.episode_steps = 0
        self.last_episode_end = time.time()

        # 连续成功追踪（用于发现高效模式）
        self._success_streak: List[str] = []
        self._last_learned_step = 0

        # RL 训练子进程
        self._rl_process: Optional[subprocess.Popen] = None

    # ── 核心回调 ──

    def on_decision(self, env_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """决策回调 — 每次收到游戏状态时调用"""
        self.step_count += 1
        self.episode_steps += 1

        # 同步用户任务
        user_task = self.dashboard.state.get_user_task()
        self.agent.core.user_task = user_task

        # 意图理解：当有用户任务时，生成分阶段执行计划并注入决策上下文
        if user_task:
            try:
                plan = self.agent.intent_understanding.plan_execution(user_task, env_state)
                self.agent.core.user_task_decomposition = plan.to_dict()
                self.logger.info(
                    f"执行计划: phase={plan.phase}, "
                    f"未满足条件={len(plan.unmet_conditions)}, "
                    f"准备动作={len(plan.prerequisite_actions)}, "
                    f"主任务动作={len(plan.main_task_actions)}"
                )
            except Exception as e:
                self.logger.error(f"执行计划生成出错: {e}")
                self.agent.core.user_task_decomposition = None
        else:
            self.agent.core.user_task_decomposition = None

        # 更新 Dashboard 状态
        self.dashboard.state.update_env_state(env_state)
        self.dashboard.state.update_step_count(self.step_count)
        self.dashboard.state.update_episode_info(episode_count=0, episode_steps=self.episode_steps)

        try:
            result = self.agent.step(env_state)
            decision = result["decision"]
            agent_state = result.get("agent_state", {})

            self.logger.info(
                f"决策 #{self.step_count}: 模式={decision['mode']}, "
                f"优先={agent_state.get('priority_issue', '?')}, "
                f"目标={decision['goal']}, "
                f"动作数={len(decision['action_plan'])}"
            )

            self.dashboard.state.update_agent_state(agent_state)
            self.dashboard.state.update_memory_stats(self.agent.memory.get_stats())

            task_suffix = f" [任务: {user_task}]" if user_task else ""
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "decision",
                "message": (
                    f"#{self.step_count} 模式={decision['mode']} "
                    f"目标={decision['goal']} "
                    f"动作={[a.get('action') for a in decision['action_plan']]}"
                    f"{task_suffix}"
                )
            })

            agent_actions = decision.get("action_plan", [])
            return ActionTranslator.translate(agent_actions, env_state)

        except Exception as e:
            self.logger.error(f"决策出错: {e}")
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "error",
                "message": f"决策出错: {e}"
            })
            return [{"action": "look_around", "params": {}}]

    def on_result(self, action_result: Dict[str, Any]):
        """动作结果回调"""
        formatted_result = {
            "success": action_result.get("success", False),
            "action": action_result.get("action", "unknown"),
            "outcome": action_result.get("outcome", ""),
            "error_type": None if action_result.get("success") else "action_failed",
            "resources_gained": 1 if action_result.get("success") else 0,
            "damage_taken": 0,
        }

        status_icon = "✓" if formatted_result["success"] else "✗"
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "action",
            "message": f"{status_icon} {formatted_result['action']}: {formatted_result['outcome']}"
        })

        try:
            if not formatted_result["success"]:
                self.agent.core.record_failure(
                    action=formatted_result["action"],
                    reason=formatted_result.get("outcome", "action_failed")
                )
                self._success_streak.clear()
            else:
                self._accumulate_experience(formatted_result)

            step_result = self.agent.receive_result(formatted_result)

            if step_result.get("replan_needed"):
                self.logger.info("[需要重新规划]")

            if step_result.get("reflection"):
                ref = step_result["reflection"]
                if ref.get("status") != "ok":
                    self.logger.info(f"反思: {ref.get('status')} - {ref.get('cause', '')}")
                    self.dashboard.state.add_log({
                        "time": time.strftime("%H:%M:%S"),
                        "type": "reflect",
                        "message": f"反思: {ref.get('status')} - {ref.get('cause', '')}"
                    })

        except Exception as e:
            self.logger.error(f"处理结果出错: {e}")

        # 定期 episode 结束和复盘
        if self.episode_steps >= 50:
            self._end_and_restart_episode()

    def _accumulate_experience(self, result: Dict[str, Any]):
        """实时经验积累：成功动作 → 情景记忆 / 语义规则"""
        action_name = result["action"]
        self._success_streak.append(action_name)

        # 重要动作成功 → 存储为情景记忆
        if action_name in self.NOTABLE_ACTIONS and self.step_count - self._last_learned_step >= 5:
            env_state = self.agent.core.current_env_state
            self.agent.memory.store_episode(
                summary=f"{action_name} 成功: {result['outcome']}",
                lesson=f"在 {env_state.get('time', '?')} 时段，附近有 {env_state.get('nearby_blocks', [])[:3]} 时，{action_name} 可以成功执行",
                tags=[action_name, "success", env_state.get("time", "unknown")],
                context=env_state,
                outcome="success"
            )
            self._last_learned_step = self.step_count

        # 连续成功 3+ 个动作 → 提取为语义规则
        if len(self._success_streak) >= 3 and self.step_count - self._last_learned_step >= 10:
            pattern = " → ".join(self._success_streak[-3:])
            self.agent.memory.store_rule(
                rule=f"动作序列 [{pattern}] 在当前条件下有效",
                confidence=0.7,
                conditions=self.agent.memory._extract_conditions(self.agent.core.current_env_state)
            )
            self._last_learned_step = self.step_count
            self.logger.info(f"[经验积累] 学到有效动作模式: {pattern}")
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "learn",
                "message": f"学到有效模式: {pattern}"
            })

    def _end_and_restart_episode(self):
        """结束当前 episode 并开始新的"""
        try:
            self.logger.info("[结束当前 episode，执行复盘]")
            episode_result = self.agent.end_episode("auto_cycle")
            if episode_result.get("reflection"):
                self.logger.info(f"复盘: {episode_result['reflection'].get('episode_summary', '')}")
                self.dashboard.state.add_log({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "reflect",
                    "message": f"复盘: {episode_result['reflection'].get('episode_summary', '')}"
                })
            self.logger.info(f"新技能: {episode_result['new_skills_generated']}")

            self.agent.save()
            self.agent.start_new_episode()
            self.episode_steps = 0
            self.last_episode_end = time.time()
            self.dashboard.state.update_evolution_stats(self.agent.evolution.get_evolution_summary())
        except Exception as e:
            self.logger.error(f"Episode 结束出错: {e}")

    # ── Dashboard 回调 ──

    def do_shutdown(self):
        """退出回调"""
        self.logger.info("[Web Dashboard 请求退出]")
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": "用户请求退出，正在保存状态..."
        })
        try:
            self.agent.end_episode("web_shutdown")
            self.agent.save()
            self.logger.info("[状态已保存]")
        except Exception as e:
            self.logger.error(f"保存出错: {e}")
        time.sleep(1)
        if self._rl_process is not None and self._rl_process.poll() is None:
            self._rl_process.terminate()
        self.env.stop()
        self.dashboard.stop()
        os._exit(0)

    def do_launch(self):
        """启动 Luanti 回调"""
        self.logger.info("[Web Dashboard 请求启动 Luanti]")
        self.dashboard.state.set_luanti_launched(True)
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": "正在启动 Luanti..."
        })
        try:
            check = subprocess.run(["pgrep", "-f", "Luanti"], capture_output=True, text=True)
            if check.returncode == 0:
                subprocess.Popen(["osascript", "-e", 'tell application "Luanti" to activate'])
                self.dashboard.state.add_log({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "system",
                    "message": "Luanti 已在运行，已切换到前台"
                })
            else:
                subprocess.Popen(["open", "-a", "Luanti"])
                self.dashboard.state.add_log({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "system",
                    "message": "Luanti 已启动，请进入 'wild world' 存档开始游戏"
                })
        except Exception as e:
            self.logger.error(f"启动 Luanti 出错: {e}")
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "error",
                "message": f"启动 Luanti 失败: {e}"
            })

    def do_evolve(self):
        """执行一代演化回调"""
        self.logger.info("[Web Dashboard 请求演化]")
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "evolve",
            "message": "开始执行一代演化..."
        })
        try:
            self.agent.evolution.evolve()
            summary = self.agent.evolution.get_evolution_summary()
            self.dashboard.state.update_evolution_stats(summary)
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "evolve",
                "message": (
                    f"演化完成: 第{summary['current_generation']}代 "
                    f"最佳适应度={summary['best_fitness']:.2f} "
                    f"平均适应度={summary['avg_fitness']:.2f}"
                )
            })
        except Exception as e:
            self.logger.error(f"演化出错: {e}")
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "error",
                "message": f"演化出错: {e}"
            })

    def do_save(self):
        """手动保存回调"""
        self.logger.info("[Web Dashboard 请求保存]")
        try:
            self.agent.save()
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "system",
                "message": "状态已手动保存"
            })
        except Exception as e:
            self.logger.error(f"保存出错: {e}")

    def do_new_episode(self):
        """手动开始新 Episode 回调"""
        self.logger.info("[Web Dashboard 请求新 Episode]")
        try:
            episode_result = self.agent.end_episode("web_manual")
            if episode_result.get("reflection"):
                self.dashboard.state.add_log({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "reflect",
                    "message": f"复盘: {episode_result['reflection'].get('episode_summary', '')}"
                })
            self.agent.save()
            self.agent.start_new_episode()
            self.episode_steps = 0
            self.dashboard.state.update_episode_info(
                episode_count=self.agent.core.episode_count if hasattr(self.agent.core, 'episode_count') else 0,
                episode_steps=0
            )
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "system",
                "message": "已开始新的 Episode"
            })
        except Exception as e:
            self.logger.error(f"新 Episode 出错: {e}")

    # ── 训练模式切换 ──

    def do_training_mode(self, mode: str):
        """训练模式切换回调"""
        current_mode = self.dashboard.state.get_training_mode()
        self.logger.info(f"[训练模式切换] {current_mode} → {mode}")

        # 停止旧的 RL 进程
        if self._rl_process is not None and self._rl_process.poll() is None:
            self.logger.info("[停止 RL 训练进程]")
            self._rl_process.terminate()
            try:
                self._rl_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._rl_process.kill()
            self._rl_process = None
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "system",
                "message": "已停止 RL 训练进程"
            })

        # 切换模式
        self.dashboard.state.set_training_mode(mode)

        if mode == "llm":
            if self.env._shutdown_flag:
                self.env._shutdown_flag = False
                self.env.start()
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "system",
                "message": "已切换到 LLM 决策模式"
            })
        else:
            self._start_rl_training(mode)

    def _start_rl_training(self, algo: str):
        """启动 RL 训练子进程"""
        if not self.env._shutdown_flag:
            self.env.stop()
            self.dashboard.state.set_connected(False)
            time.sleep(1)

        cmd = [
            sys.executable, "train_rl.py",
            "--algo", algo,
            "--total-steps", "50000",
            "--port", str(self.port),
        ]
        self.logger.info(f"[启动 RL 训练] {' '.join(cmd)}")
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": f"正在启动 {algo.upper()} 训练..."
        })

        try:
            self._rl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            import threading as _th
            _th.Thread(
                target=self._monitor_rl_output,
                daemon=True
            ).start()
        except Exception as e:
            self.logger.error(f"启动 RL 训练失败: {e}")
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "error",
                "message": f"启动 RL 训练失败: {e}"
            })
            self.dashboard.state.set_training_mode("llm")
            if self.env._shutdown_flag:
                self.env._shutdown_flag = False
                self.env.start()

    def _monitor_rl_output(self):
        """监控 RL 训练子进程输出"""
        import re
        ep_pattern = re.compile(r'\[Step\s+(\d+)\]\s+Ep\s+(\d+):\s+R=([\d.+-]+)\s+len=(\d+)')
        loss_pattern = re.compile(r'loss=([\d.]+)')
        eps_pattern = re.compile(r'[εε]=([\d.]+)')

        proc = self._rl_process
        if proc is None or proc.stdout is None:
            return

        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            m = ep_pattern.search(line)
            if m:
                stats = {
                    "total_steps": int(m.group(1)),
                    "episodes": int(m.group(2)),
                    "avg_reward": float(m.group(3)),
                }
                lm = loss_pattern.search(line)
                if lm:
                    stats["loss"] = float(lm.group(1))
                em = eps_pattern.search(line)
                if em:
                    stats["epsilon"] = float(em.group(1))
                self.dashboard.state.update_training_stats(stats)
            self.dashboard.state.add_log({
                "time": time.strftime("%H:%M:%S"),
                "type": "system",
                "message": f"[RL] {line}"
            })

        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": "RL 训练进程已结束"
        })

    # ── 绑定与主循环 ──

    def do_llm_provider(self, provider: str, model: str = "", api_base: str = ""):
        """LLM 提供者切换回调"""
        self.logger.info(f"[LLM 切换] → {provider} (model={model or '默认'}, api_base={api_base or '默认'})")
        self.agent.switch_llm_provider(
            provider=provider,
            model=model or None,
            api_base=api_base or None,
        )
        self.dashboard.state.set_llm_provider(provider, model, api_base)
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": f"LLM 已切换到 {provider}" + (f" (模型: {model})" if model else "")
        })

    def do_get_intents(self) -> list:
        """获取已知意图列表回调"""
        return self.agent.intent_understanding.get_known_intents()

    def do_intent_decompose(self, intent: str) -> dict:
        """意图分解回调 — 返回分阶段执行计划"""
        env_state = self.agent.core.current_env_state or {}
        plan = self.agent.intent_understanding.plan_execution(intent, env_state)
        self.dashboard.state.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "system",
            "message": (
                f"意图分解: 「{intent}」→ phase={plan.phase}, "
                f"未满足={len(plan.unmet_conditions)}, "
                f"准备动作={len(plan.prerequisite_actions)}, "
                f"主任务动作={len(plan.main_task_actions)} ({plan.summary})"
            )
        })
        return plan.to_dict()

    def bind_callbacks(self):
        """将所有回调绑定到 env 和 dashboard"""
        self.env.set_decision_callback(self.on_decision)
        self.env.set_result_callback(self.on_result)
        self.dashboard.set_shutdown_callback(self.do_shutdown)
        self.dashboard.set_launch_callback(self.do_launch)
        self.dashboard.set_evolve_callback(self.do_evolve)
        self.dashboard.set_save_callback(self.do_save)
        self.dashboard.set_new_episode_callback(self.do_new_episode)
        self.dashboard.set_training_mode_callback(self.do_training_mode)
        self.dashboard.set_manual_action_callback(self.do_manual_action)
        self.dashboard.set_llm_provider_callback(self.do_llm_provider)
        self.dashboard.set_get_intents_callback(self.do_get_intents)
        self.dashboard.set_intent_decompose_callback(self.do_intent_decompose)

    def do_manual_action(self, actions: list) -> int:
        """手动动作回调 — 将前端选择的动作注入到环境"""
        self.logger.info(f"手动动作: {[a.get('action') for a in actions]}")
        return self.env.inject_manual_actions(actions)

    def run_main_loop(self):
        """主循环：监控连接状态并输出周期报告"""
        was_connected = False
        last_status_step = 0

        while True:
            time.sleep(1)

            if self.env.connected and not was_connected:
                was_connected = True
                print("\n[已连接! 智能体开始运行]")
                print("按 Ctrl+C 停止\n")
                self.dashboard.state.set_connected(True)
                self.dashboard.state.add_log({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "system",
                    "message": "Luanti 已连接，智能体开始运行"
                })

            if was_connected and not self.env.connected:
                elapsed = time.time() - self.env.last_state_time
                if elapsed > 10:
                    was_connected = False
                    self.dashboard.state.set_connected(False)
                    self.dashboard.state.add_log({
                        "time": time.strftime("%H:%M:%S"),
                        "type": "system",
                        "message": "Luanti 连接断开，等待重新连接..."
                    })

            if self.step_count > 0 and self.step_count - last_status_step >= 30:
                last_status_step = self.step_count
                status = self.agent.get_status()
                print(f"\n--- 状态报告 (步骤 #{self.step_count}) ---")
                print(f"  情景记忆: {status['memory_stats']['episodic_count']}")
                print(f"  语义规则: {status['memory_stats']['semantic_rules_count']}")
                print(f"  技能: {status['memory_stats']['skills_count']}")
                print(f"  当前模式: {status['agent_state']['current_mode']}")


def simulate_environment_state(step: int) -> Dict[str, Any]:
    """模拟环境状态（用于演示）"""
    import random
    
    # 模拟日夜循环
    time = "day" if (step // 10) % 2 == 0 else "night"
    
    return {
        "time": time,
        "health": max(5, 20 - random.randint(0, 5)),
        "hunger": max(3, 20 - random.randint(0, 8)),
        "position": {"x": random.randint(-100, 100), "y": 64, "z": random.randint(-100, 100)},
        "inventory": {
            "wood": random.randint(0, 20),
            "cobblestone": random.randint(0, 30),
            "wooden_pickaxe": random.choice([True, False]),
        },
        "nearby_entities": [
            {"type": "pig", "hostile": False, "distance": random.randint(5, 20)}
        ] if random.random() > 0.7 else [],
        "nearby_blocks": random.sample(["tree", "stone", "grass", "dirt"], k=random.randint(1, 4)),
        "has_shelter": random.random() > 0.6
    }


def simulate_action_result(action: Dict[str, Any]) -> Dict[str, Any]:
    """模拟动作执行结果（用于演示）"""
    import random
    
    success = random.random() > 0.2  # 80% 成功率
    
    return {
        "success": success,
        "action": action.get("action", "unknown"),
        "outcome": "completed" if success else "failed",
        "error_type": None if success else random.choice(["action_failed", "precondition_not_met"]),
        "resources_gained": random.randint(0, 3) if success else 0,
        "damage_taken": random.randint(0, 2) if not success else 0
    }


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("Minecraft 自主智能体 - 演示模式")
    print("=" * 60)
    
    # 创建智能体
    agent = MinecraftAgent(llm_provider="mock", storage_path="./demo_data")
    
    print("\n[初始化完成]")
    print(f"记忆状态: {agent.memory.get_stats()}")
    
    # 模拟一局游戏
    print("\n[开始新的一局]")
    agent.start_new_episode()
    
    for step in range(10):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 获取模拟的环境状态
        env_state = simulate_environment_state(step)
        print(f"环境状态: 时间={env_state['time']}, 生命={env_state['health']}, 饥饿={env_state['hunger']}")
        
        # 执行决策
        result = agent.step(env_state)
        decision = result["decision"]
        
        print(f"决策模式: {decision['mode']}")
        print(f"目标: {decision['goal']}")
        print(f"原因: {decision['reason']}")
        print(f"动作计划: {[a.get('action') for a in decision['action_plan']]}")
        
        # 模拟执行动作
        if result["next_action"]:
            action_result = simulate_action_result(result["next_action"])
            print(f"执行结果: {'成功' if action_result['success'] else '失败'}")
            
            # 接收结果
            step_result = agent.receive_result(action_result)
            
            if step_result["replan_needed"]:
                print("[需要重新规划]")
            
            if step_result["reflection"]:
                ref = step_result["reflection"]
                if ref["status"] != "ok":
                    print(f"反思: {ref['status']} - {ref['cause']}")
    
    # 结束这一局
    print("\n[结束这一局]")
    episode_result = agent.end_episode("demo_completed")
    
    if episode_result["reflection"]:
        print(f"复盘摘要: {episode_result['reflection']['episode_summary']}")
    
    print(f"新生成技能数: {episode_result['new_skills_generated']}")
    
    # 保存状态
    agent.save()
    print("\n[状态已保存]")
    
    # 显示最终状态
    status = agent.get_status()
    print(f"\n最终状态:")
    print(f"  - 总步数: {status['agent_state']['total_steps']}")
    print(f"  - 情景记忆: {status['memory_stats']['episodic_count']}")
    print(f"  - 语义规则: {status['memory_stats']['semantic_rules_count']}")
    print(f"  - 技能数量: {status['memory_stats']['skills_count']}")


def run_interactive():
    """交互模式"""
    print("=" * 60)
    print("Minecraft 自主智能体 - 交互模式")
    print("=" * 60)
    print("\n命令:")
    print("  state <json>  - 输入环境状态")
    print("  result <json> - 输入动作结果")
    print("  status        - 查看智能体状态")
    print("  save          - 保存状态")
    print("  load          - 加载状态")
    print("  new           - 开始新的一局")
    print("  end           - 结束当前局")
    print("  quit          - 退出")
    print()
    
    agent = MinecraftAgent(llm_provider="mock", storage_path="./interactive_data")
    agent.start_new_episode()
    
    while True:
        try:
            cmd = input("> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "quit":
                print("再见!")
                break
            
            elif command == "state":
                try:
                    env_state = json.loads(args) if args else simulate_environment_state(0)
                    result = agent.step(env_state)
                    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print("JSON 解析错误")
            
            elif command == "result":
                try:
                    action_result = json.loads(args) if args else {"success": True}
                    result = agent.receive_result(action_result)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print("JSON 解析错误")
            
            elif command == "status":
                status = agent.get_status()
                print(json.dumps(status, ensure_ascii=False, indent=2))
            
            elif command == "save":
                agent.save()
                print("已保存")
            
            elif command == "load":
                agent.load()
                print("已加载")
            
            elif command == "new":
                agent.start_new_episode()
                print("已开始新的一局")
            
            elif command == "end":
                result = agent.end_episode("manual_end")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
            else:
                print(f"未知命令: {command}")
        
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def run_evolution():
    """演化模式"""
    print("=" * 60)
    print("Minecraft 自主智能体 - 演化模式")
    print("=" * 60)
    
    agent = MinecraftAgent(llm_provider="mock", storage_path="./evolution_data")
    
    # 初始化种群
    print("\n[初始化种群]")
    population = agent.evolution.initialize_population()
    print(f"种群大小: {len(population)}")
    
    # 模拟几代演化
    for gen in range(3):
        print(f"\n--- 第 {gen + 1} 代 ---")
        
        # 评估每个候选配置
        for candidate in agent.evolution.population:
            # 模拟评估结果
            import random
            episode_results = [
                {
                    "survived": random.random() > 0.3,
                    "steps": random.randint(10, 100),
                    "resources_collected": random.randint(0, 20),
                    "tools_crafted": random.randint(0, 3),
                    "damage_taken": random.randint(0, 10),
                    "died": random.random() < 0.2
                }
                for _ in range(3)  # 每个配置评估3局
            ]
            
            fitness = agent.evolution.evaluate_candidate(candidate, episode_results)
            print(f"  配置 {candidate.config_id}: 适应度 = {fitness:.2f}")
        
        # 演化
        new_population = agent.evolution.evolve()
        
        summary = agent.evolution.get_evolution_summary()
        print(f"  最佳适应度: {summary['best_fitness']:.2f}")
        print(f"  平均适应度: {summary['avg_fitness']:.2f}")
    
    # 获取最佳配置
    best = agent.evolution.get_best_config()
    if best:
        print(f"\n[最佳配置]")
        print(f"  ID: {best.config_id}")
        print(f"  适应度: {best.fitness_score:.2f}")
        print(f"  阈值: {best.risk_thresholds}")
        print(f"  权重: {best.memory_retrieval_weights}")
    
    # 保存
    agent.save()
    print("\n[演化状态已保存]")


def main():
    parser = argparse.ArgumentParser(description="Luanti/Minecraft 自主智能体")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "luanti", "interactive", "evolve"],
        help="运行模式: demo(演示), luanti(连接Luanti), interactive(交互), evolve(演化)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="local",
        choices=["mock", "openai", "anthropic", "local"],
        help="LLM 提供者（默认 local = Ollama qwen3-coder:30b）"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="数据存储路径"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Luanti 模式 HTTP 服务器端口"
    )
    
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        help="Web 控制面板端口（默认 8080）"
    )

    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "luanti":
        run_luanti(llm_provider=args.llm, port=args.port, data_path=args.data_path,
                   web_port=args.web_port)
    elif args.mode == "interactive":
        run_interactive()
    elif args.mode == "evolve":
        run_evolution()


if __name__ == "__main__":
    main()
