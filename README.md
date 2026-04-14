# Luanti 自主生存智能体

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LLM: Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.com/)

> **作者**: Jiangsheng Yu · **许可证**: MIT License

一个能够在 **Luanti**（开源 Minecraft 替代）生存模式中自主行动的 AI 智能体系统。  
通过本地 LLM 驱动决策，具备 **记忆 → 反思 → 学习 → 进化** 的完整认知循环，  
同时支持 **强化学习训练**（PPO / DQN）作为替代决策方式。

---

## 特性一览

| 能力 | 说明 |
|------|------|
| **LLM 驱动决策** | 默认使用本地 Ollama（qwen2.5:7b / qwen3-coder:30b），也支持 OpenAI / Anthropic |
| **36 种游戏动作** | 移动、挖掘、放置、合成、攻击、进食、游泳、搜索、建造、农业等全覆盖 |
| **三层记忆系统** | 情景记忆（经验）+ 语义规则（知识）+ 技能库（可复用动作序列）|
| **双层反思** | 局部反思（每步偏差检测）+ 整局复盘（每 50 步总结模式）|
| **技能自动生成** | 分析成功/失败模式，从经验中发明新的可复用技能 |
| **策略演化** | 遗传算法优化风险阈值、记忆权重、技能优先级 |
| **RL 训练** | 纯 PyTorch 实现 PPO / DQN，Gymnasium 标准接口 |
| **Web 控制面板** | 实时状态监控、日志流、任务设定、训练模式切换 |

---

## 项目结构

```
Minecraft_Agent/
├── main.py                # 主程序入口（demo / luanti / interactive / evolve 四种模式）
├── config.py              # 全局配置（风险阈值、LLM、记忆、演化等 dataclass）
├── luanti_env.py          # Luanti 环境连接器（HTTP 桥接 + 动作翻译 + 模糊匹配）
├── web_dashboard.py       # Web 控制面板（SSE 实时推送 + 训练模式管理）
├── gym_env.py             # Gymnasium 环境封装（26维观测 / 36维动作 / 奖励函数）
├── train_rl.py            # RL 训练器（纯 PyTorch PPO + DQN + 随机基线）
├── requirements.txt       # Python 依赖
├── .gitignore             # Git 忽略规则
│
├── agent/                 # 智能体核心模块
│   ├── core.py            #   感知 → 规则预判 → LLM 流水线决策 → 计划执行
│   ├── planner.py         #   优先级识别 → 技能选择 → ≤5步动作序列
│   ├── reflection.py      #   局部反思（QuickReflection）+ 整局复盘（LongReflection）
│   ├── skill_builder.py   #   从失败/成功模式自动发明可复用技能
│   └── evolution.py       #   遗传算法：种群管理 → 适应度评估 → 交叉变异
│
├── memory/                # 三层记忆系统
│   ├── memory_manager.py  #   统一管理器 + 轨迹窗口 + JSON 持久化
│   ├── episodic.py        #   情景记忆（标签搜索、Jaccard 相似度、LRU 驱逐）
│   ├── semantic.py        #   语义规则（8 条预置规则 + 条件检索 + 置信度学习）
│   └── skills.py          #   技能库（6 个预置技能 + 触发匹配 + 使用统计）
│
├── prompts/               # Prompt 模板
│   └── __init__.py        #   6 层 Prompt 流水线 + 动作注册表 + 上下文格式化
│
├── utils/                 # 工具模块
│   └── llm_client.py      #   多后端 LLM 客户端（Ollama / OpenAI / Anthropic / Mock）
│
├── static/                # 前端资源
│   └── index.html         #   Web Dashboard 单页面（状态监控 + 日志流 + 训练控制）
│
├── luanti_mod/            # Luanti Lua 模组
│   └── agent_bridge/
│       ├── mod.conf        #   模组元信息
│       └── init.lua        #   状态收集 + 36 种动作执行器 + HTTP 通信
│
├── demo_data/             # 演示数据（预置记忆和技能，供 demo 模式使用）
├── data/                  # 运行时数据（自动创建，.gitignore 已排除）
├── models/                # RL 模型检查点（.pt 文件）
└── logs/                  # 训练日志
```

---

## 系统架构

```
┌──────────────────────────────┐     HTTP (8765)     ┌─────────────────────────┐
│         Python 智能体         │◄──────────────────►│       Luanti 游戏        │
│                              │                     │                         │
│  ┌── Agent Core ──────────┐  │  POST /state        │  ┌── agent_bridge ──┐   │
│  │ 感知 → 规则预判 → LLM   │  │◄────────────────── │  │ collect_state()  │   │
│  │ 决策 → 计划执行 → 反思  │  │ ──actions(响应)──►│  │ execute_action() │   │
│  └────────────────────────┘  │                     │  └──────────────────┘   │
│                              │  POST /action_result │                         │
│  ┌── Memory ──────────────┐  │◄────────────────── │  36 种动作:              │
│  │ 情景 + 语义 + 技能库    │  │                     │  move, dig, craft,      │
│  └────────────────────────┘  │                     │  build, attack, ...     │
│                              │                     └─────────────────────────┘
│  ┌── LLM 客户端 ─────────┐  │
│  │ Ollama / OpenAI / ...  │  │      Web (8080)
│  └────────────────────────┘  │◄───────────────────► 浏览器控制面板
│                              │      SSE 实时推送
│  ┌── RL 训练（可选）──────┐  │
│  │ PPO / DQN (PyTorch)    │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

---

## 快速开始

### 前置条件

- **Python 3.9+**
- **Ollama**（推荐，默认 LLM 后端）：https://ollama.com/
- **Luanti** 5.x+（游戏运行时需要）：https://www.luanti.org/

### 1. 安装

```bash
git clone <repo-url>
cd Minecraft_Agent
pip install -r requirements.txt

# 安装并启动 Ollama，下载默认模型
ollama pull qwen2.5:7b-instruct
# 或更大的模型（需要 ≥16GB 显存）
# ollama pull qwen3-coder:30b
```

### 2. 演示模式（无需 Luanti / GPU）

```bash
# Mock LLM — 零依赖快速体验
python main.py --mode demo --llm mock

# 使用本地 Ollama（需要 Ollama 运行中）
python main.py --mode demo
```

### 3. 连接 Luanti 游戏

#### 步骤 1：安装桥接模组

```bash
# macOS
cp -r luanti_mod/agent_bridge ~/Library/Application\ Support/minetest/mods/

# Linux
cp -r luanti_mod/agent_bridge ~/.minetest/mods/
```

#### 步骤 2：配置 Luanti

编辑 `minetest.conf`（macOS: `~/Library/Application Support/minetest/minetest.conf`）：

```ini
# 必须：授予模组 HTTP 权限
secure.http_mods = agent_bridge

# 可选配置
agent_bridge.server_url = http://localhost:8765
agent_bridge.tick_interval = 1.0
agent_bridge.player_name = singleplayer
```

在世界配置 `worlds/<世界名>/world.mt` 中添加：

```
load_mod_agent_bridge = true
```

#### 步骤 3：启动

```bash
# 终端 1：启动智能体 + Web 面板
python main.py --mode luanti --port 8765 --web-port 8080

# 终端 2：启动 Luanti
# macOS:
open -a Luanti
# Linux:
luanti
```

打开浏览器访问 http://localhost:8080 查看 Web 控制面板。  
在 Luanti 中进入已启用 `agent_bridge` 的世界，智能体将自动连接。

### 4. 使用不同的 LLM

```bash
# 本地 Ollama（默认）
python main.py --mode luanti

# OpenAI GPT-4
export OPENAI_API_KEY="sk-..."
python main.py --mode luanti --llm openai

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --mode luanti --llm anthropic

# Mock 模式（固定响应，用于调试）
python main.py --mode luanti --llm mock
```

### 5. RL 训练

```bash
# 确保 Luanti 已启动并加载了 agent_bridge 世界

# PPO 训练
python train_rl.py --algo ppo --total-steps 50000 --port 8765

# DQN 训练
python train_rl.py --algo dqn --total-steps 50000 --port 8765

# 随机基线
python train_rl.py --algo random --total-steps 2000

# 加载模型评估
python train_rl.py --eval --resume models/ppo_final.pt
```

也可以在 Web 面板中切换训练模式（LLM / PPO / DQN / Random）。

### 6. 其他模式

```bash
# 交互模式：手动输入状态 JSON，观察决策输出
python main.py --mode interactive

# 演化模式：运行遗传算法优化智能体配置
python main.py --mode evolve
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `demo` | 运行模式：`demo` / `luanti` / `interactive` / `evolve` |
| `--llm` | `local` | LLM 后端：`local`（Ollama）/ `openai` / `anthropic` / `mock` |
| `--port` | `8765` | Luanti HTTP 桥接端口 |
| `--web-port` | `8080` | Web 控制面板端口 |
| `--data-path` | `./data` | 运行时数据存储路径 |

---

## 配置参考

核心配置定义在 `config.py` 中，使用 Python dataclass：

```python
# LLM 配置
LLMConfig(
    provider="local",                       # Ollama 本地推理
    model="qwen2.5:7b-instruct",           # 默认模型
    api_base="http://localhost:11434/v1",   # Ollama API 端点
    temperature=0.5,                        # 生成温度
    max_tokens=1024,                        # 最大输出 token
    timeout=60,                             # 超时秒数
)

# 风险阈值
RiskThresholds(
    health_critical=5,    # HP < 5 → 紧急撤退
    health_low=8,         # HP < 8 → 避免战斗
    hunger_critical=3,    # 饥饿 < 3 → 紧急寻食
    hunger_low=6,         # 饥饿 < 6 → 优先找食物
)

# 记忆系统
MemoryConfig(
    episodic_max_size=100,   # 情景记忆上限
    semantic_max_rules=50,   # 语义规则上限
    skills_max_count=30,     # 技能上限
    trajectory_window=20,    # 轨迹窗口步数
)
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 游戏引擎 | [Luanti](https://www.luanti.org/) (Minetest) |
| 通信协议 | HTTP（Python HTTPServer ↔ Lua mod） |
| LLM 推理 | Ollama (OpenAI 兼容 API) |
| RL 训练 | PyTorch（PPO + DQN，纯手写实现） |
| 环境接口 | Gymnasium |
| Web 面板 | 原生 HTML/CSS/JS + SSE |
| 持久化 | JSON 文件 |

---

## 常见问题

### Luanti 连接不上？
1. 确认 `minetest.conf` 中 `secure.http_mods = agent_bridge`
2. 确认世界配置 `load_mod_agent_bridge = true`
3. 确认 Python 端口（默认 8765）未被占用
4. 如果 Luanti 已在运行，尝试重新进入世界

### LLM 响应很慢？
- 小模型推荐 `qwen2.5:7b-instruct`（CPU 可用，约 3-5s/决策）
- 大模型 `qwen3-coder:30b` 需要 GPU（RTX 3060+ 或 M1 Mac 16GB+）
- 在 `config.py` 中调整 `timeout` 和 `max_tokens`

### SB3 (stable-baselines3) 崩溃？
本项目的 RL 训练已改用纯 PyTorch 实现，无需 SB3。  
已知 SB3 在 Python 3.9 + macOS 上存在 grpcio C++ mutex 崩溃问题。

---

## 许可证

[MIT License](LICENSE) © 2025 Jiangsheng Yu
