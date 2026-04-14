"""
Luanti Gymnasium 环境封装

将 Luanti 生存游戏通过 agent_bridge HTTP 协议封装为标准 Gymnasium 环境，
支持纯 PyTorch PPO/DQN 训练。

架构:
  train_rl.py → LuantiGymEnv (Gymnasium) → HTTP Server ← Luanti (Lua mod)

观测空间: Box(27,)  归一化的游戏状态向量（含体能值）
动作空间: Discrete(36) 覆盖全部 Lua handler 及关键参数变体

体能系统:
  每步动作消耗体能，不同动作成本不同（重型动作如 tunnel/bridge 消耗多）
  体能过低时动作受限，迫使智能体学会规划高收益动作序列
  进食和等待可恢复体能

同步机制:
  step() = 等待 Lua 发状态 → 回复动作 → 等待结果 → 等待新状态 → 回复空
  约 1.2s/step (tick_interval=0.6)

提速: 在 minetest.conf 设置 agent_bridge.tick_interval = 0.3
"""

import json
import math
import logging
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple

from luanti_env import LuantiState

logger = logging.getLogger("LuantiGym")


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LuantiGymEnv(gym.Env):
    """
    Luanti 生存游戏 Gymnasium 封装

    观测 Box(27,):
      [0] health/20  [1] hunger/20  [2] breath/10
      [3] time_of_day  [4] is_night  [5] light/15
      [6] has_shelter  [7] has_weapon
      [8-10] pos (x/1000, y/100, z/1000)
      [11] n_hostile/10  [12] hostile_proximity  [13] n_passive/10
      [14] n_drops/10
      [15-19] blocks: tree, stone, ore, water, dirt
      [20-24] inv: wood, stone, tool, food, ore (/64)
      [25] total_items/200
      [26] stamina (体能, 0~1)

    动作 Discrete(36): 见 ACTIONS

    体能系统:
      初始体能 100, 每步消耗因动作而异 (1~12)
      体能 < 10 时强制 wait, 体能=0 开始扣血
      进食回复 +15, 等待回复 +5, 日间缓慢回复 +1/step
    """

    # ═══════════════════════════════════════════════
    #  每个动作的体能消耗 (索引与 ACTIONS 对应)
    # ═══════════════════════════════════════════════
    STAMINA_COST = [
        # 移动 (0-3)
        3,   # 0  move (跑动)
        4,   # 1  retreat (紧急撤退)
        2,   # 2  jump
        4,   # 3  swim (水中移动消耗大)
        # 采集 (4-8)
        5,   # 4  dig(tree) — 砍树
        6,   # 5  dig(stone) — 采石
        7,   # 6  dig(ore) — 采矿
        4,   # 7  dig(通用)
        5,   # 8  dig_down
        # 搜索/拾取 (9-12)
        2,   # 9  find_resource(tree)
        2,   # 10 find_resource(stone)
        2,   # 11 find_resource(food)
        1,   # 12 pickup_item (轻量)
        # 建造 (13-17)
        4,   # 13 place
        10,  # 14 build_shelter (重型)
        2,   # 15 light_area
        12,  # 16 tunnel (非常消耗)
        10,  # 17 bridge (重型)
        # 合成 (18-19)
        3,   # 18 craft
        4,   # 19 smelt
        # 战斗/生存 (20-22)
        6,   # 20 attack (战斗消耗大)
        -15, # 21 eat (进食恢复体能!)
        1,   # 22 equip
        # 库存 (23-26)
        1,   # 23 drop_item
        2,   # 24 deposit_item
        2,   # 25 take_from_container
        1,   # 26 sort_inventory
        # 感知 (27-28)
        1,   # 27 look_around
        1,   # 28 look_at
        # 交互 (29-30)
        2,   # 29 use_node
        3,   # 30 punch_node
        # 农业 (31-32)
        4,   # 31 farm_plant
        4,   # 32 farm_harvest
        # 其他 (33-35)
        8,   # 33 tower_up (重型)
        1,   # 34 sneak
        -5,  # 35 wait (等待恢复体能)
    ]

    STAMINA_MAX = 100.0
    STAMINA_LOW = 10.0       # 体能过低阈值

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # ═══════════════════════════════════════════════
    #  36 个离散动作
    # ═══════════════════════════════════════════════
    ACTIONS: List[Tuple[str, dict]] = [
        # 移动 (4)
        ("move",           {"speed": 4}),              # 0
        ("retreat",        {"speed": 6}),              # 1
        ("jump",           {}),                        # 2
        ("swim",           {}),                        # 3
        # 采集 (5)
        ("dig",            {"target_type": "tree"}),   # 4
        ("dig",            {"target_type": "stone"}),  # 5
        ("dig",            {"target_type": "ore"}),    # 6
        ("dig",            {}),                        # 7
        ("dig_down",       {}),                        # 8
        # 搜索/拾取 (4)
        ("find_resource",  {"resource": "tree"}),      # 9
        ("find_resource",  {"resource": "stone"}),     # 10
        ("find_resource",  {"resource": "food"}),      # 11
        ("pickup_item",    {}),                        # 12
        # 建造 (5)
        ("place",          {}),                        # 13
        ("build_shelter",  {}),                        # 14
        ("light_area",     {}),                        # 15
        ("tunnel",         {"length": 3}),             # 16
        ("bridge",         {"length": 4}),             # 17
        # 合成 (2)
        ("craft",          {}),                        # 18
        ("smelt",          {}),                        # 19
        # 战斗/生存 (3)
        ("attack",         {}),                        # 20
        ("eat",            {}),                        # 21
        ("equip",          {}),                        # 22
        # 库存 (4)
        ("drop_item",      {}),                        # 23
        ("deposit_item",   {}),                        # 24
        ("take_from_container", {}),                   # 25
        ("sort_inventory", {}),                        # 26
        # 感知 (2)
        ("look_around",    {}),                        # 27
        ("look_at",        {}),                        # 28
        # 交互 (2)
        ("use_node",       {}),                        # 29
        ("punch_node",     {}),                        # 30
        # 农业 (2)
        ("farm_plant",     {}),                        # 31
        ("farm_harvest",   {}),                        # 32
        # 其他 (3)
        ("tower_up",       {"height": 3}),             # 33
        ("sneak",          {"enable": True}),          # 34
        ("wait",           {"duration": 2}),           # 35
    ]

    ACTION_NAMES = [f"{i}:{a[0]}" for i, a in enumerate(ACTIONS)]
    OBS_DIM = 27

    _BLOCK_KW = {
        "tree":  ("tree", "leaves", "wood", "log", "trunk"),
        "stone": ("stone", "cobble", "gravel"),
        "ore":   ("ore", "iron", "coal", "copper", "gold", "diamond", "tin", "mese"),
        "water": ("water", "river"),
        "dirt":  ("dirt", "grass", "sand", "clay"),
    }
    _INV_KW = {
        "wood":  ("wood", "tree", "log", "stick", "plank"),
        "stone": ("stone", "cobble", "gravel", "flint"),
        "tool":  ("pick", "axe", "shovel", "sword", "hoe"),
        "food":  ("apple", "bread", "meat", "berry", "mushroom", "wheat"),
        "ore":   ("ore", "iron", "coal", "copper", "gold", "diamond", "tin",
                  "mese", "lump", "ingot"),
    }

    def __init__(self, host="localhost", port=8765, max_steps=500,
                 render_mode=None):
        super().__init__()
        self.host, self.port = host, port
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self._state_q: Queue = Queue()
        self._resp_q: Queue = Queue()
        self._result_q: Queue = Queue()

        self._prev: Optional[Dict] = None
        self._prev_inv: Dict[str, int] = {}
        self._steps = 0
        self._ep_reward = 0.0
        self._ep_count = 0
        self._positions: deque = deque(maxlen=10)
        self._last_act = -1
        self._consecutive_fails = 0
        self._stamina = self.STAMINA_MAX

        self._server = None
        self._started = False

    # ═══════════════════════════════════════════════
    #  HTTP Server
    # ═══════════════════════════════════════════════
    def _start_server(self):
        if self._started:
            return
        env = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _json(self, d, c=200):
                try:
                    self.send_response(c)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(d).encode())
                except (ConnectionResetError, BrokenPipeError):
                    pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                raw = json.loads(self.rfile.read(n)) if n else None
                if self.path == "/state" and raw:
                    s = LuantiState(raw).to_agent_format()
                    env._state_q.put(s)
                    try:
                        a = env._resp_q.get(timeout=120)
                    except Empty:
                        a = []
                    self._json({"actions": a})
                elif self.path == "/action_result" and raw:
                    env._result_q.put(raw)
                    self._json({"status": "ok"})
                else:
                    self._json({"actions": []})

            def do_GET(self):
                self._json({"status": "ok", "mode": "rl"})

        import socket
        self._server = _ThreadingHTTPServer((self.host, self.port), H)
        try:
            self._server.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        threading.Thread(
            target=self._server.serve_forever, daemon=True).start()
        self._started = True
        logger.info(f"Gym HTTP: http://{self.host}:{self.port}")

    def _stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._started = False

    # ═══════════════════════════════════════════════
    #  Gymnasium API
    # ═══════════════════════════════════════════════
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_server()
        self._drain()

        logger.info("等待 Luanti…")
        s = self._state_q.get(timeout=180)
        self._resp_q.put([])

        self._prev = s
        self._prev_inv = dict(s.get("inventory", {}))
        self._steps = 0
        self._ep_reward = 0.0
        self._positions.clear()
        self._last_act = -1
        self._consecutive_fails = 0
        self._stamina = self.STAMINA_MAX
        self._ep_count += 1

        p = s.get("position", {})
        self._positions.append((p.get("x", 0), p.get("y", 0), p.get("z", 0)))
        logger.info(f"Ep {self._ep_count} | HP={s['health']} HG={s['hunger']}")
        return self._vec(s), {}

    def step(self, action: int):
        # 体能过低时强制等待
        if self._stamina < self.STAMINA_LOW and action != 35:  # 35=wait
            action = 35

        # 消耗/恢复体能
        cost = self.STAMINA_COST[action]
        self._stamina = np.clip(self._stamina - cost, 0.0, self.STAMINA_MAX)

        # 日间缓慢回复 +1
        if self._prev and self._prev.get("time") != "night":
            self._stamina = min(self._stamina + 1.0, self.STAMINA_MAX)

        name, params = self.ACTIONS[action]

        # Tick 1: 等状态 → 回复动作
        try:
            self._state_q.get(timeout=30)
        except Empty:
            return self._obs0(), -1.0, False, True, {}

        self._resp_q.put([{"action": name, "params": dict(params)}])

        # 获取结果
        result = self._get(self._result_q, 15)

        # Tick 2: 等执行后状态
        try:
            post = self._state_q.get(timeout=30)
        except Empty:
            return self._obs0(), -1.0, False, True, {}
        self._resp_q.put([])

        p = post.get("position", {})
        self._positions.append((p.get("x", 0), p.get("y", 0), p.get("z", 0)))

        r = self._reward(self._prev, post, result, action)

        # 体能惩罚：体能越低惩罚越大
        if self._stamina <= 0:
            r -= 1.0   # 体能耗尽，严重惩罚
        elif self._stamina < 20:
            r -= 0.3   # 体能偏低，轻微惩罚

        self._ep_reward += r
        self._steps += 1

        done = post.get("health", 20) <= 0
        trunc = self._steps >= self.max_steps

        if done or trunc:
            logger.info(
                f"Ep {self._ep_count} 结束: "
                f"{'死亡' if done else '截断'} "
                f"steps={self._steps} R={self._ep_reward:.1f}")

        self._prev = post
        self._prev_inv = dict(post.get("inventory", {}))
        self._last_act = action

        return self._vec(post), r, done, trunc, {
            "result": result, "step": self._steps}

    def close(self):
        self._stop()

    # ═══════════════════════════════════════════════
    #  观测向量化 (27维，归一化到 [-1,1])
    # ═══════════════════════════════════════════════
    def _vec(self, s: Dict) -> np.ndarray:
        o = np.zeros(self.OBS_DIM, dtype=np.float32)
        o[0] = s.get("health", 20) / 20.0
        o[1] = s.get("hunger", 20) / 20.0
        o[2] = s.get("breath", 10) / 10.0
        o[3] = s.get("time_of_day", 0.5)
        o[4] = 1.0 if s.get("time") == "night" else 0.0
        o[5] = s.get("light_level", 15) / 15.0
        o[6] = 1.0 if s.get("has_shelter") else 0.0
        o[7] = 1.0 if s.get("has_weapon") else 0.0

        p = s.get("position", {})
        o[8]  = np.clip(p.get("x", 0) / 1000, -1, 1)
        o[9]  = np.clip(p.get("y", 0) / 100, -1, 1)
        o[10] = np.clip(p.get("z", 0) / 1000, -1, 1)

        ents = s.get("nearby_entities", [])
        hostile = [e for e in ents if e.get("hostile")]
        passive = [e for e in ents
                   if not e.get("hostile") and e.get("type") != "dropped_item"]
        drops = [e for e in ents if e.get("type") == "dropped_item"]

        o[11] = min(len(hostile), 10) / 10.0
        if hostile:
            o[12] = 1.0 - min(min(h.get("distance", 16) for h in hostile), 16) / 16.0
        o[13] = min(len(passive), 10) / 10.0
        o[14] = min(len(drops), 10) / 10.0

        blks = " ".join(str(b).lower() for b in s.get("nearby_blocks", []))
        for i, (_, kws) in enumerate(self._BLOCK_KW.items()):
            o[15 + i] = 1.0 if any(k in blks for k in kws) else 0.0

        inv = s.get("inventory", {})
        tot = 0
        for i, (_, kws) in enumerate(self._INV_KW.items()):
            c = sum(v for k, v in inv.items()
                    if any(kw in k.lower() for kw in kws))
            o[20 + i] = min(c, 64) / 64.0
            tot += c
        o[25] = min(tot, 200) / 200.0
        o[26] = self._stamina / self.STAMINA_MAX
        return o

    # ═══════════════════════════════════════════════
    #  奖励函数 (匹配 prompt 奖励定义)
    #
    #  +3  采集新资源种类 (wood/stone/ore)
    #  +1  采集已有种类
    #  +5  合成新工具/物品
    #  +5  建造庇护所
    #  +2  进食恢复
    #  +1  有效移动 (位移>2)
    #  -2  重复失败
    #  -3  原地不动 (stuck)
    #  -5  死亡
    #  -0.05 时间成本
    # ═══════════════════════════════════════════════
    def _reward(self, prev, curr, result, action) -> float:
        if not prev:
            return 0.0

        r = 0.0

        # 死亡
        if curr.get("health", 20) <= 0:
            return -5.0

        # 采集资源
        cur_inv = curr.get("inventory", {})
        res_kw = ("wood", "stone", "cobble", "ore", "iron", "coal",
                  "copper", "gold", "diamond", "tin", "mese", "log",
                  "tree", "plank", "stick")
        for item, cnt in cur_inv.items():
            old = self._prev_inv.get(item, 0)
            if cnt > old and any(kw in item.lower() for kw in res_kw):
                r += 3.0 if item not in self._prev_inv else 1.0

        # 合成新物品
        new = set(cur_inv) - set(self._prev_inv)
        craft_kw = ("pick", "axe", "shovel", "sword", "hoe", "furnace",
                    "chest", "door", "torch", "ladder", "ingot")
        for t in new:
            if any(kw in t.lower() for kw in craft_kw):
                r += 5.0

        # 建造庇护所
        if not prev.get("has_shelter") and curr.get("has_shelter"):
            r += 5.0

        # 进食恢复
        hg = curr.get("hunger", 20) - prev.get("hunger", 20)
        hp = curr.get("health", 20) - prev.get("health", 20)
        if hg > 0 or (hp > 0 and action == 21):
            r += 2.0

        # 有效移动
        if len(self._positions) >= 2:
            p1, p2 = self._positions[-2], self._positions[-1]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            if d > 2.0:
                r += 1.0

        # 动作失败
        if result:
            if result.get("success"):
                self._consecutive_fails = 0
            else:
                self._consecutive_fails += 1
                if self._consecutive_fails >= 2:
                    r -= 2.0

        # 卡住
        if len(self._positions) >= 5:
            p0, pn = self._positions[0], self._positions[-1]
            td = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, pn)))
            if td < 1.5:
                r -= 3.0

        r -= 0.05
        return r

    # 工具
    def _drain(self):
        for q in (self._state_q, self._resp_q, self._result_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

    def _get(self, q, t):
        try:
            return q.get(timeout=t)
        except Empty:
            return None

    def _obs0(self):
        return self._vec(self._prev) if self._prev else \
            np.zeros(self.OBS_DIM, dtype=np.float32)
