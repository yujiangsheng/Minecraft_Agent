"""
Luanti 环境连接器 - Python <-> Luanti HTTP 桥接层

作者: Jiangsheng Yu
许可证: MIT License

架构：
  Python (HTTP Server) <--> Luanti (Lua Mod: agent_bridge)
  
  数据流：
  1. Lua mod 每隔 tick_interval 秒 POST 游戏状态到 /state
  2. Python 处理状态，运行智能体决策，在 HTTP 响应中返回动作命令
  3. Lua mod 接收动作并在游戏中执行
  4. Lua mod POST 执行结果到 /action_result

模块组成：
  - LuantiState:         游戏状态解析与格式化
  - ActionTranslator:    智能体动作 → Luanti 命令翻译
  - AgentBridgeHandler:  HTTP 请求处理器
  - LuantiEnvironment:   环境管理器（服务器生命周期、回调管理）
"""

import json
import threading
import time
from difflib import get_close_matches
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional, Callable, List
from queue import Queue
import logging

logger = logging.getLogger("LuantiEnv")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器，防止 LLM 调用阻塞后续请求"""
    daemon_threads = True
    allow_reuse_address = True
    timeout = 1  # handle_request 超时，允许 shutdown 标志检查

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        super().server_bind()


class LuantiState:
    """Luanti 游戏状态，格式化为智能体期望的格式"""

    def __init__(self, raw_state: Dict[str, Any]):
        self.raw = raw_state

    def to_agent_format(self) -> Dict[str, Any]:
        """将 Luanti 原始状态转换为智能体期望的格式"""
        raw = self.raw

        # 提取位置
        pos = raw.get("position") or {}

        # 提取库存 - 简化键名
        inventory = {}
        raw_inv = raw.get("inventory")
        if isinstance(raw_inv, dict):
            for item_name, count in raw_inv.items():
                # 去掉模组前缀 (default:wood -> wood)
                short_name = item_name.split(":")[-1] if ":" in item_name else item_name
                inventory[short_name] = count
        elif isinstance(raw_inv, list):
            # Lua 空表可能序列化为 []
            pass

        # 检查是否有武器
        weapon_names = ["sword", "axe"]
        has_weapon = any(
            any(w in item for w in weapon_names)
            for item in inventory.keys()
        )

        # 提取附近方块名
        nearby_blocks = []
        for block_info in (raw.get("nearby_blocks") or []):
            if isinstance(block_info, dict):
                name = block_info.get("name", "")
            elif isinstance(block_info, str):
                name = block_info
            else:
                continue
            short_name = name.split(":")[-1] if ":" in name else name
            if short_name:
                nearby_blocks.append(short_name)

        # 提取附近实体
        nearby_entities = []
        for entity in (raw.get("nearby_entities") or []):
            if not isinstance(entity, dict):
                continue
            ent_type = entity.get("type", "unknown")
            short_type = ent_type.split(":")[-1] if ":" in ent_type else ent_type
            nearby_entities.append({
                "type": short_type,
                "hostile": entity.get("hostile", False),
                "distance": entity.get("distance", 0),
                "health": entity.get("health", 0),
            })

        return {
            "time": raw.get("time") or "day",
            "time_of_day": raw.get("time_of_day") or 0.5,
            "day_count": raw.get("day_count") or 0,
            "health": raw.get("health") or 20,
            "max_health": raw.get("max_health") or 20,
            "hunger": raw.get("hunger") or 20,
            "breath": raw.get("breath") or 10,
            "position": pos,
            "inventory": inventory,
            "nearby_entities": nearby_entities,
            "nearby_blocks": nearby_blocks,
            "has_shelter": bool(raw.get("has_shelter")),
            "light_level": raw.get("light_level") or 15,
            "biome_name": raw.get("biome_name") or "unknown",
            "has_weapon": has_weapon,
        }


class ActionTranslator:
    """将智能体动作转换为 Luanti 命令格式"""

    # 智能体动作 -> Luanti 动作映射
    # 键：智能体 prompt 中定义的动作名称
    # 值：Luanti Lua mod 可执行的 {action, params} 命令
    ACTION_MAP = {
        # === 基础移动 ===
        "explore":        {"action": "move", "params": {"speed": 4}},
        "move_to":        {"action": "move", "params": {}},
        "jump":           {"action": "jump", "params": {}},
        "retreat":        {"action": "retreat", "params": {"speed": 6}},
        "flee_from":      {"action": "retreat", "params": {"speed": 8}},
        "swim":           {"action": "swim", "params": {}},
        "sneak":          {"action": "sneak", "params": {"enable": True}},
        "sprint":         {"action": "sprint", "params": {"enable": True}},

        # === 视角控制 ===
        "look_around":    {"action": "look_around", "params": {}},
        "look_at":        {"action": "look_at", "params": {}},

        # === 采集/挖掘 ===
        "gather_wood":    {"action": "dig", "params": {"target_type": "tree"}},
        "mine_stone":     {"action": "dig", "params": {"target_type": "stone"}},
        "mine_ore":       {"action": "dig", "params": {"target_type": "ore"}},
        "dig":            {"action": "dig", "params": {}},
        "dig_down":       {"action": "dig_down", "params": {}},
        "dig_up":         {"action": "dig_up", "params": {}},
        "tunnel":         {"action": "tunnel", "params": {"length": 3}},

        # === 放置/建造 ===
        "place_block":    {"action": "place", "params": {}},
        "place_at":       {"action": "place_at", "params": {}},
        "build_shelter":  {"action": "build_shelter", "params": {"material": "default:cobble"}},
        "bridge":         {"action": "bridge", "params": {"length": 4}},
        "tower_up":       {"action": "tower_up", "params": {"height": 3}},
        "light_area":     {"action": "light_area", "params": {}},

        # === 物品管理 ===
        "equip":          {"action": "equip", "params": {}},
        "drop_item":      {"action": "drop_item", "params": {}},
        "set_hotbar":     {"action": "set_hotbar", "params": {}},
        "sort_inventory": {"action": "sort_inventory", "params": {}},

        # === 合成/冶炼 ===
        "craft_tool":     {"action": "craft", "params": {}},
        "craft_item":     {"action": "craft", "params": {}},
        "smelt":          {"action": "smelt", "params": {}},
        "check_recipe":   {"action": "check_recipe", "params": {}},

        # === 进食/战斗 ===
        "eat_food":       {"action": "eat", "params": {}},
        "attack":         {"action": "attack", "params": {}},
        "attack_entity":  {"action": "attack", "params": {}},
        "punch_node":     {"action": "punch_node", "params": {}},

        # === 容器交互 ===
        "deposit_item":   {"action": "deposit_item", "params": {}},
        "take_from_container": {"action": "take_from_container", "params": {}},
        "use_node":       {"action": "use_node", "params": {}},

        # === 搜索/感知 ===
        "find_resource":  {"action": "find_resource", "params": {}},
        "pickup_item":    {"action": "pickup_item", "params": {}},

        # === 农业 ===
        "farm_plant":     {"action": "farm_plant", "params": {}},
        "farm_harvest":   {"action": "farm_harvest", "params": {}},

        # === 等待 ===
        "wait":           {"action": "wait", "params": {"duration": 2}},

        # === 兼容映射：常见的 LLM 生成的动作别名 ===
        "find_tree":        {"action": "find_resource", "params": {"resource": "tree"}},
        "break_log":        {"action": "dig", "params": {"target_type": "tree"}},
        "chop_tree":        {"action": "dig", "params": {"target_type": "tree"}},
        "mine":             {"action": "dig", "params": {"target_type": "stone"}},
        "move":             {"action": "move", "params": {}},
        "craft":            {"action": "craft", "params": {}},
        "eat":              {"action": "eat", "params": {}},
        "collect_resource": {"action": "dig", "params": {}},
        "flee":             {"action": "retreat", "params": {"speed": 6}},
        "run_away":         {"action": "retreat", "params": {"speed": 6}},
        "assess_situation": {"action": "look_around", "params": {}},
        "find_safe_spot":   {"action": "move", "params": {"target_type": "shelter"}},
        "gather_food":      {"action": "find_resource", "params": {"resource": "food"}},
        "hunt_food":        {"action": "attack", "params": {"target_type": "animal"}},
        "collect_wood":     {"action": "dig", "params": {"target_type": "tree"}},
        "collect_stone":    {"action": "dig", "params": {"target_type": "stone"}},
        "place":            {"action": "place", "params": {}},
        "build":            {"action": "build_shelter", "params": {}},
        "hide":             {"action": "build_shelter", "params": {"material": "default:dirt"}},
        "rest":             {"action": "wait", "params": {"duration": 3}},
        "sleep":            {"action": "wait", "params": {"duration": 5}},
        "crouch":           {"action": "sneak", "params": {"enable": True}},
        "stop_sneak":       {"action": "sneak", "params": {"enable": False}},
        "stop_sprint":      {"action": "sprint", "params": {"enable": False}},
        "plant":            {"action": "farm_plant", "params": {}},
        "harvest":          {"action": "farm_harvest", "params": {}},
        "cook":             {"action": "smelt", "params": {}},
        "furnace":          {"action": "smelt", "params": {}},
        "drop":             {"action": "drop_item", "params": {}},
        "discard":          {"action": "drop_item", "params": {}},
        "open_chest":       {"action": "use_node", "params": {}},
        "interact":         {"action": "use_node", "params": {}},
        "use":              {"action": "use_node", "params": {}},
        "pick_up":          {"action": "pickup_item", "params": {}},
        "collect":          {"action": "pickup_item", "params": {}},
        "organize_inventory": {"action": "sort_inventory", "params": {}},
        "dig_forward":      {"action": "tunnel", "params": {"length": 3}},
        "pillar_up":        {"action": "tower_up", "params": {"height": 3}},
        "scaffold":         {"action": "tower_up", "params": {"height": 5}},
    }

    # 所有 Lua handler 支持的原生动作名（用于二次兜底）
    LUA_HANDLERS = {
        "move", "dig", "place", "craft", "attack", "eat", "equip",
        "look_around", "build_shelter", "retreat", "jump", "wait",
        "swim", "find_resource", "deposit_item", "light_area",
        "sneak", "sprint", "look_at", "drop_item", "take_from_container",
        "use_node", "place_at", "dig_down", "dig_up", "tunnel",
        "bridge", "tower_up", "farm_plant", "farm_harvest", "smelt",
        "pickup_item", "check_recipe", "punch_node", "sort_inventory",
        "set_hotbar",
    }

    # 语义关键词 → 动作速查表（比 difflib 更精准）
    _KEYWORD_MAP = {
        "tree": {"action": "dig", "params": {"target_type": "tree"}},
        "wood": {"action": "dig", "params": {"target_type": "tree"}},
        "log":  {"action": "dig", "params": {"target_type": "tree"}},
        "stone": {"action": "dig", "params": {"target_type": "stone"}},
        "ore":  {"action": "dig", "params": {"target_type": "ore"}},
        "iron": {"action": "dig", "params": {"target_type": "ore"}},
        "coal": {"action": "dig", "params": {"target_type": "ore"}},
        "food": {"action": "eat", "params": {}},
        "eat":  {"action": "eat", "params": {}},
        "run":  {"action": "retreat", "params": {"speed": 6}},
        "escape": {"action": "retreat", "params": {"speed": 6}},
        "shelter": {"action": "build_shelter", "params": {}},
        "house": {"action": "build_shelter", "params": {}},
        "torch": {"action": "light_area", "params": {}},
        "light": {"action": "light_area", "params": {}},
        "fight": {"action": "attack", "params": {}},
        "kill":  {"action": "attack", "params": {}},
        "chop":  {"action": "dig", "params": {"target_type": "tree"}},
        "mine":  {"action": "dig", "params": {"target_type": "stone"}},
        "craft": {"action": "craft", "params": {}},
        "build": {"action": "build_shelter", "params": {}},
        "explore": {"action": "move", "params": {"speed": 4}},
        "walk":  {"action": "move", "params": {"speed": 3}},
        "gather": {"action": "dig", "params": {}},
        "collect": {"action": "pickup_item", "params": {}},
        "pick":  {"action": "pickup_item", "params": {}},
        "place": {"action": "place", "params": {}},
        "plant": {"action": "farm_plant", "params": {}},
        "harvest": {"action": "farm_harvest", "params": {}},
        "smelt": {"action": "smelt", "params": {}},
        "cook":  {"action": "smelt", "params": {}},
        "search": {"action": "find_resource", "params": {}},
        "find":  {"action": "find_resource", "params": {}},
        "look":  {"action": "look_around", "params": {}},
        "dig":   {"action": "dig", "params": {}},
    }

    @classmethod
    def _fuzzy_resolve(cls, action_name: str, params: dict) -> Dict[str, Any]:
        """对未知动作名进行语义模糊匹配，返回最佳映射"""
        name_lower = action_name.lower().replace("-", "_").replace(" ", "_")

        # 1. 是否直接是 Lua handler 名？
        if name_lower in cls.LUA_HANDLERS:
            return {"action": name_lower, "params": params}

        # 2. 关键词匹配（在动作名中搜索语义关键词）
        for kw, mapped in cls._KEYWORD_MAP.items():
            if kw in name_lower:
                result = mapped.copy()
                merged = mapped.get("params", {}).copy()
                merged.update(params)
                result["params"] = merged
                logger.info(f"  模糊匹配: '{action_name}' → {result['action']}（关键词: {kw}）")
                return result

        # 3. difflib 模糊匹配 ACTION_MAP 键名
        all_names = list(cls.ACTION_MAP.keys())
        matches = get_close_matches(name_lower, all_names, n=1, cutoff=0.5)
        if matches:
            best = matches[0]
            result = cls.ACTION_MAP[best].copy()
            merged = cls.ACTION_MAP[best].get("params", {}).copy()
            merged.update(params)
            result["params"] = merged
            logger.info(f"  模糊匹配: '{action_name}' → {result['action']}（相似: {best}）")
            return result

        # 4. 兜底：explore（移动总比卡住好）
        logger.warning(f"  未知动作 '{action_name}'，无法匹配，降级为 explore")
        return {"action": "move", "params": {"speed": 4}}

    @classmethod
    def translate(cls, agent_actions: List[Dict[str, Any]],
                  env_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将智能体动作计划翻译为 Luanti 可执行命令"""
        luanti_actions = []

        for action in agent_actions:
            action_name = action.get("action", "")
            params = action.get("params", action.get("args", {}))
            if params is None:
                params = {}

            # 精确查找映射
            if action_name in cls.ACTION_MAP:
                mapped = cls.ACTION_MAP[action_name].copy()
                mapped_params = cls.ACTION_MAP[action_name].get("params", {}).copy()
                mapped_params.update(params)
                mapped["params"] = mapped_params
                luanti_actions.append(mapped)
            else:
                # 模糊语义匹配：关键词 → difflib → 兜底 explore
                resolved = cls._fuzzy_resolve(action_name, params)
                luanti_actions.append(resolved)

        return luanti_actions


class AgentBridgeHandler(BaseHTTPRequestHandler):
    """处理来自 Luanti mod 的 HTTP 请求"""

    # 类级别引用，由 LuantiEnvironment 设置
    env = None

    def log_message(self, format, *args):
        """重定向日志到 logger"""
        logger.debug(format % args)

    def _send_json(self, data: Any, code: int = 200):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("客户端已断开连接（超时），跳过响应")

    def _read_body(self) -> Optional[Dict]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def do_POST(self):
        if self.path == "/state":
            self._handle_state()
        elif self.path == "/action_result":
            self._handle_action_result()
        elif self.path == "/shutdown":
            self._send_json({"status": "shutting_down"})
            if self.env:
                self.env._shutdown_flag = True
        else:
            self._send_json({"error": "unknown_endpoint"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/status":
            if self.env:
                self._send_json({
                    "connected": self.env.connected,
                    "steps": self.env.step_count,
                    "last_state_time": self.env.last_state_time,
                })
            else:
                self._send_json({"status": "no_env"})
        else:
            self._send_json({"error": "unknown_endpoint"}, 404)

    def _handle_state(self):
        """处理游戏状态并返回动作"""
        try:
            raw_state = self._read_body()
            if not raw_state:
                self._send_json({"error": "empty_body"}, 400)
                return

            if self.env:
                # 异步决策：如果 LLM 正在思考，立即返回缓存动作
                actions = self.env._on_state_received(raw_state)
                self._send_json({"actions": actions})
            else:
                self._send_json({"actions": []})

        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, 400)
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("客户端超时断开，决策仍在继续")
        except Exception as e:
            logger.error(f"处理状态时出错: {e}")
            try:
                self._send_json({"error": str(e)}, 500)
            except (ConnectionResetError, BrokenPipeError):
                pass

    def _handle_action_result(self):
        """处理动作执行结果"""
        try:
            result = self._read_body()
            if result and self.env:
                self.env._on_action_result(result)
            self._send_json({"status": "ok"})
        except Exception as e:
            logger.error(f"处理动作结果时出错: {e}")
            self._send_json({"error": str(e)}, 500)


class LuantiEnvironment:
    """
    Luanti 游戏环境连接器
    
    运行一个 HTTP 服务器，接收来自 Luanti agent_bridge mod 的游戏状态，
    调用智能体决策，并返回动作命令。
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

        # 状态
        self.connected = False
        self.step_count = 0
        self.last_state_time = 0
        self.current_state: Optional[Dict[str, Any]] = None
        self.last_raw_state: Optional[Dict[str, Any]] = None

        # 回调
        self._decision_callback: Optional[Callable] = None
        self._result_callback: Optional[Callable] = None

        # 动作结果队列
        self.action_results: Queue = Queue()

        # 异步决策：缓存上次动作，防止 LLM 阻塞
        self._cached_actions: List[Dict[str, Any]] = []
        self._decision_lock = threading.Lock()
        self._deciding = False
        self._last_decide_time: float = 0
        self._fallback_counter: int = 0

        # 控制
        self._shutdown_flag = False

    def set_decision_callback(self, callback: Callable[[Dict[str, Any]], List[Dict[str, Any]]]):
        """
        设置决策回调函数
        
        callback 接收 agent_format 的环境状态，应返回 Luanti 动作列表
        """
        self._decision_callback = callback

    def set_result_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """设置动作结果回调"""
        self._result_callback = callback

    def _on_state_received(self, raw_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理收到的游戏状态，返回动作列表

        异步决策模式 + 反应式后备动作：
        - 如果 LLM 正在思考：根据当前状态返回后备生存动作（而非空列表）
        - 如果 LLM 已完成：返回缓存的 LLM 决策，并启动下一轮决策
        """
        self.connected = True
        self.last_state_time = time.time()
        self.last_raw_state = raw_state
        self.step_count += 1

        # 转换为智能体格式
        luanti_state = LuantiState(raw_state)
        self.current_state = luanti_state.to_agent_format()

        logger.info(
            f"[Step {self.step_count}] 状态: 时间={self.current_state['time']}, "
            f"HP={self.current_state['health']}, "
            f"饥饿={self.current_state['hunger']}, "
            f"位置=({self.current_state['position'].get('x', 0):.1f}, "
            f"{self.current_state['position'].get('y', 0):.1f}, "
            f"{self.current_state['position'].get('z', 0):.1f})"
        )

        # 如果 LLM 正在思考，返回反应式后备动作
        if self._deciding:
            elapsed = time.time() - self._last_decide_time
            # 超时保护：超过 45 秒强制重置（但不立即启动新决策，等下个 tick）
            if elapsed > 45:
                logger.warning(f"  LLM 决策超时 ({elapsed:.0f}s)，强制重置")
                self._deciding = False
                self._cached_actions = []
                # 不在此处启动新决策，返回后备动作，下个 tick 再启动
                return self._get_fallback_actions(self.current_state)
            else:
                fallback = self._get_fallback_actions(self.current_state)
                if fallback and self.step_count % 5 == 0:  # 减少日志刷屏
                    logger.info(f"  LLM 思考中({elapsed:.0f}s)，后备动作: {[a['action'] for a in fallback]}")
                return fallback

        # 在后台启动 LLM 决策（严格单线程：只有 _deciding=False 时才启动）
        if self._decision_callback:
            self._deciding = True
            self._last_decide_time = time.time()
            state_snapshot = dict(self.current_state)
            t = threading.Thread(target=self._async_decide, args=(state_snapshot,), daemon=True)
            t.start()

        # 返回缓存的动作（首次为空 → 返回后备）
        actions = self._cached_actions
        self._cached_actions = []
        if actions:
            logger.info(f"  -> 返回 {len(actions)} 个 LLM 决策动作")
            self._fallback_counter = 0
            return actions

        # 没有缓存动作时也返回后备
        return self._get_fallback_actions(self.current_state)

    def _get_fallback_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据当前游戏状态生成反应式后备动作（无需 LLM）

        优先级：
        1. 紧急逃跑（低 HP + 附近有敌对实体）
        2. 进食（饥饿值低且有食物）
        3. 主动采集附近资源
        4. 积极探索新区域
        """
        self._fallback_counter += 1
        hp = state.get("health", 20)
        hunger = state.get("hunger", 20)
        entities = state.get("nearby_entities", [])
        inventory = state.get("inventory", {})
        nearby_blocks = state.get("nearby_blocks", [])

        # 紧急：低 HP + 附近有敌对实体 → 逃跑
        hostile_nearby = any(
            e.get("hostile") and e.get("distance", 999) < 8
            for e in entities
        )
        if hp < 8 and hostile_nearby:
            return [{"action": "retreat", "params": {"speed": 6}}]

        # 饥饿：尝试进食
        if hunger < 10:
            food_keys = [k for k in inventory if any(
                f in k for f in ("apple", "bread", "meat", "berry", "food")
            )]
            if food_keys:
                return [{"action": "eat", "params": {"item": food_keys[0]}}]

        # 检查附近有无树、石头等可采集资源
        has_tree = any("tree" in str(b).lower() or "wood" in str(b).lower() for b in nearby_blocks)
        has_stone = any("stone" in str(b).lower() or "cobble" in str(b).lower() for b in nearby_blocks)

        # 检查附近有无掉落物
        has_drops = any(
            e.get("type") == "dropped_item" and e.get("distance", 999) < 6
            for e in entities
        )

        # 优先捡掉落物
        if has_drops:
            return [{"action": "pickup_item", "params": {}}]

        # 主动采集：使用 Lua 原生动作名（dig/move/find_resource 等）
        cycle = self._fallback_counter % 8
        if has_tree and cycle in (0, 1, 2):
            return [{"action": "dig", "params": {"target_type": "tree"}},
                    {"action": "pickup_item", "params": {}}]
        elif has_stone and cycle in (3, 4):
            return [{"action": "dig", "params": {"target_type": "stone"}},
                    {"action": "pickup_item", "params": {}}]
        elif cycle == 5:
            return [{"action": "find_resource", "params": {"resource": "tree"}}]
        elif cycle == 6:
            return [{"action": "move", "params": {"speed": 4}}]
        else:
            # 快速移动到新区域
            return [{"action": "move", "params": {"speed": 5}},
                    {"action": "look_around", "params": {}}]

    def _on_action_result(self, result: Dict[str, Any]):
        """处理动作执行结果"""
        self.action_results.put(result)

        success = result.get("success", False)
        action = result.get("action", "unknown")
        outcome = result.get("outcome", "")
        logger.info(f"  <- 动作结果: {action} = {'成功' if success else '失败'} ({outcome})")

        if self._result_callback:
            try:
                self._result_callback(result)
            except Exception as e:
                logger.error(f"结果回调出错: {e}")

    def _async_decide(self, state_snapshot: Dict[str, Any]):
        """后台线程：调用 LLM 生成决策，完成后缓存动作"""
        start = time.time()
        try:
            actions = self._decision_callback(state_snapshot)
            self._cached_actions = actions or []
            elapsed = time.time() - start
            logger.info(f"  LLM 决策完成 ({elapsed:.1f}s)，生成 {len(self._cached_actions)} 个动作")
        except Exception as e:
            logger.error(f"后台决策出错 ({time.time()-start:.1f}s): {e}")
            self._cached_actions = []
        finally:
            self._deciding = False

    def start(self):
        """启动 HTTP 服务器"""
        AgentBridgeHandler.env = self

        self.server = ThreadingHTTPServer((self.host, self.port), AgentBridgeHandler)
        self.server_thread = threading.Thread(target=self._serve, daemon=True)
        self.server_thread.start()

        logger.info(f"Luanti 环境服务器已启动: http://{self.host}:{self.port}")
        logger.info("等待 Luanti agent_bridge mod 连接...")

    def _serve(self):
        """服务器主循环"""
        while not self._shutdown_flag:
            self.server.handle_request()

    def stop(self):
        """停止服务器"""
        self._shutdown_flag = True
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        logger.info("Luanti 环境服务器已停止")

    def wait_for_connection(self, timeout: float = 60) -> bool:
        """等待 Luanti mod 连接"""
        start = time.time()
        while time.time() - start < timeout:
            if self.connected:
                logger.info("Luanti mod 已连接！")
                return True
            time.sleep(0.5)
        logger.warning(f"等待连接超时 ({timeout}s)")
        return False

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """获取最新的游戏状态（agent 格式）"""
        return self.current_state

    def get_latest_action_result(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """获取最新的动作结果"""
        try:
            return self.action_results.get(timeout=timeout)
        except Exception:
            return None

    def inject_manual_actions(self, actions: List[Dict[str, Any]]) -> int:
        """注入手动动作，下一个 tick 优先执行（覆盖缓存的 LLM 动作）"""
        self._cached_actions = list(actions)
        logger.info(f"注入 {len(actions)} 个手动动作: {[a.get('action') for a in actions]}")
        return len(actions)

    def get_stats(self) -> Dict[str, Any]:
        """获取环境统计"""
        return {
            "connected": self.connected,
            "step_count": self.step_count,
            "last_state_time": self.last_state_time,
            "server_url": f"http://{self.host}:{self.port}",
        }
