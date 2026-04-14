"""
Web Dashboard — 智能体 Web 控制面板

作者: Jiangsheng Yu
许可证: MIT License

提供 Web 界面用于：
  - 一键启动 Luanti 游戏 + 自动等待连接
  - 实时查看智能体状态（生命/饥饿/库存/位置）
  - 查看决策日志流
  - 设定/修改智能体任务目标
  - 查看演化统计 + 手动触发演化
  - 终止智能体服务（退出按钮）

架构：
  Flask 风格的轻量 HTTP 服务器（纯标准库实现）
  GET  /               → 主页面（静态 HTML）
  GET  /api/status     → 智能体实时状态 JSON
  GET  /api/logs       → 最近决策日志
  POST /api/task       → 设定用户自定义任务
  POST /api/launch     → 启动 Luanti 游戏
  POST /api/evolve     → 触发一次演化
  POST /api/save       → 手动保存状态
  POST /api/new_episode→ 手动开始新 Episode
  POST /api/shutdown   → 终止服务
  GET  /api/events     → SSE（Server-Sent Events）实时推送
"""

import json
import os
import queue
import threading
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("WebDashboard")

# 最大日志条目数
MAX_LOG_ENTRIES = 200


class DashboardState:
    """仪表盘共享状态（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._agent_state: Dict[str, Any] = {}
        self._env_state: Dict[str, Any] = {}
        self._memory_stats: Dict[str, Any] = {}
        self._evolution_stats: Dict[str, Any] = {}
        self._logs: List[Dict[str, Any]] = []
        self._user_task: Optional[str] = None
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_steps: int = 0
        self._connected: bool = False
        self._luanti_launched: bool = False
        self._start_time: float = time.time()

        # 训练模式: llm / ppo / dqn / random
        self._training_mode: str = "llm"
        self._training_stats: Dict[str, Any] = {}

        # LLM 提供者: local / openai / anthropic / mock
        self._llm_provider: str = "mock"
        self._llm_model: str = ""
        self._llm_api_base: str = ""

        # SSE 客户端队列
        self._sse_queues: List[queue.Queue] = []
        self._sse_lock = threading.Lock()

    # ── 状态更新 ──

    def update_agent_state(self, state: Dict[str, Any]):
        with self._lock:
            self._agent_state = state

    def update_env_state(self, state: Dict[str, Any]):
        with self._lock:
            self._env_state = state
            self._connected = True

    def update_memory_stats(self, stats: Dict[str, Any]):
        with self._lock:
            self._memory_stats = stats

    def update_evolution_stats(self, stats: Dict[str, Any]):
        with self._lock:
            self._evolution_stats = stats

    def update_episode_info(self, episode_count: int, episode_steps: int):
        with self._lock:
            self._episode_count = episode_count
            self._episode_steps = episode_steps

    def set_luanti_launched(self, launched: bool):
        with self._lock:
            self._luanti_launched = launched

    def update_step_count(self, count: int):
        with self._lock:
            self._step_count = count

    def set_connected(self, connected: bool):
        with self._lock:
            self._connected = connected

    # ── 训练模式 ──

    # ── LLM 提供者 ──

    def set_llm_provider(self, provider: str, model: str = "", api_base: str = ""):
        with self._lock:
            self._llm_provider = provider
            self._llm_model = model
            self._llm_api_base = api_base
        self._push_sse_event("llm_provider", {"provider": provider, "model": model, "api_base": api_base})

    def get_llm_provider(self) -> Dict[str, str]:
        with self._lock:
            return {
                "provider": self._llm_provider,
                "model": self._llm_model,
                "api_base": self._llm_api_base,
            }

    def set_training_mode(self, mode: str):
        with self._lock:
            self._training_mode = mode
            self._training_stats = {}
        self._push_sse_event("training_mode", {"mode": mode})

    def get_training_mode(self) -> str:
        with self._lock:
            return self._training_mode

    def update_training_stats(self, stats: Dict[str, Any]):
        with self._lock:
            self._training_stats.update(stats)

    # ── 日志 ──

    def add_log(self, log_entry: Dict[str, Any]):
        with self._lock:
            self._logs.append(log_entry)
            if len(self._logs) > MAX_LOG_ENTRIES:
                self._logs = self._logs[-MAX_LOG_ENTRIES:]
        self._push_sse_event("log", log_entry)

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._logs[-limit:])

    # ── 用户任务 ──

    def set_user_task(self, task: str):
        with self._lock:
            self._user_task = task if task.strip() else None
        self._push_sse_event("task", {"task": task})
        self.add_log({
            "time": time.strftime("%H:%M:%S"),
            "type": "user_task",
            "message": f"用户设定任务: {task}" if task.strip() else "用户清除任务（恢复自主模式）"
        })

    def get_user_task(self) -> Optional[str]:
        with self._lock:
            return self._user_task

    # ── 完整快照 ──

    def get_full_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "agent_state": dict(self._agent_state),
                "env_state": dict(self._env_state),
                "memory_stats": dict(self._memory_stats),
                "evolution_stats": dict(self._evolution_stats),
                "step_count": self._step_count,
                "episode_count": self._episode_count,
                "episode_steps": self._episode_steps,
                "connected": self._connected,
                "luanti_launched": self._luanti_launched,
                "user_task": self._user_task,
                "uptime": int(time.time() - self._start_time),
                "training_mode": self._training_mode,
                "training_stats": dict(self._training_stats),
                "llm_provider": self._llm_provider,
                "llm_model": self._llm_model,
                "llm_api_base": self._llm_api_base,
            }

    # ── SSE 推送 ──

    def register_sse_client(self) -> queue.Queue:
        q = queue.Queue(maxsize=50)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def unregister_sse_client(self, q: queue.Queue):
        with self._sse_lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    def _push_sse_event(self, event_type: str, data: Any):
        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        with self._sse_lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def push_status_update(self):
        """推送一次完整状态到所有 SSE 客户端"""
        self._push_sse_event("status", self.get_full_status())


# ════════════════════════════════════════════════════════════════

class DashboardHandler(BaseHTTPRequestHandler):
    """处理 Web Dashboard HTTP 请求"""

    dashboard: "WebDashboard" = None  # 类级引用，由 WebDashboard 设置

    def log_message(self, format, *args):
        logger.debug(format % args)

    # ── 路由 ──

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/status":
            self._json_response(self.dashboard.state.get_full_status())
        elif path == "/api/logs":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["50"])[0])
            self._json_response(self.dashboard.state.get_logs(limit))
        elif path == "/api/training_mode":
            self._json_response({
                "mode": self.dashboard.state.get_training_mode(),
                "stats": self.dashboard.state._training_stats,
            })
        elif path == "/api/llm_provider":
            self._json_response(self.dashboard.state.get_llm_provider())
        elif path == "/api/intents":
            self._handle_get_intents()
        elif path == "/api/events":
            self._serve_sse()
        else:
            self._json_response({"error": "not_found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/task":
            self._handle_set_task()
        elif path == "/api/launch":
            self._handle_launch()
        elif path == "/api/evolve":
            self._handle_evolve()
        elif path == "/api/save":
            self._handle_save()
        elif path == "/api/new_episode":
            self._handle_new_episode()
        elif path == "/api/shutdown":
            self._handle_shutdown()
        elif path == "/api/training_mode":
            self._handle_training_mode()
        elif path == "/api/llm_provider":
            self._handle_llm_provider()
        elif path == "/api/intent_decompose":
            self._handle_intent_decompose()
        elif path == "/api/manual_action":
            self._handle_manual_action()
        else:
            self._json_response({"error": "not_found"}, 404)

    # ── 处理方法 ──

    def _handle_set_task(self):
        body = self._read_json_body()
        if body is None:
            self._json_response({"error": "invalid_json"}, 400)
            return
        task = body.get("task", "")
        self.dashboard.state.set_user_task(task)
        self._json_response({"ok": True, "task": task})

    def _handle_launch(self):
        if self.dashboard and self.dashboard.launch_callback:
            threading.Thread(target=self.dashboard.launch_callback, daemon=True).start()
            self._json_response({"ok": True, "message": "正在启动 Luanti..."})
        else:
            self._json_response({"error": "launch_callback not set"}, 500)

    def _handle_evolve(self):
        if self.dashboard and self.dashboard.evolve_callback:
            threading.Thread(target=self.dashboard.evolve_callback, daemon=True).start()
            self._json_response({"ok": True, "message": "正在执行演化..."})
        else:
            self._json_response({"error": "evolve_callback not set"}, 500)

    def _handle_save(self):
        if self.dashboard and self.dashboard.save_callback:
            try:
                self.dashboard.save_callback()
                self._json_response({"ok": True, "message": "已保存"})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "save_callback not set"}, 500)

    def _handle_new_episode(self):
        if self.dashboard and self.dashboard.new_episode_callback:
            threading.Thread(target=self.dashboard.new_episode_callback, daemon=True).start()
            self._json_response({"ok": True, "message": "开始新 Episode"})
        else:
            self._json_response({"error": "new_episode_callback not set"}, 500)

    def _handle_shutdown(self):
        self._json_response({"ok": True, "message": "正在关闭..."})
        if self.dashboard and self.dashboard.shutdown_callback:
            threading.Thread(target=self.dashboard.shutdown_callback, daemon=True).start()

    def _handle_training_mode(self):
        body = self._read_json_body()
        if body is None:
            self._json_response({"error": "invalid_json"}, 400)
            return
        mode = body.get("mode", "").strip()
        if mode not in ("llm", "ppo", "dqn", "random"):
            self._json_response({"error": "invalid mode, must be llm/ppo/dqn/random"}, 400)
            return
        if self.dashboard and self.dashboard.training_mode_callback:
            threading.Thread(
                target=self.dashboard.training_mode_callback,
                args=(mode,), daemon=True
            ).start()
            self._json_response({"ok": True, "mode": mode})
        else:
            self._json_response({"error": "training_mode_callback not set"}, 500)

    def _handle_llm_provider(self):
        body = self._read_json_body()
        if body is None:
            self._json_response({"error": "invalid_json"}, 400)
            return
        provider = body.get("provider", "").strip()
        if provider not in ("local", "openai", "anthropic", "mock"):
            self._json_response({"error": "invalid provider, must be local/openai/anthropic/mock"}, 400)
            return
        model = body.get("model", "").strip()
        api_base = body.get("api_base", "").strip()
        if self.dashboard and self.dashboard.llm_provider_callback:
            try:
                self.dashboard.llm_provider_callback(provider, model, api_base)
                self._json_response({"ok": True, "provider": provider, "model": model, "api_base": api_base})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "llm_provider_callback not set"}, 500)

    def _handle_get_intents(self):
        if self.dashboard and self.dashboard.get_intents_callback:
            try:
                intents = self.dashboard.get_intents_callback()
                self._json_response({"ok": True, "intents": intents})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"ok": True, "intents": []})

    def _handle_intent_decompose(self):
        body = self._read_json_body()
        if body is None:
            self._json_response({"error": "invalid_json"}, 400)
            return
        intent = body.get("intent", "").strip()
        if not intent:
            self._json_response({"error": "intent is required"}, 400)
            return
        if self.dashboard and self.dashboard.intent_decompose_callback:
            try:
                result = self.dashboard.intent_decompose_callback(intent)
                self._json_response({"ok": True, "decomposition": result})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "intent_decompose_callback not set"}, 500)

    def _handle_manual_action(self):
        body = self._read_json_body()
        if body is None:
            self._json_response({"error": "invalid_json"}, 400)
            return
        actions = body.get("actions", [])
        if not actions or not isinstance(actions, list):
            self._json_response({"error": "actions must be a non-empty list"}, 400)
            return
        if self.dashboard and self.dashboard.manual_action_callback:
            count = self.dashboard.manual_action_callback(actions)
            self.dashboard.state.add_log({
                "time": __import__('time').strftime('%H:%M:%S'),
                "type": "action",
                "message": f"手动发送 {count} 个动作: {[a.get('action') for a in actions]}"
            })
            self._json_response({"ok": True, "count": count})
        else:
            self._json_response({"error": "manual_action_callback not set"}, 500)

    def _serve_html(self):
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except FileNotFoundError:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"static/index.html not found")

    def _serve_sse(self):
        """Server-Sent Events 流"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = self.dashboard.state.register_sse_client()
        try:
            # 先推送一次完整状态
            initial = json.dumps({
                "type": "status",
                "data": self.dashboard.state.get_full_status()
            }, ensure_ascii=False)
            self.wfile.write(f"data: {initial}\n\n".encode("utf-8"))
            self.wfile.flush()

            while not self.dashboard._shutdown_flag:
                try:
                    payload = q.get(timeout=2)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # 心跳
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            self.dashboard.state.unregister_sse_client(q)

    # ── 工具方法 ──

    def _json_response(self, data: Any, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _read_json_body(self) -> Optional[Dict]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class ThreadedDashboardServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebDashboard:
    """Web 控制面板"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.state = DashboardState()
        self.shutdown_callback: Optional[Callable] = None
        self.launch_callback: Optional[Callable] = None
        self.evolve_callback: Optional[Callable] = None
        self.save_callback: Optional[Callable] = None
        self.new_episode_callback: Optional[Callable] = None
        self.training_mode_callback: Optional[Callable] = None
        self.manual_action_callback: Optional[Callable] = None
        self.llm_provider_callback: Optional[Callable] = None
        self.get_intents_callback: Optional[Callable] = None
        self.intent_decompose_callback: Optional[Callable] = None
        self._shutdown_flag = False
        self._server: Optional[ThreadedDashboardServer] = None
        self._thread: Optional[threading.Thread] = None
        # 周期推送线程
        self._push_thread: Optional[threading.Thread] = None

    def set_shutdown_callback(self, callback: Callable):
        """设置退出回调（终止整个智能体服务）"""
        self.shutdown_callback = callback

    def set_launch_callback(self, callback: Callable):
        """设置启动 Luanti 回调"""
        self.launch_callback = callback

    def set_evolve_callback(self, callback: Callable):
        """设置演化回调"""
        self.evolve_callback = callback

    def set_save_callback(self, callback: Callable):
        """设置保存回调"""
        self.save_callback = callback

    def set_new_episode_callback(self, callback: Callable):
        """设置新 Episode 回调"""
        self.new_episode_callback = callback

    def set_training_mode_callback(self, callback: Callable):
        """设置训练模式切换回调"""
        self.training_mode_callback = callback

    def set_manual_action_callback(self, callback: Callable):
        """设置手动动作回调"""
        self.manual_action_callback = callback

    def set_llm_provider_callback(self, callback: Callable):
        """设置 LLM 提供者切换回调"""
        self.llm_provider_callback = callback

    def set_get_intents_callback(self, callback: Callable):
        """设置获取已知意图列表回调"""
        self.get_intents_callback = callback

    def set_intent_decompose_callback(self, callback: Callable):
        """设置意图分解回调"""
        self.intent_decompose_callback = callback

    def start(self):
        """启动 Web 服务器"""
        DashboardHandler.dashboard = self
        self._server = ThreadedDashboardServer((self.host, self.port), DashboardHandler)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

        # 周期推送状态更新
        self._push_thread = threading.Thread(target=self._periodic_push, daemon=True)
        self._push_thread.start()

        logger.info(f"Web Dashboard 已启动: http://{self.host}:{self.port}")

    def _serve(self):
        while not self._shutdown_flag:
            self._server.handle_request()

    def _periodic_push(self):
        """每秒推送一次状态到 SSE 客户端"""
        while not self._shutdown_flag:
            time.sleep(1)
            self.state.push_status_update()

    def stop(self):
        self._shutdown_flag = True
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        logger.info("Web Dashboard 已停止")
