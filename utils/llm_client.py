"""
LLM 调用客户端 - 多后端统一接口

作者: Jiangsheng Yu
许可证: MIT License

支持的 LLM 后端：
  - local    : 本地 Ollama (qwen3-coder:30b) / vLLM / LM Studio ← 默认
  - openai   : OpenAI API (GPT-4 等)
  - anthropic: Anthropic Claude API
  - mock     : 模拟响应（用于测试，无需 GPU）
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    parsed: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""


class BaseLLMProvider(ABC):
    """LLM 提供者基类"""
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000) -> LLMResponse:
        """发送聊天请求"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供者"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", api_base: str = None):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": api_key}
            if api_base:
                client_kwargs["base_url"] = api_base
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return LLMResponse(
                content=content,
                success=True,
                tokens_used=tokens_used,
                model=self.model
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=str(e),
                model=self.model
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) 提供者"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000) -> LLMResponse:
        try:
            # 分离 system 消息
            system_msg = ""
            chat_msgs = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg = msg.get("content", "")
                else:
                    chat_msgs.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_msg,
                messages=chat_msgs
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=content,
                success=True,
                tokens_used=tokens_used,
                model=self.model
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=str(e),
                model=self.model
            )


class LocalProvider(BaseLLMProvider):
    """本地模型提供者（Ollama / vLLM / LM Studio 等 OpenAI 兼容 API）
    
    默认连接 Ollama 的 OpenAI 兼容端点 http://localhost:11434/v1，
    使用 qwen3-coder:30b 作为默认模型。
    
    支持的本地推理后端：
      - Ollama:    http://localhost:11434/v1
      - vLLM:      http://localhost:8000/v1
      - LM Studio: http://localhost:1234/v1
    """
    
    def __init__(self, api_base: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b-instruct"):
        self.api_base = api_base
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="ollama",    # Ollama 不需要真实 API key
                base_url=api_base,
                timeout=30.0,        # 30s 超时：7B 模型应在 10-20s 内响应
                max_retries=0,       # 不重试：避免并发请求淹没 Ollama
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 4096) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = getattr(response.usage, 'total_tokens', 0)
            
            return LLMResponse(
                content=content,
                success=True,
                tokens_used=tokens_used,
                model=self.model
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=str(e),
                model=self.model
            )


class MockProvider(BaseLLMProvider):
    """模拟提供者（用于测试）"""
    
    def __init__(self):
        self.model = "mock"
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000) -> LLMResponse:
        # 返回一个基本的模拟响应
        mock_response = {
            "mode": "gather",
            "goal": "收集基础资源",
            "subgoal": "采集木头",
            "reason": "刚开始游戏，需要收集基础资源制作工具",
            "risk_level": "low",
            "memory_references": [],
            "action_plan": [
                {"action": "explore", "args": {}},
                {"action": "gather_wood", "args": {"count": 5}}
            ],
            "replan_trigger": "inventory_full or danger_detected",
            "reflection_needed": False
        }
        
        return LLMResponse(
            content=json.dumps(mock_response, ensure_ascii=False),
            parsed=mock_response,
            success=True,
            tokens_used=100,
            model="mock"
        )


class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = None,
                 api_key: str = None,
                 api_base: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000,
                 timeout: int = 60):
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # 从环境变量获取 API Key
        if api_key is None:
            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # 初始化提供者
        if provider == "openai":
            self.provider = OpenAIProvider(
                api_key=api_key,
                model=model or "gpt-4",
                api_base=api_base
            )
        elif provider == "anthropic":
            self.provider = AnthropicProvider(
                api_key=api_key,
                model=model or "claude-3-sonnet-20240229"
            )
        elif provider == "local":
            self.provider = LocalProvider(
                api_base=api_base or "http://localhost:11434/v1",
                model=model or "qwen3-coder:30b"
            )
        elif provider == "mock":
            self.provider = MockProvider()
        else:
            raise ValueError(f"不支持的提供者: {provider}")
    
    def chat(self, 
             prompt: str,
             system_prompt: str = None,
             parse_json: bool = True) -> LLMResponse:
        """发送聊天请求"""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.provider.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # 尝试解析 JSON
        if response.success and parse_json:
            response.parsed = self._extract_json(response.content)
        
        return response
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """从响应内容中提取 JSON"""
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取代码块中的 JSON
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def generate_decision(self, 
                          system_prompt: str,
                          context: Dict[str, Any]) -> LLMResponse:
        """生成决策"""
        user_prompt = f"当前输入：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        
        return self.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            parse_json=True
        )
