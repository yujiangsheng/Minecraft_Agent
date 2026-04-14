"""
utils — 通用工具模块

提供 LLM 多后端客户端等基础设施。

模块:
  llm_client  LLMClient — 支持 Ollama / OpenAI / Anthropic / Mock 四种后端
"""

from utils.llm_client import LLMClient, LLMResponse

__all__ = ['LLMClient', 'LLMResponse']
