"""可扩展的 Prompt 管理：统一上下文 + 注册表 + 多构建器。"""

from __future__ import annotations

from app.utils.prompts.context import PromptContext
from app.utils.prompts.registry import compose_generation_prompt, register_prompt_builder, resolve_prompt_key

__all__ = [
    "PromptContext",
    "compose_generation_prompt",
    "register_prompt_builder",
    "resolve_prompt_key",
]
