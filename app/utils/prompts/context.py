"""Prompt 组装所需的统一上下文。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    """一次生成所用的输入；新增字段时保持向后兼容默认值。"""

    question: str
    need_rag: bool
    domain_hint: str
    context_block: str = ""
    route_reason: str = ""
