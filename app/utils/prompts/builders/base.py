"""Prompt 构建器协议。"""

from __future__ import annotations

from typing import Protocol

from app.utils.prompts.context import PromptContext


class PromptBuilder(Protocol):
    def build(self, ctx: PromptContext) -> str: ...
