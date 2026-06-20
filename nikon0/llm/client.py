"""Model clients used by nikon0 generation nodes."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class ChatModelClient(Protocol):
    """Minimal async chat-completion interface."""

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class BailianOllamaChatClient:
    """Uses the existing project LLM client: 百炼 first, Ollama fallback."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict[str, Any]]) -> str:
        from app.services.llm_clients import chat_text

        return chat_text(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
