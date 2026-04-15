"""Generation service wrapper.

This uses LangChain's Ollama integration to call qwen2.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import settings


class Qwen2Generator:
    """LLM generator wrapper for answer synthesis.

    TODO (you):
    - Tune temperature/top_p for your benchmark.
    - Add retry logic and timeout handling.
    - Add optional structured output format enforcement.
    """

    def __init__(self) -> None:
        self.client = ChatOllama(
            model=settings.gen_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

    def generate(self, prompt: str) -> str:
        """Run the generation model and return plain text."""
        result = self.client.invoke(prompt)
        return getattr(result, "content", "").strip()
