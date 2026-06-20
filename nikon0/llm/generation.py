"""LLM answer generation with deterministic fallbacks."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import Evidence
from nikon0.llm.client import ChatModelClient
from nikon0.llm.prompts import build_general_messages, build_product_support_messages


class LlmAnswerGenerator:
    """Generates final user-facing answers from governed context and evidence."""

    def __init__(self, client: ChatModelClient) -> None:
        self.client = client

    async def product_support_answer(
        self,
        *,
        context: AgentContext,
        evidence: list[Evidence],
        answer_hints: list[str],
        fallback_answer: str,
        product_context: dict | None = None,
    ) -> str:
        messages = build_product_support_messages(
            context=context,
            evidence=evidence,
            answer_hints=answer_hints,
            product_context=product_context,
        )
        try:
            answer = (await self.client.complete(messages)).strip()
        except Exception as exc:  # noqa: BLE001
            context.trace.add_event(
                "llm.answer.error",
                "product support answer generation failed",
                node="product_support",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return fallback_answer
        if not answer:
            context.trace.add_event(
                "llm.answer.empty",
                "product support answer generation returned empty text",
                node="product_support",
            )
            return fallback_answer
        context.trace.add_event(
            "llm.answer",
            "generated product support answer",
            node="product_support",
            prompt="PRODUCT_SUPPORT_SYSTEM_PROMPT",
            evidence_count=len(evidence),
        )
        return answer

    async def general_answer(self, *, context: AgentContext, fallback_answer: str) -> str:
        messages = build_general_messages(context=context)
        try:
            answer = (await self.client.complete(messages)).strip()
        except Exception as exc:  # noqa: BLE001
            context.trace.add_event(
                "llm.answer.error",
                "general answer generation failed",
                node="general_handle",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return fallback_answer
        if not answer:
            context.trace.add_event(
                "llm.answer.empty",
                "general answer generation returned empty text",
                node="general_handle",
            )
            return fallback_answer
        context.trace.add_event(
            "llm.answer",
            "generated general answer",
            node="general_handle",
            prompt="GENERAL_SYSTEM_PROMPT",
        )
        return answer
