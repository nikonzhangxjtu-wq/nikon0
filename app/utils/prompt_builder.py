"""Prompt building utilities.

The core idea: generator model should only answer from structured context.
"""

from __future__ import annotations

from app.services.retriever import RetrievedChunk


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Convert retrieval results into a compact context string.

    TODO (you):
    - Add truncation rules to control token budget.
    - Include only chunks above a score threshold.
    """

    if not chunks:
        return "No retrieved context."

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[CHUNK {idx}]")
        lines.append(f"chunk_id: {chunk.chunk_id}")
        lines.append(f"manual: {chunk.manual_name}")
        lines.append(f"score: {chunk.score}")
        lines.append(f"text: {chunk.text}")
        if chunk.image_ids:
            lines.append(f"image_ids: {', '.join(chunk.image_ids)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_generation_prompt(question: str, context_block: str, domain_hint: str) -> str:
    """Create a strict generation prompt for qwen2."""

    return f"""
You are a customer-service assistant.

Rules:
1) Answer the user's question directly.
2) Prioritize information from CONTEXT.
3) If context is insufficient, explicitly say what is unknown.
4) Keep response clear and structured.
5) Domain hint: {domain_hint}

CONTEXT:
{context_block}

USER QUESTION:
{question}
""".strip()
