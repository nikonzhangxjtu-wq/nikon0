"""Prompt 拼装工具。

核心原则：生成模型应主要依据结构化上下文回答，减少幻觉。
"""

from __future__ import annotations

from app.services.retriever import RetrievedChunk
from app.utils.prompts import PromptContext, compose_generation_prompt


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """将检索结果压成一段便于塞进 prompt 的上下文字符串。

    TODO（你来补）：
    - 按 token 预算做截断
    - 只保留分数高于阈值的 chunk
    """

    if not chunks:
        return "（无检索上下文）"

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[片段 {idx}]")
        lines.append(f"chunk_id: {chunk.chunk_id}")
        lines.append(f"手册: {chunk.manual_name}")
        lines.append(f"分数: {chunk.score}")
        lines.append(f"正文: {chunk.text}")
        if chunk.image_ids:
            lines.append(f"图片ID: {', '.join(chunk.image_ids)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_generation_prompt(
    question: str,
    context_block: str,
    domain_hint: str,
    need_rag: bool = True,
    route_reason: str = "",
) -> str:
    """为 qwen2 构造生成 prompt；实现委托给 `app.utils.prompts` 注册表。"""
    return compose_generation_prompt(
        PromptContext(
            question=question,
            need_rag=need_rag,
            domain_hint=domain_hint,
            context_block=context_block,
            route_reason=route_reason,
        )
    )