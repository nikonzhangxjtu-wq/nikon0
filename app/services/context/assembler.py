"""证据保真型上下文总装配器。"""

from __future__ import annotations

from dataclasses import replace

from app.core.config import settings
from app.services.context.budget import estimate_tokens, token_budget_to_chars, truncate_at_boundary
from app.services.context.evidence_extractor import compress_evidence_blocks
from app.services.context.fact_extractor import extract_critical_facts, render_facts
from app.services.context.types import AssembledPromptContext, CompressionTrace, ContextPacket
from app.services.context.verifier import verify_evidence
from app.utils.prompts import PromptContext


class ContextAssembler:
    """将 PromptContext 压缩成事实 + 证据 + 少量历史的可控信息包。"""

    def assemble(self, ctx: PromptContext) -> AssembledPromptContext:
        original_tokens = self._total_tokens(ctx)
        if not settings.context_assembler_enabled:
            return AssembledPromptContext(
                context=ctx,
                trace=CompressionTrace(original_tokens=original_tokens, final_tokens=original_tokens),
            )

        facts = extract_critical_facts(
            question=ctx.question,
            context_block=ctx.context_block,
            visual_context=ctx.visual_context,
            conversation_history=ctx.conversation_history,
            memory_context=ctx.memory_context,
        )

        evidence_text, evidence_blocks, notes = compress_evidence_blocks(
            context_block=ctx.context_block,
            question=ctx.question,
            facts=facts,
            max_tokens=settings.context_rag_token_budget,
        )
        verifier_reasons = verify_evidence(ctx.question, facts, evidence_blocks)
        if verifier_reasons and ctx.context_block:
            # 关键自检失败时回退完整 RAG 上下文，保证正确性优先于省 token。
            evidence_text = ctx.context_block
            notes.append("verifier_fallback_full_rag")

        facts_text = render_facts(facts)
        context_block = "\n\n".join(part for part in (facts_text, evidence_text) if part.strip())

        history_text = _compress_history(ctx.conversation_history, settings.context_history_token_budget)
        visual_text = _compress_visual(ctx.visual_context, settings.context_visual_token_budget)
        memory_text, memory_notes = _compress_memory(
            ctx.memory_context,
            settings.context_memory_token_budget,
        )
        notes.extend(memory_notes)

        packet = ContextPacket(
            question=ctx.question,
            facts=facts,
            evidence_blocks=evidence_blocks,
            history_text=history_text,
            visual_text=visual_text,
            memory_text=memory_text,
        )

        assembled = replace(
            ctx,
            context_block=context_block,
            conversation_history=packet.history_text,
            visual_context=packet.visual_text,
            memory_context=packet.memory_text,
        )

        final_tokens = self._total_tokens(assembled)
        if final_tokens > settings.context_total_token_budget:
            # 总预算仍超限时，优先收缩历史和视觉；RAG 证据已做结构保护，最后才动。
            history_text = _compress_history(history_text, max(180, settings.context_history_token_budget // 2))
            visual_text = _compress_visual(visual_text, max(120, settings.context_visual_token_budget // 2))
            memory_text, shrink_notes = _compress_memory(
                memory_text,
                max(120, settings.context_memory_token_budget // 2),
            )
            notes.extend(shrink_notes)
            assembled = replace(
                assembled,
                conversation_history=history_text,
                visual_context=visual_text,
                memory_context=memory_text,
            )
            final_tokens = self._total_tokens(assembled)
            notes.append("total_budget:shrink_history_visual_memory")

        return AssembledPromptContext(
            context=assembled,
            trace=CompressionTrace(
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                notes=notes,
                evidence_block_count=len(evidence_blocks),
                critical_fact_count=facts.count(),
                compression_fallback_count=sum(1 for b in evidence_blocks if b.fallback_used)
                + (1 if verifier_reasons else 0),
                verifier_failed_reasons=verifier_reasons,
                preserved_block_types=[b.block_type for b in evidence_blocks],
            ),
        )

    @staticmethod
    def _total_tokens(ctx: PromptContext) -> int:
        return sum(
            estimate_tokens(part)
            for part in (
                ctx.question,
                ctx.context_block,
                ctx.route_reason,
                ctx.visual_context,
                ctx.conversation_history,
                ctx.memory_context,
            )
        )


def _compress_history(text: str, max_tokens: int) -> str:
    if not text or estimate_tokens(text) <= max_tokens:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 历史采用“两层记忆”的 V1：保留最近原文，旧历史只保留较短行。
    kept = lines[-8:]
    older = [ln for ln in lines[:-8] if any(k in ln for k in ("型号", "订单", "故障", "已尝试", "诉求"))]
    out = "\n".join([*older[-6:], *kept])
    return truncate_at_boundary(out, token_budget_to_chars(max_tokens))


def _compress_visual(text: str, max_tokens: int) -> str:
    if not text or estimate_tokens(text) <= max_tokens:
        return text
    priority: list[str] = []
    rest: list[str] = []
    for line in text.splitlines():
        if line.startswith(("OCR文字", "关键实体", "产品类型")):
            priority.append(line.strip())
        elif line.strip():
            rest.append(line.strip())
    out = "\n".join([*priority, *rest])
    return truncate_at_boundary(out, token_budget_to_chars(max_tokens))


def _compress_memory(text: str, max_tokens: int) -> tuple[str, list[str]]:
    if not text:
        return "", []
    if estimate_tokens(text) <= max_tokens:
        return text, []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    priority_prefixes = (
        "[记忆]",
        "[会话事实]",
        "[长期用户画像]",
        "订单号",
        "联系电话",
        "产品/型号",
        "故障码/状态码",
        "用户诉求",
        "用户产品",
        "相关订单",
        "历史问题",
        "已尝试操作",
    )
    priority = [ln for ln in lines if ln.startswith(priority_prefixes)]
    rest = [ln for ln in lines if ln not in priority]
    out = "\n".join([*priority, *rest[:8]])
    return truncate_at_boundary(out, token_budget_to_chars(max_tokens)), ["memory:compressed"]
