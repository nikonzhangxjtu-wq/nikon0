"""上下文压缩的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.utils.prompts import PromptContext


@dataclass(frozen=True)
class CriticalFacts:
    """不能被普通文本压缩丢掉的 P0 事实。"""

    order_ids: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    product_models: list[str] = field(default_factory=list)
    fault_codes: list[str] = field(default_factory=list)
    user_goals: list[str] = field(default_factory=list)
    visual_entities: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)

    def count(self) -> int:
        return sum(
            len(v)
            for v in (
                self.order_ids,
                self.phones,
                self.product_models,
                self.fault_codes,
                self.user_goals,
                self.visual_entities,
                self.missing_slots,
            )
        )


@dataclass(frozen=True)
class EvidenceBlock:
    """RAG 压缩的最小单位；不要再把关键步骤切成孤立句子。"""

    chunk_id: str
    manual_name: str
    block_type: str
    title: str
    text: str
    score: float = 0.0
    preserved_reason: str = ""
    image_refs: list[str] = field(default_factory=list)
    raw_text: str = ""
    fallback_used: bool = False


@dataclass(frozen=True)
class ContextPacket:
    """上下文压缩前后的中间包，便于后续扩展工具结果/安全状态。"""

    question: str
    facts: CriticalFacts
    evidence_blocks: list[EvidenceBlock]
    history_text: str
    visual_text: str
    memory_text: str = ""


@dataclass(frozen=True)
class CompressionTrace:
    """压缩观测信息。"""

    original_tokens: int
    final_tokens: int
    notes: list[str] = field(default_factory=list)
    evidence_block_count: int = 0
    critical_fact_count: int = 0
    compression_fallback_count: int = 0
    verifier_failed_reasons: list[str] = field(default_factory=list)
    preserved_block_types: list[str] = field(default_factory=list)


# 兼容旧 pipeline 命名。
ContextAssemblyTrace = CompressionTrace


@dataclass(frozen=True)
class AssembledPromptContext:
    """压缩后的 PromptContext 与 trace。"""

    context: PromptContext
    trace: CompressionTrace
