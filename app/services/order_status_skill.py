"""订单进度 Skill：查询订单/物流状态并生成结构化上下文块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class OrderStatusHit:
    order_id: str
    status: str
    logistics_status: str = ""
    eta: str = ""
    updated_at: str = ""
    can_refund: str = ""
    note: str = ""


@dataclass(frozen=True)
class OrderStatusResult:
    ok: bool
    question: str
    search_query: str = ""
    context_block: str = ""
    hits: list[OrderStatusHit] = field(default_factory=list)
    fallback_reason: str = ""


class OrderStatusProvider(Protocol):
    def search_order_status(self, query: str, *, top_k: int = 3) -> list[OrderStatusHit]:
        ...


class NullOrderStatusProvider:
    def search_order_status(self, query: str, *, top_k: int = 3) -> list[OrderStatusHit]:
        _ = (query, top_k)
        return []


class OrderStatusSkill:
    """订单状态查询 Skill：适用于“订单到哪了/物流进度/催单”类问题。"""

    def __init__(self, provider: OrderStatusProvider | None = None) -> None:
        self._provider = provider or NullOrderStatusProvider()

    def run(self, question: str, *, enrichment: str = "", top_k: int = 3) -> OrderStatusResult:
        q = (question or "").strip()
        if not q:
            return OrderStatusResult(ok=False, question="", fallback_reason="empty_question")
        search_query = self._build_query(q, enrichment=enrichment)
        try:
            hits = self._provider.search_order_status(search_query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return OrderStatusResult(
                ok=False,
                question=q,
                search_query=search_query,
                fallback_reason=f"provider_error:{exc}",
            )
        if not hits:
            return OrderStatusResult(
                ok=False,
                question=q,
                search_query=search_query,
                fallback_reason="no_hits",
            )
        context_block = self._build_context_block(hits)
        return OrderStatusResult(
            ok=True,
            question=q,
            search_query=search_query,
            context_block=context_block,
            hits=hits,
        )

    @staticmethod
    def _build_query(question: str, *, enrichment: str) -> str:
        e = (enrichment or "").strip()
        if e:
            return f"{question} 订单状态 物流进度 售后 {e[:120]}"
        return f"{question} 订单状态 物流进度 售后"

    @staticmethod
    def _build_context_block(hits: list[OrderStatusHit]) -> str:
        lines = ["[订单进度信息]"]
        for idx, h in enumerate(hits[:5], start=1):
            lines.append(f"{idx}. 订单号: {h.order_id or '-'}")
            lines.append(f"   状态: {h.status or '-'}")
            if h.logistics_status:
                lines.append(f"   物流: {h.logistics_status}")
            if h.eta:
                lines.append(f"   预计送达: {h.eta}")
            if h.updated_at:
                lines.append(f"   更新时间: {h.updated_at}")
            if h.can_refund:
                lines.append(f"   退款可行性: {h.can_refund}")
            if h.note:
                lines.append(f"   备注: {h.note}")
        return "\n".join(lines).strip()
