"""把 pipeline 的分散上下文组装成统一证据包。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.services.memory.v3.types import TurnEvidencePacket


class TurnEvidencePacketBuilder:
    @staticmethod
    def build(
        *,
        session_id: str | None,
        user_id: str | None,
        question: str,
        answer: str,
        route_domain_hint: str,
        route_needs_rag: bool,
        branch_name: str,
        recent_history: str = "",
        memory_context_used: str = "",
        visual_context: str = "",
        rag_context: str = "",
        branch_result: dict[str, Any] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> TurnEvidencePacket:
        return TurnEvidencePacket(
            session_id=(session_id or "").strip(),
            user_id=(user_id or None),
            turn_id=f"turn_{uuid.uuid4().hex[:16]}",
            timestamp=time.time(),
            question=question or "",
            answer=answer or "",
            route_domain_hint=route_domain_hint or "",
            route_needs_rag=bool(route_needs_rag),
            branch_name=branch_name or "",
            recent_history=recent_history or "",
            memory_context_used=memory_context_used or "",
            visual_context=visual_context or "",
            rag_context=rag_context or "",
            branch_result=branch_result,
            tool_results=tool_results or [],
        )
