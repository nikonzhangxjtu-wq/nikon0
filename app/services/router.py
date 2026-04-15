"""Routing and RAG gate logic.

Important:
- Keep this module lightweight.
- In V1, a heuristic baseline is enough.
- You will improve with data-driven thresholds later.
"""

from __future__ import annotations

from dataclasses import dataclass


MANUAL_HINT_KEYWORDS = {
    "手册",
    "说明书",
    "安装",
    "拆卸",
    "更换",
    "步骤",
    "指示灯",
    "故障",
    "如何",
    "怎么",
    "clean",
    "install",
    "manual",
    "troubleshoot",
}

CS_HINT_KEYWORDS = {
    "退货",
    "退款",
    "换货",
    "发票",
    "物流",
    "投诉",
    "售后",
    "保修",
    "运费",
    "维修",
}


@dataclass
class RouteDecision:
    """Decision object passed to the pipeline."""

    needs_rag: bool
    domain_hint: str
    reason: str


class QuestionRouter:
    """V1 heuristic router.

    TODO (you):
    - Replace keyword matching with model-assisted classification if needed.
    - Add confidence scoring and logging.
    """

    def decide(self, question: str) -> RouteDecision:
        q = question.lower()

        manual_hits = sum(1 for kw in MANUAL_HINT_KEYWORDS if kw.lower() in q)
        cs_hits = sum(1 for kw in CS_HINT_KEYWORDS if kw.lower() in q)

        # Strategy:
        # 1) If manual intent is stronger, force RAG.
        # 2) If cs intent is stronger, default to non-RAG fallback.
        # 3) Tie/unknown: conservative default = try RAG first (you can change).
        if manual_hits > cs_hits:
            return RouteDecision(needs_rag=True, domain_hint="manual", reason="manual keywords matched")

        if cs_hits > manual_hits:
            return RouteDecision(
                needs_rag=False,
                domain_hint="customer_service",
                reason="customer service keywords matched",
            )

        return RouteDecision(
            needs_rag=True,
            domain_hint="unknown",
            reason="no strong domain signal; try retrieval first",
        )
