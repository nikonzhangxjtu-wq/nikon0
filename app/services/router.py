"""路由与 RAG 门控逻辑。

注意：
- 本模块保持轻量即可。
- V1 用启发式关键词足够起步。
- 后续可用数据驱动阈值或小型分类模型替换。
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
    """传给下游流水线的路由决策。"""

    needs_rag: bool
    domain_hint: str
    reason: str
    confidence: float = 0.0
    strategy: str = "heuristic_keyword"


class QuestionRouter:
    """V1 启发式路由器。

    TODO（你来补）：
    - 如需可换成小模型/大模型做意图分类
    - 增加置信度与日志，便于调参
    """


    def decide(self, question: str) -> RouteDecision:
        q = question.lower()

        manual_hits = sum(1 for kw in MANUAL_HINT_KEYWORDS if kw.lower() in q)
        cs_hits = sum(1 for kw in CS_HINT_KEYWORDS if kw.lower() in q)

        # 策略：
        # 1）说明书意图更强 → 走 RAG
        # 2）客服意图更强 → 默认不走 RAG（走兜底话术，后续可接政策库 RAG）
        # 3）平局/不明确 → 保守默认先试检索（可按需改成 false）
        if manual_hits > cs_hits:
            return RouteDecision(
                needs_rag=True,
                domain_hint="manual",
                reason="命中说明书类关键词",
                confidence=self._confidence_from_hits(manual_hits, cs_hits),
            )

        if cs_hits > manual_hits:
            return RouteDecision(
                needs_rag=False,
                domain_hint="customer_service",
                reason="命中客服类关键词",
                confidence=self._confidence_from_hits(cs_hits, manual_hits),
            )

        return RouteDecision(
            needs_rag=True,
            domain_hint="unknown",
            reason="领域信号不强，先尝试检索",
            confidence=0.35,
        )

    @staticmethod
    def _confidence_from_hits(primary_hits: int, secondary_hits: int) -> float:
        """将启发式命中数映射为 0～1 的粗粒度置信度。"""
        margin = max(primary_hits - secondary_hits, 0)
        score = 0.55 + 0.1 * margin + 0.05 * primary_hits
        return min(score, 0.95)
