"""按需读取计划。"""

from __future__ import annotations

import re

from app.core.config import settings
from app.services.memory.v3.types import MemoryReadRequest

# TODO:获取意图太粗糙
class MemoryReadPlanner:
    def plan(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        question: str,
        recent_history: str = "",
        route_domain_hint: str | None = None,
    ) -> MemoryReadRequest:
        query = question or ""
        intents = _detect_intents(query, route_domain_hint or "")
        entities = _extract_entities(query + "\n" + (recent_history or ""))
        # 决定是否是指代性问题
        has_reference = bool(re.search(r"这个|那个|刚才|之前|上次|还是|进展|历史", query))
        # 这是偏“多轮上下文连续性”的记忆层，需要session记忆
        include_session = bool(session_id) and (
            has_reference
            or bool(intents & {"repair", "refund", "complaint", "status", "case_intake"})
            or bool(recent_history)
        )
        # 这是偏“用户画像”的记忆层，需要用户画像
        include_profile = bool(user_id) and bool(
            intents & {"profile", "repair", "status"} or re.search(r"联系|手机号|偏好|默认|我的", query)
        )
        # 这是偏“情景向量”的记忆层，需要情景向量
        include_episodic = bool(user_id) and bool(
            intents & {"status", "complaint", "repair"} or re.search(r"之前|上次|历史|进展|处理过", query)
        )
        if not session_id and not user_id:
            # 如果既没有session_id也没有user_id，则不读取session记忆、用户画像和情景向量
            include_session = include_profile = include_episodic = False
        if "chitchat" in intents and len(intents) == 1:
            # 如果只有chitchat，则不读取情景向量
            include_episodic = False
        reason = ",".join(sorted(intents)) or "default"
        return MemoryReadRequest(
            session_id=session_id,
            user_id=user_id,
            query=query,
            intents=sorted(intents),
            entities=entities,
            include_session=include_session,
            include_profile=include_profile,
            include_episodic=include_episodic,
            budget_tokens=settings.context_memory_token_budget,
            reason=reason,
        )


def _detect_intents(query: str, route_domain_hint: str) -> set[str]:
    intents: set[str] = set()
    text = query or ""
    if route_domain_hint == "case_intake":
        intents.add("case_intake")
    if re.search(r"报修|维修|故障|不行|故障码|红灯|闪", text):
        intents.add("repair")
    if re.search(r"退款|退货", text):
        intents.add("refund")
    if re.search(r"投诉|升级处理", text):
        intents.add("complaint")
    if re.search(r"进展|状态|到哪|处理了吗|工单", text):
        intents.add("status")
    if re.search(r"记住|以后|默认|偏好|联系", text):
        intents.add("profile")
    if not intents:
        intents.add("chitchat")
    return intents


def _extract_entities(text: str) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    models = re.findall(r"\b[A-Z]{1,6}[-_]?\d{2,6}[A-Z0-9-]*\b", text or "")
    if models:
        entities["product_model"] = sorted(set(models))
    fault_codes = re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", text or "")
    if fault_codes:
        entities["fault_code"] = sorted(set(fault_codes))
    phones = re.findall(r"(?<!\d)1[3-9]\d{9}(?!\d)", text or "")
    if phones:
        entities["phone"] = sorted(set(phones))
    return entities
