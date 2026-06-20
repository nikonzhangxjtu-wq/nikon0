"""Case-intake helper tools for slot extraction."""

from __future__ import annotations

import re
from typing import Any

from nikon0.app.schemas.capability import ToolCallRequest, ToolCallResult, ToolSpec


class ExtractCaseSlotsTool:
    """Extract preliminary service/refund/complaint slots from free text."""

    spec = ToolSpec(
        service_id="case-intake",
        tool_name="extract_case_slots",
        description="Extract intent, product model, contact phone, order id, and missing slots for case intake.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        message = str(request.arguments.get("message") or request.arguments.get("question") or "").strip()
        slots = _extract_slots(message)
        intent = _intent(message)
        missing_slots = _missing_slots(intent, slots)
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={
                "intent": intent,
                "slots": slots,
                "missing_slots": missing_slots,
                "confidence": _confidence(intent, slots, missing_slots),
            },
        )


def _extract_slots(message: str) -> dict[str, Any]:
    phone = _first_match(r"(?<!\d)(1[3-9]\d{9})(?!\d)", message)
    order_id = _first_match(r"(?:订单号|订单|order)[:：\s#-]*([A-Za-z0-9-]{4,})", message, flags=re.I)
    explicit_model = _first_match(r"(?:型号|机型|产品型号)[:：\s]*([A-Za-z][A-Za-z0-9-]{2,})", message)
    model = explicit_model or _first_match(r"\b([A-Z]{1,8}[A-Z0-9-]*\d{2,}[A-Z0-9-]*)\b", message)
    issue = message if _looks_like_issue(message) else ""
    return {
        "contact_phone": phone,
        "order_id": order_id,
        "product_model": model,
        "issue": issue,
    }


def _intent(message: str) -> str:
    if any(keyword in message for keyword in ("投诉", "主管", "人工", "升级")):
        return "complaint"
    if any(keyword in message for keyword in ("退款", "退货", "换货", "赔偿", "换新")):
        return "refund"
    if any(keyword in message for keyword in ("报修", "维修", "售后", "坏了", "故障", "不能用", "无法启动")):
        return "repair"
    return "unknown"


def _missing_slots(intent: str, slots: dict[str, Any]) -> list[str]:
    required_by_intent = {
        "repair": ("product_model", "contact_phone", "issue"),
        "refund": ("order_id", "contact_phone"),
        "complaint": ("contact_phone", "issue"),
        "unknown": ("intent",),
    }
    required = required_by_intent.get(intent, ("intent",))
    missing = [name for name in required if not slots.get(name)]
    return missing


def _confidence(intent: str, slots: dict[str, Any], missing_slots: list[str]) -> float:
    if intent == "unknown":
        return 0.2
    filled = len([value for value in slots.values() if value])
    score = 0.45 + min(0.4, filled * 0.12) - min(0.25, len(missing_slots) * 0.08)
    return round(max(0.1, min(0.95, score)), 2)


def _looks_like_issue(message: str) -> bool:
    return any(keyword in message for keyword in ("坏", "故障", "不能", "无法", "不转", "异响", "漏水", "报修", "维修"))


def _first_match(pattern: str, text: str, *, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""
