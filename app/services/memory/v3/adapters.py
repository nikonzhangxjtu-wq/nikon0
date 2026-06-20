"""多来源证据 adapter。

关键设计：adapter 只负责把来源转为 RawEvidence，不负责最终写库。
这样用户当前轮、RAG 反馈、视觉结果、MCP 工具结果都能统一进入后续门控。
"""

from __future__ import annotations

import re
from typing import Any

from app.services.memory.v3.types import RawEvidence, TurnEvidencePacket


class UserUtteranceAdapter:
    source = "user_current"
    priority = 80

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        text = packet.question or ""
        structured: dict[str, Any] = {}
        models = _extract_product_models(text)
        if models:
            structured["product_model"] = models
        fault_codes = _extract_fault_codes(text)
        if fault_codes:
            structured["fault_code"] = fault_codes
        phones = _extract_phones(text)
        if phones:
            structured["phone"] = phones
        orders = _extract_order_ids(text)
        if orders:
            structured["order_id"] = orders
        attempted = _extract_attempted_actions(text)
        if attempted:
            structured["attempted_action"] = attempted
        goals = _extract_user_goals(text)
        if goals:
            structured["user_goal"] = goals
        symptoms = _extract_symptoms(text)
        if symptoms:
            structured["symptom"] = symptoms

        evidence_type = "explicit_fact"
        if _has_forget_intent(text):
            evidence_type = "forget"
        elif _has_correction_intent(text):
            evidence_type = "correction"
        elif _has_remember_intent(text):
            evidence_type = "remember"

        return [
            RawEvidence(
                source=self.source,
                evidence_type=evidence_type,
                text=text,
                structured=structured,
                confidence=0.9 if structured else 0.0,
                source_priority=self.priority,
            )
        ]


class HistoryReferenceAdapter:
    source = "history"
    priority = 45

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        if not packet.recent_history:
            return []
        # 这里先做轻量指代补全；更复杂的“这个/刚才那个”交给 LLM Judge。
        structured: dict[str, Any] = {}
        models = _extract_product_models(packet.recent_history)
        if re.search(r"这个|刚才|之前|上次|还是|试过", packet.question) and models:
            structured["product_model"] = models[-1:]
        return [
            RawEvidence(
                source=self.source,
                evidence_type="reference",
                text=packet.recent_history,
                structured=structured,
                confidence=0.55,
                source_priority=self.priority,
            )
        ]


class AssistantActionAdapter:
    source = "assistant"
    priority = 55

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        text = packet.answer or ""
        actions = _extract_suggested_actions(text)
        missing = _extract_missing_slots(text)
        structured: dict[str, Any] = {}
        if actions:
            structured["assistant_commitment"] = actions
        if missing:
            structured["missing_slot"] = missing
        return [
            RawEvidence(
                source=self.source,
                evidence_type="assistant_action",
                text=text,
                structured=structured,
                confidence=0.65 if structured else 0.0,
                source_priority=self.priority,
            )
        ]


class RagInteractionAdapter:
    source = "rag"
    priority = 20

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        if not packet.rag_context:
            return []
        # RAG 原文是知识源，不是用户记忆；这里只允许问题里的用户事实进入候选。
        structured: dict[str, Any] = {}
        attempted = _extract_attempted_actions(packet.question)
        if attempted:
            structured["attempted_action"] = attempted
        return [
            RawEvidence(
                source=self.source,
                evidence_type="rag_interaction",
                text=packet.question,
                structured=structured,
                confidence=0.45 if structured else 0.0,
                source_priority=self.priority,
            )
        ]


class VisualEvidenceAdapter:
    source = "visual"
    priority = 65

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        if not packet.visual_context:
            return []
        structured: dict[str, Any] = {}
        fault_codes = _extract_fault_codes(packet.visual_context)
        if fault_codes:
            structured["fault_code"] = fault_codes
        indicators = re.findall(r"(红灯|绿灯|黄灯|蓝灯|闪烁|常亮)", packet.visual_context)
        if indicators:
            structured["visible_indicator"] = sorted(set(indicators))
        return [
            RawEvidence(
                source=self.source,
                evidence_type="visual_fact",
                text=packet.visual_context,
                structured=structured,
                confidence=0.65 if structured else 0.0,
                source_priority=self.priority,
            )
        ]


class BranchResultAdapter:
    source = "branch"
    priority = 90

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        payload = packet.branch_result or {}
        if not payload:
            return []
        structured = _flatten_business_payload(payload)
        return [
            RawEvidence(
                source=self.source,
                evidence_type="tool_fact" if structured else "branch_result",
                text=str(payload),
                structured=structured,
                confidence=0.9 if structured else 0.0,
                source_priority=self.priority,
            )
        ]


class ToolResultAdapter:
    source = "tool"
    priority = 100

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        evidence: list[RawEvidence] = []
        for payload in packet.tool_results:
            structured = _flatten_business_payload(payload)
            evidence.append(
                RawEvidence(
                    source=self.source,
                    evidence_type="tool_fact",
                    text=str(payload),
                    structured=structured,
                    confidence=0.98 if structured else 0.0,
                    source_priority=self.priority,
                )
            )
        return evidence


class EvidenceAdapterPipeline:
    def __init__(self, adapters: list[object] | None = None) -> None:
        self.adapters = adapters or [
            UserUtteranceAdapter(),
            HistoryReferenceAdapter(),
            AssistantActionAdapter(),
            RagInteractionAdapter(),
            VisualEvidenceAdapter(),
            BranchResultAdapter(),
            ToolResultAdapter(),
        ]

    def collect(self, packet: TurnEvidencePacket) -> list[RawEvidence]:
        evidence: list[RawEvidence] = []
        for adapter in self.adapters:
            evidence.extend(adapter.collect(packet))
        return [item for item in evidence if item.confidence > 0 and item.structured]


def _extract_product_models(text: str) -> list[str]:
    values = re.findall(r"\b[A-Z]{1,6}[-_]?\d{2,6}[A-Z0-9-]*\b", text or "")
    return _unique(values)


def _extract_fault_codes(text: str) -> list[str]:
    values = re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", text or "")
    return _unique([v for v in values if not re.fullmatch(r"1[3-9]\d{9}", v)])


def _extract_phones(text: str) -> list[str]:
    return _unique(re.findall(r"(?<!\d)1[3-9]\d{9}(?!\d)", text or ""))


def _extract_order_ids(text: str) -> list[str]:
    return _unique(re.findall(r"(?:订单号?|order)[:：\s]*([A-Za-z0-9-]{8,32})", text or "", re.I))


def _extract_attempted_actions(text: str) -> list[str]:
    actions: list[str] = []
    patterns = [
        (r"(?:已经|已|试过|尝试过|按你说的)(?:把|进行)?([^，。；;!?？]{2,18}?)(?:了|过)", ""),
        (r"(断电重启|重启|清洗滤网|清洗过滤网|拆下滤网|更换滤芯|重新安装)", ""),
    ]
    for pattern, _ in patterns:
        for match in re.findall(pattern, text or ""):
            value = match if isinstance(match, str) else match[0]
            value = value.strip(" ，,。了过")
            if value:
                actions.append(_normalize_action(value))
    return _unique(actions)


def _extract_user_goals(text: str) -> list[str]:
    goals = []
    if re.search(r"报修|维修|上门", text or ""):
        goals.append("报修")
    if re.search(r"退款|退货|退钱", text or ""):
        goals.append("退款")
    if re.search(r"投诉|升级处理", text or ""):
        goals.append("投诉")
    return goals


def _extract_symptoms(text: str) -> list[str]:
    symptoms = []
    for pattern in [r"(还是不行)", r"((?:红灯|绿灯|黄灯|蓝灯).{0,4}(?:闪|闪烁|常亮))", r"(无法启动|不能启动|不制冷|不工作)"]:
        symptoms.extend(re.findall(pattern, text or ""))
    return _unique(symptoms)


def _extract_suggested_actions(text: str) -> list[str]:
    actions = []
    for pattern in [r"建议你?先([^，。；;]{2,24})", r"请(?:你)?(?:先)?([^，。；;]{2,24})"]:
        actions.extend([m.strip() for m in re.findall(pattern, text or "")])
    return _unique(actions)


def _extract_missing_slots(text: str) -> list[str]:
    slots = []
    mapping = {
        "手机号": r"手机号|联系电话",
        "地址": r"地址|上门地址|详细地址",
        "购买时间": r"购买时间|购买日期|购机时间",
        "订单号": r"订单号|订单",
    }
    for slot, pattern in mapping.items():
        if re.search(rf"(?:还需要|请补充|需要提供).{{0,12}}{pattern}", text or ""):
            slots.append(slot)
    return slots


def _flatten_business_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "case_id": "case_id",
        "status": "case_status",
        "case_status": "case_status",
        "missing_slots": "missing_slot",
        "collected_fields": "collected_field",
        "product_model": "product_model",
        "order_id": "order_id",
        "phone": "phone",
    }
    structured: dict[str, Any] = {}
    for key, out_key in mapping.items():
        if key in payload and payload[key] not in (None, "", []):
            structured[out_key] = payload[key]
    return structured


def _has_remember_intent(text: str) -> bool:
    return bool(re.search(r"记住|以后默认|下次还用|以后都用|默认用", text or ""))


def _has_forget_intent(text: str) -> bool:
    return bool(re.search(r"忘掉|删除|不要记住|别保存|不要保存", text or ""))


def _has_correction_intent(text: str) -> bool:
    return bool(re.search(r"不是.+是|刚才说错|改成|更正", text or ""))


def _normalize_action(value: str) -> str:
    if "断电" in value and "重启" in value:
        return "断电重启"
    if "过滤网" in value:
        return value.replace("过滤网", "滤网")
    return value


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
