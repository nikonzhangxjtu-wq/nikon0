"""Declarative workflow protocol decisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowRiskLevel = Literal["low", "medium", "high"]


class WorkflowProtocol(BaseModel):
    name: str
    intent: str
    risk_level: WorkflowRiskLevel
    required_slots: list[str] = Field(default_factory=list)
    approval_required: bool = False
    handoff_required: bool = False
    next_tool: str = ""
    stop_when: list[str] = Field(default_factory=list)
    user_message_when_blocked: str = ""


class WorkflowDecision(BaseModel):
    workflow_name: str
    intent: str
    risk_level: WorkflowRiskLevel
    missing_slots: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    handoff_required: bool = False
    next_tool: str = ""
    stop_when: list[str] = Field(default_factory=list)
    reason: str = ""
    user_message_when_blocked: str = ""


class WorkflowRuntime:
    def __init__(self, protocols: list[WorkflowProtocol]) -> None:
        self._protocols = {protocol.intent: protocol for protocol in protocols}
        self._by_name = {protocol.name: protocol for protocol in protocols}

    def decide(
        self,
        *,
        message: str,
        slots: dict[str, Any],
        intent: str,
        is_cancel: bool = False,
    ) -> WorkflowDecision:
        if is_cancel:
            protocol = self._by_name["case_intake_cancel"]
            return self._decision(protocol, slots, reason="cancel intent short-circuited active intake")

        protocol = self._protocols.get(intent) or self._protocols["repair"]
        reason = f"matched workflow protocol by intent={intent or 'unknown'}"
        if intent == "unknown":
            reason = "unknown intent falls back to repair intake collection"
        if "投诉" in message or "转人工" in message or "人工客服" in message or "升级" in message:
            protocol = self._protocols["complaint"]
            reason = "message contains complaint or handoff signal"
        elif "退款" in message or "退货" in message or "换货" in message or "赔偿" in message:
            protocol = self._protocols["refund"]
            reason = "message contains refund or compensation signal"
        return self._decision(protocol, slots, reason=reason)

    @staticmethod
    def _decision(protocol: WorkflowProtocol, slots: dict[str, Any], *, reason: str) -> WorkflowDecision:
        missing_slots = [
            slot
            for slot in protocol.required_slots
            if not str(slots.get(slot) or "").strip()
        ]
        return WorkflowDecision(
            workflow_name=protocol.name,
            intent=protocol.intent,
            risk_level=protocol.risk_level,
            missing_slots=missing_slots,
            requires_approval=protocol.approval_required,
            handoff_required=protocol.handoff_required,
            next_tool=protocol.next_tool,
            stop_when=list(protocol.stop_when),
            reason=reason,
            user_message_when_blocked=protocol.user_message_when_blocked,
        )


def default_workflow_runtime() -> WorkflowRuntime:
    return WorkflowRuntime(
        [
            WorkflowProtocol(
                name="repair_intake",
                intent="repair",
                risk_level="low",
                required_slots=["product_model", "issue", "contact_phone"],
                next_tool="case-intake.collect_case_intake",
                stop_when=["ready", "collecting"],
            ),
            WorkflowProtocol(
                name="refund_intake",
                intent="refund",
                risk_level="high",
                required_slots=["order_id", "refund_reason", "contact_phone"],
                approval_required=True,
                next_tool="case-intake.collect_case_intake",
                stop_when=["approval_created", "collecting"],
                user_message_when_blocked="当前请求涉及高风险服务动作，审批通过前不会自动执行或承诺退款结果。",
            ),
            WorkflowProtocol(
                name="complaint_escalation",
                intent="complaint",
                risk_level="high",
                required_slots=["issue", "contact_phone"],
                handoff_required=True,
                next_tool="case-intake.collect_case_intake",
                stop_when=["handoff_required"],
                user_message_when_blocked="当前请求需要人工处理，已生成转人工请求。",
            ),
            WorkflowProtocol(
                name="case_intake_cancel",
                intent="cancel",
                risk_level="low",
                next_tool="case-intake.try_cancel_case_intake",
                stop_when=["cancelled"],
            ),
        ]
    )
