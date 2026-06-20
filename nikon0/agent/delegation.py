"""Model-driven, guard-railed business-agent delegation."""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from nikon0.app.schemas.agent import AgentContext


DelegationAction = Literal["general", "support", "service", "clarify", "handoff"]


class DelegationModelClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class AgentDelegationPlan(BaseModel):
    action: DelegationAction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    source: Literal["llm", "fallback"] = "llm"
    raw_response: str = ""


class AgentDelegationPlanner:
    """Lets the model decide business ownership while Runtime owns the bounds."""

    def __init__(self, client: DelegationModelClient, *, min_confidence: float = 0.60) -> None:
        self.client = client
        self.min_confidence = min(1.0, max(0.0, min_confidence))

    async def plan(self, context: AgentContext, *, stage: str = "initial", handoff: dict[str, Any] | None = None) -> AgentDelegationPlan:
        try:
            raw = await self.client.complete(self._messages(context, stage=stage, handoff=handoff))
            payload = _parse_json(raw)
            plan = AgentDelegationPlan(
                action=str(payload.get("action") or ""),
                confidence=_confidence(payload.get("confidence")),
                reason=str(payload.get("reason") or "model delegation"),
                source="llm",
                raw_response=raw,
            )
            if plan.confidence < self.min_confidence:
                raise ValueError("delegation confidence below threshold")
            return plan
        except Exception as exc:  # noqa: BLE001
            return self._fallback(context, f"invalid delegation: {type(exc).__name__}")

    @staticmethod
    def _messages(context: AgentContext, *, stage: str, handoff: dict[str, Any] | None) -> list[dict[str, Any]]:
        payload = {
            "stage": stage,
            "user_message": context.request.message,
            "governed_memory": context.memory_context[-1200:],
            "recent_conversation": context.transcript_context[-1600:],
            "support_handoff": handoff or {},
            "allowed_actions": ["general", "support", "service", "clarify", "handoff"],
            "rules": [
                "general is only for greetings, thanks, or capability questions without a business request",
                "support handles product/manual/diagnosis questions",
                "service handles repair, refund, complaint, or service workflow",
                "after a support handoff, decide from its evidence and uncertainty whether service is actually needed",
                "return only JSON with action, confidence, and reason",
            ],
        }
        return [
            {
                "role": "system",
                "content": "You are nikon0's delegation planner. You recommend one bounded next action; Runtime enforces policy.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    @staticmethod
    def _fallback(context: AgentContext, reason: str) -> AgentDelegationPlan:
        # This is a safety fallback, not a routing substitute. It only distinguishes
        # requests that must not continue without a reliable service decision.
        message = context.request.message.lower()
        high_risk = any(token in message for token in ("退款", "退货", "投诉", "换货", "refund", "complaint"))
        return AgentDelegationPlan(
            action="handoff" if high_risk else "clarify",
            confidence=0.0,
            reason=reason,
            source="fallback",
        )


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("delegation response must be a JSON object")
    return value


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
