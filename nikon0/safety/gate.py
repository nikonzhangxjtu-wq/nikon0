"""Minimal safety gate."""

from __future__ import annotations

from uuid import uuid4

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import AgentResult
from nikon0.app.schemas.safety import ApprovalRequest, HandoffRequest, SafetyDecision
from nikon0.skills.routing_signals import has_approval_signal, has_handoff_signal


class SafetyGate:
    """Conservative phase1 safety gate."""

    async def check(self, context: AgentContext, result: AgentResult) -> SafetyDecision:
        message = context.request.message
        workflow = _latest_workflow_decision(context)
        if (
            bool(workflow.get("handoff_required"))
            or has_handoff_signal(message)
            or result.status == "handoff_required"
        ):
            handoff = HandoffRequest(
                handoff_id=f"handoff_{uuid4().hex}",
                trace_id=context.trace.trace_id,
                session_id=context.request.session_id,
                reason=result.handoff_reason or "request requires human service handling",
                payload={"message": message, "selected_skill": context.selected_skill, "workflow": workflow},
            )
            decision = SafetyDecision(
                allowed=False,
                risk_level="high",
                requires_human=True,
                reason="handoff required by safety gate",
                blocked_actions=["final_auto_resolution"],
                handoff_request=handoff,
            )
        elif bool(workflow.get("requires_approval")) or has_approval_signal(message) or result.risk_level == "high":
            approval = ApprovalRequest(
                approval_id=f"approval_{uuid4().hex}",
                trace_id=context.trace.trace_id,
                session_id=context.request.session_id,
                approval_type="answer",
                title="High-risk service response requires approval",
                reason="workflow requires approval" if workflow else "request contains high-risk enterprise service intent",
                risk_level="high",
                requested_action="send_high_risk_answer",
                payload={"message": message, "selected_skill": context.selected_skill, "workflow": workflow},
            )
            decision = SafetyDecision(
                allowed=False,
                risk_level="high",
                requires_human=True,
                reason="approval required by safety gate",
                blocked_actions=["send_high_risk_answer"],
                approval_request=approval,
            )
        else:
            decision = SafetyDecision(
                allowed=True,
                risk_level=result.risk_level,
                requires_human=False,
                reason="phase1 safety gate allowed low-risk response",
            )
        context.trace.safety_decisions.append(decision.model_dump())
        context.trace.add_event(
            "safety.check",
            decision.reason,
            allowed=decision.allowed,
            risk_level=decision.risk_level,
            requires_human=decision.requires_human,
            approval_id=decision.approval_request.approval_id if decision.approval_request else None,
            handoff_id=decision.handoff_request.handoff_id if decision.handoff_request else None,
        )
        return decision


def _latest_workflow_decision(context: AgentContext) -> dict:
    for event in reversed(context.trace.events):
        if event.stage == "workflow.decision":
            return dict(event.payload)
    return {}
