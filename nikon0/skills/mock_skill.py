"""Phase 1 mock skill used to prove the runtime loop."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import Evidence, SkillManifest, SkillMatch, SkillResult, StateUpdate


class MockSkill:
    name = "mock_enterprise_assistant"
    description = "Phase 1 mock skill for nikon0 runtime verification."
    risk_level = "low"
    manifest = SkillManifest(
        name=name,
        title="Mock Enterprise Assistant",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        capabilities=["phase1_runtime_fallback"],
        required_tools=[],
        risk_level="low",
    )

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        if context.request.message.strip():
            return SkillMatch(
                matched=True,
                confidence=0.5,
                reason="phase1 fallback skill handles any non-empty message",
            )
        return SkillMatch(matched=False, confidence=0.0, reason="empty message")

    async def run(self, context: AgentContext) -> SkillResult:
        message = context.request.message
        evidence = Evidence(
            evidence_id=f"user:{context.trace.trace_id}:message",
            source="user",
            text=message,
            confidence=1.0,
        )
        return SkillResult(
            status="success",
            answer_draft=(
                "nikon0 Phase 1 runtime 已接收到你的请求。"
                "当前由 SupervisorAgent 调度 MockSkill 完成最小闭环；"
                "后续会逐步接入产品支持、工单、订单、退款和安全审批能力。"
            ),
            evidence=[evidence],
            state_updates=[
                StateUpdate(
                    key="last_user_message",
                    value=message,
                    reason="record latest user request for phase1 traceability",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            risk_level="low",
        )
