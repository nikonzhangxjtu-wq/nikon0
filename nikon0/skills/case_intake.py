"""Case intake skill routed through ToolRuntime/MCP."""

from __future__ import annotations

from typing import Any

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import (
    Evidence,
    FallbackPolicy,
    SkillManifest,
    SkillMatch,
    SkillResult,
    StateUpdate,
    StickyPolicy,
    ToolCallRequest,
)
from nikon0.tools.case_intake import ExtractCaseSlotsTool
from nikon0.tools.runtime import ToolRegistry, ToolRuntime
from nikon0.workflows.runtime import WorkflowDecision, WorkflowRuntime, default_workflow_runtime
from nikon0.skills.routing_signals import CASE_INTAKE_KEYWORDS


_MANUAL_PROHIBITION_PATTERNS: tuple[str, ...] = (
    "为什么不能用",
    "为何不能用",
    "为什么不能用排插",
    "能不能用排插",
    "能不能用延长线",
)

_CANCEL_KEYWORDS: tuple[str, ...] = (
    "取消报修",
    "取消工单",
    "不报了",
    "不用了",
    "先不报",
    "算了",
)


class CaseIntakeSkill:
    """Collects service/refund/complaint fields through ToolRuntime.

    The Skill no longer calls the legacy local implementation directly. It
    requests MCP tools, then the AgentLoop feeds tool results back on the next
    turn so this Skill can produce the final answer and memory updates.
    """

    name = "case_intake"
    description = "Collects repair, refund, and complaint intake fields via MCP tools."
    risk_level = "medium"
    manifest = SkillManifest(
        name=name,
        title="Case Intake",
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["message", "session_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "missing_slots": {"type": "array", "items": {"type": "string"}},
                "ticket_payload": {"type": "object"},
            },
        },
        capabilities=[
            "repair_intake",
            "refund_intake",
            "complaint_intake",
            "case_intake_cancel",
        ],
        required_tools=[
            "case-intake.extract_case_slots",
            "case-intake.collect_case_intake",
            "case-intake.try_cancel_case_intake",
        ],
        risk_level="medium",
        sticky_policy=StickyPolicy(
            enabled=True,
            continue_when=["collecting"],
            exit_when=["ready", "cancelled"],
            max_turns=6,
            priority=10,
        ),
        fallback_policy=FallbackPolicy(allow_general_fallback=False, allow_handoff=True, retry_on_tool_error=True),
    )

    def __init__(
        self,
        *,
        service_id: str = "case-intake",
        collect_tool: str = "collect_case_intake",
        status_tool: str = "get_case_intake_status",
        cancel_tool: str = "try_cancel_case_intake",
        workflow_runtime: WorkflowRuntime | None = None,
    ) -> None:
        self.service_id = service_id
        self.collect_tool = collect_tool
        self.status_tool = status_tool
        self.cancel_tool = cancel_tool
        self.workflow_runtime = workflow_runtime or default_workflow_runtime()
        self.local_tool_runtime = ToolRuntime(registry=ToolRegistry([ExtractCaseSlotsTool()]))

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        """Rule-fallback hints only; model/planner/sticky routes trust upstream selection."""
        if self._latest_case_intake_tool_result(context) is not None:
            return SkillMatch(
                matched=True,
                confidence=0.96,
                reason="case-intake tool result is ready to consume",
            )
        if self._has_pending_intake(context):
            return SkillMatch(
                matched=True,
                confidence=0.85,
                reason="session has active case intake state",
            )
        message = context.request.message.strip()
        if any(pattern in message for pattern in _MANUAL_PROHIBITION_PATTERNS):
            return SkillMatch(
                matched=False,
                confidence=0.0,
                reason="manual prohibition question, not case intake",
            )
        lowered = message.lower()
        matched_keywords = [keyword for keyword in CASE_INTAKE_KEYWORDS if keyword.lower() in lowered]
        if matched_keywords:
            return SkillMatch(
                matched=True,
                confidence=0.9,
                reason="matched case intake keywords: " + ", ".join(matched_keywords[:3]),
            )
        return SkillMatch(
            matched=False,
            confidence=0.0,
            reason="no case intake signal for rule fallback",
        )

    async def run(self, context: AgentContext) -> SkillResult:
        tool_result = self._latest_case_intake_tool_result(context)
        if tool_result is not None:
            return self._result_from_tool_result(context, tool_result)

        decision = await self._workflow_decision(context)
        tool_ref = decision.next_tool or f"{self.service_id}.{self.collect_tool}"
        tool_name = tool_ref.split(".", 1)[1] if "." in tool_ref else tool_ref
        arguments = {
            "question": context.request.message,
            "session_id": context.request.session_id,
            "conversation_history": context.transcript_context,
            "enrichment": context.memory_context,
            "workflow": decision.model_dump(),
        }
        return SkillResult(
            status="success",
            answer_draft="",
            tool_calls=[
                ToolCallRequest(
                    service_id=self.service_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    risk_level="medium" if decision.risk_level == "high" else decision.risk_level,
                )
            ],
            risk_level=decision.risk_level,
            handoff_reason="workflow requires human escalation" if decision.handoff_required else None,
        )

    def _result_from_tool_result(self, context: AgentContext, tool_result: dict[str, Any]) -> SkillResult:
        if not bool(tool_result.get("ok")):
            evidence = Evidence(
                evidence_id=f"case_intake:{context.trace.trace_id}:tool_error",
                source="tool",
                text=str(tool_result.get("error_message") or "case intake tool failed"),
                payload=tool_result,
                confidence=1.0,
            )
            return SkillResult(
                status="failed",
                answer_draft="当前工单服务暂时不可用，请稍后再试或转人工客服处理。",
                evidence=[evidence],
                risk_level="medium",
            )

        payload = dict(tool_result.get("data") or {})
        if "cancelled" in payload:
            payload = {
                "completed": False,
                "exited": bool(payload.get("cancelled")),
                "reply_text": "好的，已取消当前工单收集。",
                "missing_slots": [],
                "ticket_payload": {},
                "context_block": "[工单收集状态]\nstatus: cancelled",
            }
        completed = bool(payload.get("completed"))
        exited = bool(payload.get("exited"))
        reply_text = str(payload.get("reply_text") or "")
        missing_slots = [
            str(item)
            for item in payload.get("missing_slots", [])
            if isinstance(item, str)
        ]
        ticket_payload_raw = payload.get("ticket_payload")
        ticket_payload = {
            str(key): str(value)
            for key, value in ticket_payload_raw.items()
        } if isinstance(ticket_payload_raw, dict) else {}
        context_block = str(payload.get("context_block") or reply_text)

        evidence = Evidence(
            evidence_id=f"case_intake:{context.trace.trace_id}:tool_result",
            source="tool",
            text=context_block,
            payload=payload,
            confidence=1.0,
        )
        status = "success" if completed else "needs_more_info"
        if exited:
            status = "failed"
        state_updates = [
            StateUpdate(
                key="case_intake",
                value={
                    "status": "cancelled" if exited else ("ready" if completed else "collecting"),
                    "completed": completed,
                    "missing_slots": missing_slots,
                    "ticket_payload": ticket_payload,
                    "exited": exited,
                    **self._workflow_state(context),
                },
                reason="case intake MCP tool returned updated intake state",
                evidence_ids=[evidence.evidence_id],
            )
        ]
        return SkillResult(
            status=status,
            answer_draft=reply_text,
            evidence=[evidence],
            state_updates=state_updates,
            risk_level="medium" if self._is_risky_intake(ticket_payload) else "low",
            handoff_reason="case intake is ready for human service" if completed else None,
        )

    async def _workflow_decision(self, context: AgentContext) -> WorkflowDecision:
        tool_runtime = context.tool_runtime if isinstance(context.tool_runtime, ToolRuntime) else self.local_tool_runtime
        extract_result = await tool_runtime.call_step(
            context,
            ToolCallRequest(
                service_id=self.service_id,
                tool_name="extract_case_slots",
                arguments={"message": context.request.message},
                risk_level="low",
            ),
        )
        data = extract_result.data if extract_result.ok else {}
        slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
        intent = str(data.get("intent") or "unknown")
        decision = self.workflow_runtime.decide(
            message=context.request.message,
            slots=slots,
            intent=intent,
            is_cancel=self._should_cancel(context),
        )
        context.trace.add_event(
            "workflow.select",
            f"selected workflow {decision.workflow_name}",
            workflow_name=decision.workflow_name,
            intent=decision.intent,
            source_intent=intent,
        )
        context.trace.add_event(
            "workflow.decision",
            decision.reason,
            **decision.model_dump(),
        )
        return decision

    @staticmethod
    def _latest_workflow_decision(context: AgentContext) -> dict[str, Any]:
        for event in reversed(context.trace.events):
            if event.stage == "workflow.decision":
                return dict(event.payload)
        return {}

    def _workflow_state(self, context: AgentContext) -> dict[str, Any]:
        decision = self._latest_workflow_decision(context)
        if not decision:
            return {}
        return {
            "workflow_name": decision.get("workflow_name"),
            "workflow_intent": decision.get("intent"),
            "workflow_status": "handoff_required"
            if decision.get("handoff_required")
            else ("approval_required" if decision.get("requires_approval") else "collecting"),
            "workflow_missing_slots": decision.get("missing_slots", []),
            "requires_approval": bool(decision.get("requires_approval")),
            "handoff_required": bool(decision.get("handoff_required")),
            "next_tool": decision.get("next_tool"),
            "risk_level": decision.get("risk_level"),
            "reason": decision.get("reason"),
        }

    def _latest_case_intake_tool_result(self, context: AgentContext) -> dict[str, Any] | None:
        for result in reversed(context.tool_results):
            if (
                result.get("service_id") == self.service_id
                and result.get("tool_name") in {self.collect_tool, self.cancel_tool, self.status_tool}
            ):
                return result
        return None

    def _has_pending_intake(self, context: AgentContext) -> bool:
        memory = context.session_state
        if memory is None:
            return False
        state = memory.flat_state.get("case_intake")
        if not isinstance(state, dict):
            return False
        if state.get("status") == "collecting":
            return True
        return not bool(state.get("completed")) and bool(state.get("missing_slots"))

    def _should_cancel(self, context: AgentContext) -> bool:
        message = context.request.message.strip()
        return self._has_pending_intake(context) and any(keyword in message for keyword in _CANCEL_KEYWORDS)

    @staticmethod
    def _is_risky_message(message: str) -> bool:
        return any(keyword in message for keyword in ("退款", "退货", "换货", "投诉"))

    @staticmethod
    def _is_risky_intake(ticket_payload: dict[str, str]) -> bool:
        intent = (ticket_payload.get("intent") or "").lower()
        priority = (ticket_payload.get("priority") or "").lower()
        return intent in {"refund", "complaint"} or priority == "high"
