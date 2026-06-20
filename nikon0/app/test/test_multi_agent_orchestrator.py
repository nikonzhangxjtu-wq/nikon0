from __future__ import annotations

import asyncio

from nikon0.agent.delegation import AgentDelegationPlan
from nikon0.agent.multi_agent import MultiAgentCoordinator
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import SkillResult
from nikon0.app.schemas.trace import ExecutionTrace


class _Planner:
    def __init__(self, *plans: AgentDelegationPlan) -> None:
        self.plans = list(plans)
        self.handoffs: list[dict] = []

    async def plan(self, context, *, stage="initial", handoff=None):
        self.handoffs.append(handoff or {})
        return self.plans.pop(0)


class _Support:
    name = "support"

    async def run(self, context):
        return SkillResult(status="success", answer_draft="已完成诊断")

    def handoff(self, context, result):
        return {
            "diagnosis_status": "grounded",
            "product_resolution": {"status": "resolved", "product_id": "dishwasher"},
            "evidence_ids": ["manual:dishwasher:1"],
            "summary": "洗碗机漏水，门封条已更换",
        }


class _Service:
    name = "service"

    async def run(self, context, handoff):
        assert handoff["product_resolution"]["product_id"] == "dishwasher"
        return SkillResult(status="needs_more_info", answer_draft="请提供订单号")


def _context() -> AgentContext:
    request = AgentRequest(session_id="multi-orchestrator", message="洗碗机漏水，我要退款")
    return AgentContext(
        request=request,
        trace=ExecutionTrace(trace_id="trace", session_id=request.session_id, user_message=request.message),
    )


def test_coordinator_replans_after_support_and_then_runs_service() -> None:
    planner = _Planner(
        AgentDelegationPlan(action="support", confidence=0.9, reason="diagnose", source="llm"),
        AgentDelegationPlan(action="service", confidence=0.9, reason="refund workflow", source="llm"),
    )
    coordinator = MultiAgentCoordinator(planner=planner, support_agent=_Support(), service_agent=_Service())

    outcome = asyncio.run(coordinator.run(_context()))

    assert outcome.agent_stages == ["support", "service"]
    assert outcome.result.answer_draft == "请提供订单号"
    assert planner.handoffs[1]["diagnosis_status"] == "grounded"


def test_coordinator_stops_after_support_when_replan_requires_clarification() -> None:
    planner = _Planner(
        AgentDelegationPlan(action="support", confidence=0.9, reason="diagnose", source="llm"),
        AgentDelegationPlan(action="clarify", confidence=0.9, reason="need model", source="llm"),
    )
    coordinator = MultiAgentCoordinator(planner=planner, support_agent=_Support(), service_agent=_Service())

    outcome = asyncio.run(coordinator.run(_context()))

    assert outcome.agent_stages == ["support"]
    assert outcome.result.status == "needs_more_info"
