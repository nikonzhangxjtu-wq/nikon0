from __future__ import annotations

import asyncio

from nikon0.agent.delegation import AgentDelegationPlanner
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.trace import ExecutionTrace


class _StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(self, messages) -> str:
        return self.response


def _context(message: str) -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="multi-agent-test", message=message),
        trace=ExecutionTrace(trace_id="trace", session_id="multi-agent-test", user_message=message),
    )


def test_delegation_planner_accepts_model_selected_support() -> None:
    planner = AgentDelegationPlanner(
        _StaticClient('{"action":"support","confidence":0.92,"reason":"manual diagnosis needed"}')
    )

    plan = asyncio.run(planner.plan(_context("洗碗机漏水怎么办")))

    assert plan.action == "support"
    assert plan.source == "llm"
    assert plan.confidence == 0.92


def test_delegation_planner_does_not_fallback_to_keyword_routing_on_invalid_json() -> None:
    planner = AgentDelegationPlanner(_StaticClient("not json"))

    plan = asyncio.run(planner.plan(_context("洗碗机不加热怎么办")))

    assert plan.action == "clarify"
    assert plan.source == "fallback"
    assert "invalid" in plan.reason


def test_delegation_planner_fails_closed_for_high_risk_service_when_model_is_invalid() -> None:
    planner = AgentDelegationPlanner(_StaticClient("{}"))

    plan = asyncio.run(planner.plan(_context("我要退款，订单号 ORD-10001")))

    assert plan.action == "handoff"
    assert plan.source == "fallback"
