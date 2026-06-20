from __future__ import annotations

import asyncio

from nikon0.agent.multi_agent import MultiAgentOutcome
from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.app.schemas.capability import AgentResult
from nikon0.memory.write_agent import MemoryWriteAgent


class _Coordinator:
    async def run(self, context):
        context.trace.add_event("agent.delegation_plan", "test plan", action="support")
        return MultiAgentOutcome(
            result=AgentResult(
                status="success",
                answer_draft="由 SupportAgent 处理完成",
                selected_skills=["product_support"],
            ),
            agent_stages=["support"],
        )


class _HighRiskServiceCoordinator:
    async def run(self, context):
        return MultiAgentOutcome(
            result=AgentResult(
                status="success",
                answer_draft="退款处理中",
                risk_level="high",
                selected_skills=["case_intake"],
            ),
            agent_stages=["service"],
        )


class _InvalidMemoryClient:
    async def complete(self, messages) -> str:
        return "not json"


def test_runtime_uses_injected_multi_agent_coordinator_when_enabled() -> None:
    runtime = AgentRuntime(multi_agent_coordinator=_Coordinator(), multi_agent_enabled=True)

    response = asyncio.run(runtime.run(AgentRequest(session_id="multi-runtime", message="洗碗机怎么清洁？")))

    assert response.answer == "由 SupportAgent 处理完成"
    assert response.debug["multi_agent"]["enabled"] is True
    assert response.debug["multi_agent"]["agent_stages"] == ["support"]


def test_runtime_keeps_legacy_loop_when_multi_agent_is_disabled() -> None:
    runtime = AgentRuntime(multi_agent_coordinator=_Coordinator(), multi_agent_enabled=False)

    response = asyncio.run(runtime.run(AgentRequest(session_id="legacy-runtime", message="你好")))

    assert response.debug["multi_agent"]["enabled"] is False


def test_high_risk_service_is_blocked_when_memory_write_agent_fails() -> None:
    runtime = AgentRuntime(
        multi_agent_coordinator=_HighRiskServiceCoordinator(),
        multi_agent_enabled=True,
        memory_write_agent=MemoryWriteAgent(_InvalidMemoryClient()),
        memory_write_agent_enabled=True,
    )

    response = asyncio.run(runtime.run(AgentRequest(session_id="multi-runtime-risk", message="我要退款")))

    assert response.risk_level == "high"
    assert response.answer.startswith("当前服务状态无法可靠保存")
    assert response.debug["memory_write_agent"]["blocked"] is True
