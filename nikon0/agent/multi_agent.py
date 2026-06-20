"""Bounded multi-agent coordination built on existing Skills and ToolRuntime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nikon0.agent.delegation import AgentDelegationPlan, AgentDelegationPlanner
from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import AgentResult, SkillResult
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.product_support import ProductSupportSkill


class SupportAgentProtocol(Protocol):
    async def run(self, context: AgentContext) -> SkillResult:
        ...

    def handoff(self, context: AgentContext, result: SkillResult) -> dict[str, Any]:
        ...


class ServiceAgentProtocol(Protocol):
    async def run(self, context: AgentContext, handoff: dict[str, Any]) -> SkillResult:
        ...


@dataclass(frozen=True)
class MultiAgentOutcome:
    result: AgentResult
    agent_stages: list[str] = field(default_factory=list)
    support_handoff: dict[str, Any] = field(default_factory=dict)
    plans: list[AgentDelegationPlan] = field(default_factory=list)


class SupportAgent:
    name = "support"
    allowed_tools = frozenset(
        {
            "product-support.resolve_product",
            "product-support.search_product_manual",
            "product-support.validate_answer_grounding",
        }
    )

    def __init__(self, skill: ProductSupportSkill) -> None:
        self.skill = skill

    async def run(self, context: AgentContext) -> SkillResult:
        context.allowed_tool_names = set(self.allowed_tools)
        context.selected_agent = self.name
        if self.name not in context.trace.selected_agents:
            context.trace.selected_agents.append(self.name)
        context.selected_skill = "product_support"
        context.trace.add_event("agent.execution_start", "support agent started", agent=self.name)
        result = await self.skill.run(context)
        context.trace.add_event("agent.execution_stop", "support agent completed", agent=self.name, status=result.status)
        return result

    @staticmethod
    def handoff(context: AgentContext, result: SkillResult) -> dict[str, Any]:
        update = next((item for item in result.state_updates if item.key == "product_support"), None)
        state = dict(update.value) if update and isinstance(update.value, dict) else {}
        resolution = state.get("product_resolution") if isinstance(state.get("product_resolution"), dict) else {}
        evidence_ids = [item.evidence_id for item in result.evidence]
        status = "grounded" if result.evidence else ("needs_clarification" if result.status == "needs_more_info" else "insufficient_evidence")
        return {
            "stage": "diagnosis",
            "product_resolution": resolution,
            "diagnosis_status": status,
            "symptoms": [context.request.message],
            "attempted_steps": [],
            "recommended_next_steps": [result.answer_draft] if result.answer_draft else [],
            "safety_risks": [],
            "evidence_ids": evidence_ids,
            "summary": result.answer_draft[:400],
        }


class ServiceAgent:
    name = "service"
    allowed_tools = frozenset(
        {
            "case-intake.extract_case_slots",
            "case-intake.collect_case_intake",
            "case-intake.try_cancel_case_intake",
            "case-intake.get_case_intake_status",
        }
    )

    def __init__(self, skill: CaseIntakeSkill) -> None:
        self.skill = skill

    async def run(self, context: AgentContext, handoff: dict[str, Any]) -> SkillResult:
        context.allowed_tool_names = set(self.allowed_tools)
        context.selected_agent = self.name
        if self.name not in context.trace.selected_agents:
            context.trace.selected_agents.append(self.name)
        context.selected_skill = "case_intake"
        context.agent_handoff = dict(handoff)
        context.trace.add_event("agent.handoff_consumed", "service consumed support handoff", agent=self.name, handoff=handoff)
        context.trace.add_event("agent.execution_start", "service agent started", agent=self.name)
        result = await self.skill.run(context)
        # CaseIntakeSkill intentionally uses a bounded act-observe-act flow.
        for request in result.tool_calls:
            await context.tool_runtime.call_step(context, request)
        if result.tool_calls:
            result = await self.skill.run(context)
        context.trace.add_event("agent.execution_stop", "service agent completed", agent=self.name, status=result.status)
        return result


class MultiAgentCoordinator:
    """Runs at most Support then Service, re-planning from real support output."""

    def __init__(
        self,
        *,
        planner: AgentDelegationPlanner,
        support_agent: SupportAgentProtocol,
        service_agent: ServiceAgentProtocol,
    ) -> None:
        self.planner = planner
        self.support_agent = support_agent
        self.service_agent = service_agent

    async def run(self, context: AgentContext) -> MultiAgentOutcome:
        initial = await self.planner.plan(context, stage="initial")
        context.trace.add_event("agent.delegation_plan", "initial delegation plan", **initial.model_dump())
        if initial.action == "general":
            return self._terminal("success", "你好，我是 nikon0 企业助手。", initial)
        if initial.action == "clarify":
            return self._terminal("needs_more_info", "请补充产品、问题现象或需要办理的服务，我再继续处理。", initial)
        if initial.action == "handoff":
            return self._terminal("handoff_required", "当前请求需要人工处理，已生成转人工请求。", initial, risk_level="high")
        if initial.action == "service":
            service = await self.service_agent.run(context, {})
            return self._from_skill(service, ["service"], {}, [initial])

        support = await self.support_agent.run(context)
        handoff = self.support_agent.handoff(context, support)
        context.trace.add_event("agent.handoff_created", "support handoff created", agent="support", handoff=handoff)
        replan = await self.planner.plan(context, stage="after_support", handoff=handoff)
        context.trace.add_event("agent.replan", "replanned after support result", **replan.model_dump())
        if replan.action == "service":
            service = await self.service_agent.run(context, handoff)
            return self._from_skill(service, ["support", "service"], handoff, [initial, replan])
        if replan.action == "handoff":
            return self._terminal("handoff_required", "当前请求需要人工处理，已生成转人工请求。", replan, handoff=handoff, stages=["support"], plans=[initial, replan], risk_level="high")
        if replan.action == "clarify":
            return self._terminal("needs_more_info", "请补充必要信息后我再继续处理。", replan, handoff=handoff, stages=["support"], plans=[initial, replan])
        return self._from_skill(support, ["support"], handoff, [initial, replan])

    @staticmethod
    def _from_skill(result: SkillResult, stages: list[str], handoff: dict[str, Any], plans: list[AgentDelegationPlan]) -> MultiAgentOutcome:
        return MultiAgentOutcome(
            result=AgentResult(
                status=result.status,
                answer_draft=result.answer_draft,
                evidence=result.evidence,
                tool_calls=[],
                state_updates=result.state_updates,
                risk_level=result.risk_level,
                selected_skills=["product_support" if item == "support" else "case_intake" for item in stages],
                handoff_reason=result.handoff_reason,
            ),
            agent_stages=stages,
            support_handoff=handoff,
            plans=plans,
        )

    @staticmethod
    def _terminal(
        status: str,
        answer: str,
        plan: AgentDelegationPlan,
        *,
        handoff: dict[str, Any] | None = None,
        stages: list[str] | None = None,
        plans: list[AgentDelegationPlan] | None = None,
        risk_level: str = "low",
    ) -> MultiAgentOutcome:
        return MultiAgentOutcome(
            result=AgentResult(status=status, answer_draft=answer, risk_level=risk_level),
            agent_stages=stages or [],
            support_handoff=handoff or {},
            plans=plans or [plan],
        )
