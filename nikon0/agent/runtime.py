"""nikon0 AgentRuntime minimal implementation."""

from __future__ import annotations

from uuid import uuid4

from nikon0.agent.base import AgentRegistry
from nikon0.agent.context_governance import ContextGovernance
from nikon0.agent.loop import AgentLoop
from nikon0.agent.planner import RuleBasedPlanner
from nikon0.agent.supervisor import SupervisorAgent
from nikon0.app.schemas.storage import TranscriptEntry
from nikon0.app.schemas.agent import AgentActionRecord, AgentContext, AgentRequest, AgentResponse
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.app.services.approvals import InMemoryApprovalStore, JsonlApprovalStore
from nikon0.app.services.storage import InMemoryTraceRecorder, InMemoryTranscriptStore, JsonlTraceRecorder, JsonlTranscriptStore
from nikon0.llm import BailianOllamaChatClient, LlmAnswerGenerator
from nikon0.context.budgeter import ContextBudgeter
from nikon0.context.conversation import ConversationCompactor
from nikon0.context.evidence import EvidenceContextManager
from nikon0.context.llm_compaction import LlmConversationCompactor
from nikon0.context.llm_span_selector import LlmEvidenceSpanSelector
from nikon0.context.read_planner import DeterministicContextReadPlanner, LlmContextReadPlanner
from nikon0.context.runtime import ContextRuntime
from nikon0.knowledge.runtime import EnterpriseRagBackend, KnowledgeRuntime
from nikon0.memory.governance import IssueThreadLifecycleManager, MemoryReadPlanner, MemoryWriteGate
from nikon0.memory.persistence import build_memory_store_from_env
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.memory.view import MemoryViewBuilder
from nikon0.safety.gate import SafetyGate
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.mock_skill import MockSkill
from nikon0.skills.model_selector import BailianOllamaSkillSelectionClient, LlmSkillSelector
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.skills.tool_echo import ToolEchoSkill
from nikon0.tools.runtime import ToolRuntime


class AgentRuntime:
    """One-turn orchestration entrypoint for nikon0."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        context_governance: ContextGovernance | None = None,
        safety_gate: SafetyGate | None = None,
        memory_store: InMemorySessionIssueStore | None = None,
        tool_runtime: ToolRuntime | None = None,
        trace_recorder: InMemoryTraceRecorder | None = None,
        transcript_store: InMemoryTranscriptStore | None = None,
        approval_store: InMemoryApprovalStore | None = None,
        planner: RuleBasedPlanner | None = None,
        answer_generator: LlmAnswerGenerator | None = None,
        memory_view_builder: MemoryViewBuilder | None = None,
        memory_write_gate: MemoryWriteGate | None = None,
        memory_read_planner: MemoryReadPlanner | None = None,
        max_turns: int = 4,
    ) -> None:
        self.skill_registry = skill_registry or SkillRegistry(_build_default_skills(answer_generator=answer_generator))
        self.answer_generator = answer_generator
        self.agent_registry = agent_registry or AgentRegistry(
            [SupervisorAgent(self.skill_registry, answer_generator=answer_generator)]
        )
        self.context_governance = context_governance or ContextGovernance()
        self.safety_gate = safety_gate or SafetyGate()
        self.memory_store = memory_store or InMemorySessionIssueStore()
        self.tool_runtime = tool_runtime or ToolRuntime()
        self.trace_recorder = trace_recorder or InMemoryTraceRecorder()
        self.transcript_store = transcript_store or InMemoryTranscriptStore()
        self.approval_store = approval_store or InMemoryApprovalStore()
        self.planner = planner or RuleBasedPlanner()
        self.memory_view_builder = memory_view_builder or MemoryViewBuilder()
        self.memory_write_gate = memory_write_gate or MemoryWriteGate()
        self.memory_read_planner = memory_read_planner or MemoryReadPlanner(
            lifecycle=IssueThreadLifecycleManager(),
        )
        self.max_turns = max(1, int(max_turns))

    async def run(self, request: AgentRequest) -> AgentResponse:
        trace = ExecutionTrace(
            trace_id=uuid4().hex,
            session_id=request.session_id,
            user_message=request.message,
        )
        trace.add_event("runtime.start", "received agent request", channel=request.channel)
        transcript_context = self.transcript_store.replay_text(request.session_id)
        self.transcript_store.append(
            TranscriptEntry(
                session_id=request.session_id,
                trace_id=trace.trace_id,
                role="user",
                content=request.message,
                metadata={"channel": request.channel, "image_count": len(request.images)},
            )
        )
        session_state = self.memory_store.load(request.session_id)
        memory_read_plan, thread_decision = await self.memory_read_planner.plan(
            session_state,
            request.message,
            transcript_context,
        )
        memory_view = self.memory_view_builder.build(session_state, read_plan=memory_read_plan)
        trace.add_event(
            "memory.read_plan",
            "selected governed memory read scope",
            **memory_read_plan.model_dump(),
        )
        trace.add_event(
            "memory.thread_decision",
            "selected issue thread lifecycle action",
            **thread_decision.model_dump(),
        )
        context = AgentContext(
            request=request,
            session_state=session_state,
            memory_context=memory_view.render(),
            transcript_context=transcript_context,
            context_governance=self.context_governance,
            tool_runtime=self.tool_runtime,
            available_tools=await self.tool_runtime.list_tools(),
            trace=trace,
        )
        context = await self.context_governance.agovern(context)
        context.plan = self.planner.plan(context)
        trace.add_event(
            "planner.plan",
            "planner produced routing plan",
            intents=[intent.model_dump() for intent in context.plan.intents],
            candidates=[candidate.model_dump() for candidate in context.plan.candidates],
            recommended_skill=context.plan.recommended_skill,
            is_composite=context.plan.is_composite,
            risk_level=context.plan.risk_level,
        )

        loop = AgentLoop(
            agent_registry=self.agent_registry,
            tool_runtime=self.tool_runtime,
            max_turns=self.max_turns,
        )
        loop_result = await loop.run(context)
        result = loop_result.result
        safety = await self.safety_gate.check(context, result)
        trace.final_risk_level = safety.risk_level

        if not safety.allowed:
            if safety.approval_request is not None:
                answer = "当前请求涉及高风险服务动作，已生成审批请求，审批通过前不会自动执行或承诺结果。"
                status = "approval_required"
            elif safety.handoff_request is not None:
                answer = "当前请求需要人工处理，已生成转人工请求。"
                status = "handoff_required"
            else:
                answer = "当前请求触发安全限制，暂时无法继续处理。"
                status = "blocked"
        else:
            answer = result.answer_draft or "已完成处理。"
            status = result.status

        candidates = self.memory_write_gate.adapt_updates(
            result.state_updates,
            risk_level=result.risk_level,
            selected_skill=context.selected_skill,
        )
        for candidate in candidates:
            candidate.target_thread_id = thread_decision.thread_id
            candidate.create_thread = thread_decision.action == "create_thread"

        gate_enabled = _memory_write_gate_enabled()
        if gate_enabled:
            decisions = self.memory_write_gate.validate(session_state, candidates)
        else:
            decisions = [self._legacy_memory_decision(candidate) for candidate in candidates]

        for decision in decisions:
            trace.add_event(
                "memory.write_validate",
                decision.reason,
                **decision.model_dump(mode="json"),
            )
            if decision.outcome == "reject":
                trace.add_event(
                    "memory.write_rejected",
                    decision.reason,
                    **decision.model_dump(mode="json"),
                )
            elif decision.outcome == "needs_confirmation":
                trace.add_event(
                    "memory.write_conflict",
                    decision.reason,
                    **decision.model_dump(mode="json"),
                )

        accepted_updates = [
            decision.update
            for decision in decisions
            if decision.outcome == "accept" and decision.update is not None
        ]
        confirmation = next(
            (decision for decision in decisions if decision.outcome == "needs_confirmation"),
            None,
        )
        memory_degraded = False
        try:
            updated_state = self.memory_store.apply_updates(
                request.session_id,
                accepted_updates,
                turn_id=trace.trace_id,
                target_thread_id=thread_decision.thread_id,
                create_thread=thread_decision.action == "create_thread",
            )
        except Exception as exc:  # noqa: BLE001
            if result.risk_level in {"high", "medium"}:
                answer = "当前服务状态无法可靠保存，已转人工处理，请勿重复提交。"
                status = "handoff_required"
                updated_state = session_state
                trace.add_event(
                    "memory.persistence.blocked",
                    "blocked high-risk memory write",
                    error_type=type(exc).__name__,
                )
            else:
                memory_degraded = True
                updated_state = InMemorySessionIssueStore().apply_updates(
                    request.session_id,
                    accepted_updates,
                    turn_id=trace.trace_id,
                    target_thread_id=thread_decision.thread_id,
                    create_thread=thread_decision.action == "create_thread",
                )
                trace.add_event(
                    "memory.persistence.degraded",
                    "low-risk write used ephemeral memory",
                    error_type=type(exc).__name__,
                )

        if confirmation is not None:
            conflict = confirmation.conflicts[0] if confirmation.conflicts else None
            field = conflict.field if conflict else "关键信息"
            answer = f"我发现本轮信息与之前记录的{field}不一致。请确认本次要使用的正确信息后，我再继续处理。"
            status = "needs_more_info"

        for update in accepted_updates:
            trace.memory_updates.append(update.model_dump())

        self._persist_memory_audit(
            request.session_id,
            decisions,
            thread_decision.model_dump(),
            turn_id=trace.trace_id,
        )
        trace.add_event(
            "memory.update",
            "applied governed session issue updates",
            candidate_count=len(candidates),
            accepted_count=len(accepted_updates),
            confirmation_count=sum(1 for item in decisions if item.outcome == "needs_confirmation"),
            rejected_count=sum(1 for item in decisions if item.outcome == "reject"),
            degraded=memory_degraded,
        )
        trace.add_event("runtime.stop", "completed agent request", status=status)
        self.transcript_store.append(
            TranscriptEntry(
                session_id=request.session_id,
                trace_id=trace.trace_id,
                role="assistant",
                content=answer,
                metadata={"risk_level": safety.risk_level, "status": status},
            )
        )
        stored_trace = self.trace_recorder.record(trace)

        actions = [
            AgentActionRecord(
                kind="agent",
                name=context.selected_agent or (trace.selected_agents[-1] if trace.selected_agents else "unknown"),
                status=status,
                detail=f"loop_stop={loop_result.stop_reason}",
            )
        ]
        actions.extend(
            AgentActionRecord(kind="skill", name=skill_name, status=status)
            for skill_name in result.selected_skills
        )
        if safety.approval_request is not None:
            self.approval_store.create_approval(safety.approval_request)
            actions.append(
                AgentActionRecord(
                    kind="approval",
                    name=safety.approval_request.approval_id,
                    status=safety.approval_request.status,
                    detail=safety.approval_request.reason,
                    payload=safety.approval_request.model_dump(),
                )
            )
        if safety.handoff_request is not None:
            self.approval_store.create_handoff(safety.handoff_request)
            actions.append(
                AgentActionRecord(
                    kind="handoff",
                    name=safety.handoff_request.handoff_id,
                    status="pending",
                    detail=safety.handoff_request.reason,
                    payload=safety.handoff_request.model_dump(),
                )
            )
        actions.extend(
            AgentActionRecord(
                kind="tool",
                name=f"{tool_result['service_id']}.{tool_result['tool_name']}",
                status="success" if tool_result["ok"] else "failed",
                detail=tool_result.get("error_message") or "",
                payload=tool_result.get("data") or {},
            )
            for tool_result in context.tool_results
        )
        return AgentResponse(
            answer=answer,
            state_summary=(
                f"active_thread_id={updated_state.active_thread_id or ''};"
                f"flat_state_keys={','.join(sorted(updated_state.flat_state.keys()))}"
            ),
            risk_level=safety.risk_level,
            trace_id=trace.trace_id,
            actions=actions,
            debug={
                "trace": trace.model_dump(),
                "context_debug": _context_debug_payload(context),
                "trace_persisted": stored_trace.trace_id,
                "transcript_entries": len(self.transcript_store.list_for_session(request.session_id)),
                "loop": {
                    "turn_count": loop_result.turn_count,
                    "stop_reason": loop_result.stop_reason,
                    "steps": [step.__dict__ for step in loop_result.steps],
                },
                "plan": context.plan.model_dump() if context.plan else None,
                "skill_selection": context.skill_selection.model_dump() if context.skill_selection else None,
                "skill_manifests": [manifest.model_dump() for manifest in self.skill_registry.manifests()],
                "memory_governance": {
                    "read_plan": memory_read_plan.model_dump(),
                    "thread_decision": thread_decision.model_dump(),
                    "write_gate_enabled": gate_enabled,
                    "write_decisions": [item.model_dump(mode="json") for item in decisions],
                    "degraded": memory_degraded,
                    "store_profile": _memory_store_profile(self.memory_store),
                },
            },
        )

    def _legacy_memory_decision(self, candidate):
        from nikon0.memory.governance.types import MemoryWriteDecision

        return MemoryWriteDecision(
            candidate_id=candidate.candidate_id,
            outcome="accept",
            target_thread_id=candidate.target_thread_id,
            reason="memory write gate disabled",
            update=candidate.update,
        )

    def _persist_memory_audit(self, session_id, decisions, thread_event, *, turn_id: str = "") -> None:
        sql = getattr(self.memory_store, "sql_persistence", None)
        if sql is None:
            return
        try:
            sql.append_write_decisions(
                session_id,
                [item.model_dump(mode="json") for item in decisions],
                turn_id=turn_id,
            )
            sql.append_thread_event(session_id, thread_event, turn_id=turn_id)
        except Exception:  # noqa: BLE001
            return


def _context_debug_payload(context: AgentContext) -> dict:
    pack = context.context_pack
    if pack is None:
        return {
            "rendered_chars": len(context.governed_context or ""),
            "section_count": 0,
            "sections": [],
            "budget_report": {},
            "governed_context_preview": (context.governed_context or "")[:1000],
        }
    sections = []
    for section in pack.sections:
        sections.append(
            {
                "name": section.name,
                "source": section.source,
                "priority": section.priority,
                "chars": len(section.content),
                "token_estimate": section.token_estimate,
                "char_budget": section.char_budget,
                "truncated": section.truncated,
                "metadata": dict(section.metadata),
                "content": section.content,
                "preview": section.content[:240],
            }
        )
    return {
        "rendered_chars": len(context.governed_context or ""),
        "section_count": len(sections),
        "sections": sections,
        "section_names": [section["name"] for section in sections],
        "budget_report": pack.budget_report.model_dump(),
        "governed_context_preview": (context.governed_context or "")[:1000],
    }


def build_default_runtime() -> AgentRuntime:
    answer_generator = _build_default_answer_generator()
    skill_registry = _build_default_skill_registry(answer_generator=answer_generator)
    return AgentRuntime(
        skill_registry=skill_registry,
        answer_generator=answer_generator,
        context_governance=_build_default_context_governance(),
        memory_store=build_memory_store_from_env(),
        trace_recorder=JsonlTraceRecorder.default(),
        transcript_store=JsonlTranscriptStore.default(),
        approval_store=JsonlApprovalStore.default(),
        memory_read_planner=_build_default_memory_read_planner(),
    )


def _memory_write_gate_enabled() -> bool:
    try:
        from app.core.config import settings

        return bool(getattr(settings, "nikon0_memory_write_gate_enabled", True))
    except Exception:  # noqa: BLE001
        return True


def _memory_store_profile(store) -> dict:
    if hasattr(store, "profile"):
        try:
            return dict(store.profile())
        except Exception as exc:  # noqa: BLE001
            return {
                "store_type": type(store).__name__,
                "healthy": False,
                "error_type": type(exc).__name__,
            }
    return {
        "store_type": type(store).__name__,
        "redis_ok": False,
        "mysql_ok": False,
        "degraded": True,
    }


def _build_default_memory_read_planner() -> MemoryReadPlanner:
    try:
        from app.core.config import settings

        enabled = bool(getattr(settings, "nikon0_memory_llm_planner_enabled", True))
        model = (
            getattr(settings, "nikon0_memory_llm_planner_model", "")
            or getattr(settings, "simple_llm_model", "")
            or getattr(settings, "gen_model", "")
        )
        if enabled and model:
            client = BailianOllamaChatClient(
                model=model,
                temperature=0.0,
                max_tokens=int(getattr(settings, "nikon0_memory_llm_planner_max_tokens", 512) or 512),
                timeout=int(getattr(settings, "nikon0_memory_llm_planner_timeout", 12) or 12),
            )
            return MemoryReadPlanner(client=client)
    except Exception:  # noqa: BLE001
        pass
    return MemoryReadPlanner()


def _build_default_context_governance(*, settings=None) -> ContextGovernance:
    if settings is None:
        try:
            from app.core.config import settings as loaded_settings
        except Exception:  # noqa: BLE001
            loaded_settings = None
        settings = loaded_settings
    total_budget = int(getattr(settings, "nikon0_context_total_char_budget", 9000) or 9000)
    llm_enabled = bool(getattr(settings, "nikon0_context_llm_enabled", True))
    model = (
        getattr(settings, "nikon0_context_llm_model", "")
        or getattr(settings, "simple_llm_model", "")
        or getattr(settings, "gen_model", "")
    )
    budgeter = ContextBudgeter(total_char_budget=total_budget)
    if not llm_enabled or not model:
        return ContextGovernance(
            context_runtime=ContextRuntime(
                budgeter=budgeter,
                read_planner=DeterministicContextReadPlanner(),
                conversation_compactor=ConversationCompactor(),
            )
        )
    timeout = int(getattr(settings, "nikon0_context_llm_timeout", 15) or 15)
    max_tokens = int(getattr(settings, "nikon0_context_llm_max_tokens", 512) or 512)
    client = BailianOllamaChatClient(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    span_selector = LlmEvidenceSpanSelector(client)
    return ContextGovernance(
        context_runtime=ContextRuntime(
            budgeter=budgeter,
            read_planner=LlmContextReadPlanner(client),
            conversation_compactor=LlmConversationCompactor(client),
            evidence_manager=EvidenceContextManager(span_selector=span_selector),
        )
    )


def _build_default_skill_registry(answer_generator: LlmAnswerGenerator | None = None) -> SkillRegistry:
    skills = _build_default_skills(answer_generator=answer_generator)
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return SkillRegistry(skills)

    if not bool(getattr(settings, "router_llm_enabled", False)):
        return SkillRegistry(skills)
    model = (
        getattr(settings, "router_llm_model", "")
        or getattr(settings, "simple_llm_model", "")
        or getattr(settings, "gen_model", "")
    )
    if not model:
        return SkillRegistry(skills)
    selector = LlmSkillSelector(
        BailianOllamaSkillSelectionClient(
            model=model,
            temperature=0.0,
            max_tokens=256,
            timeout=15,
        )
    )
    return SkillRegistry(skills, selector=selector)


def _build_default_skills(answer_generator: LlmAnswerGenerator | None = None) -> list:
    skills = [
        ToolEchoSkill(),
        CaseIntakeSkill(),
        ProductSupportSkill(
            knowledge_runtime=_build_default_product_knowledge_runtime(),
            answer_generator=answer_generator,
        ),
    ]
    if _mock_skill_enabled():
        skills.append(MockSkill())
    return skills


def _mock_skill_enabled(*, settings=None) -> bool:
    if settings is None:
        try:
            from app.core.config import settings as loaded_settings
        except Exception:  # noqa: BLE001
            loaded_settings = None
        settings = loaded_settings
    return bool(getattr(settings, "nikon0_enable_mock_skill", False))


def _build_default_answer_generator() -> LlmAnswerGenerator | None:
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return None

    model = (
        getattr(settings, "simple_llm_model", "")
        or getattr(settings, "gen_model", "")
        or getattr(settings, "router_llm_model", "")
    )
    if not model:
        return None
    return LlmAnswerGenerator(
        BailianOllamaChatClient(
            model=model,
            temperature=float(getattr(settings, "gen_temperature_competition", 0.1) or 0.1),
            max_tokens=int(getattr(settings, "gen_max_tokens", 1024) or 1024),
            timeout=30,
        )
    )


def _build_default_product_knowledge_runtime() -> KnowledgeRuntime:
    return KnowledgeRuntime(EnterpriseRagBackend())
