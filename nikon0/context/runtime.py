"""Context pack assembler and budgeter."""

from __future__ import annotations

import json
from typing import Any

from nikon0.app.schemas.agent import AgentContext
from nikon0.context.budgeter import ContextBudgeter
from nikon0.context.conversation import ConversationCompactor
from nikon0.context.evidence import EvidenceContextManager
from nikon0.context.pack import ContextPack, ContextSection
from nikon0.context.read_planner import ContextReadPlan, DeterministicContextReadPlanner
from nikon0.context.tool_observation import ToolObservationManager


DEFAULT_SECTION_BUDGETS = {
    "system_policy": 800,
    "workflow": 700,
    "memory": 1400,
    "conversation": 1800,
    "tool_observations": 1200,
    "evidence": 2400,
    "current_user": 800,
    "runtime": 400,
}


class ContextRuntime:
    """Build a governed context pack before model calls.

    Phase 1 is deterministic: assemble known runtime state, normalize large
    tool results, and enforce section budgets. LLM planning/compaction can be
    layered on top without changing prompt call sites.
    """

    def __init__(
        self,
        *,
        total_char_budget: int = 9000,
        section_budgets: dict[str, int] | None = None,
        budgeter: ContextBudgeter | None = None,
        read_planner: object | None = None,
        conversation_compactor: ConversationCompactor | None = None,
        evidence_manager: EvidenceContextManager | None = None,
        tool_observation_manager: ToolObservationManager | None = None,
    ) -> None:
        self.total_char_budget = max(100, int(total_char_budget))
        self.section_budgets = {**DEFAULT_SECTION_BUDGETS, **(section_budgets or {})}
        self.budgeter = budgeter or ContextBudgeter(
            total_char_budget=self.total_char_budget,
            section_budgets=self.section_budgets,
        )
        self.read_planner = read_planner or DeterministicContextReadPlanner()
        self.conversation_compactor = conversation_compactor or ConversationCompactor()
        self.evidence_manager = evidence_manager or EvidenceContextManager()
        self.tool_observation_manager = tool_observation_manager or ToolObservationManager()

    def build_pack(self, context: AgentContext) -> ContextPack:
        plan = self._sync_read_plan(context)
        return self._build_pack_with_plan(context, plan)

    async def build_pack_async(self, context: AgentContext) -> ContextPack:
        planner = self.read_planner
        if hasattr(planner, "aplan"):
            plan = await planner.aplan(context)  # type: ignore[attr-defined]
        else:
            plan = self._sync_read_plan(context)
        return await self._build_pack_with_plan_async(context, plan)

    def _sync_read_plan(self, context: AgentContext) -> ContextReadPlan:
        planner = self.read_planner
        if hasattr(planner, "plan"):
            return planner.plan(context)  # type: ignore[attr-defined]
        return DeterministicContextReadPlanner().plan(context)

    def _build_pack_with_plan(self, context: AgentContext, plan: ContextReadPlan) -> ContextPack:
        included = set(plan.included_sections)
        context.trace.add_event(
            "context.read_plan",
            "planned context sections",
            included_sections=plan.included_sections,
            reasons=plan.reasons,
            source=plan.source,
            confidence=plan.confidence,
        )
        sections = [
            self._section("system_policy", self._system_policy(), priority=10, source="runtime"),
            self._section("workflow", self._workflow_snapshot(context), priority=20, source="trace"),
            self._section("memory", context.memory_context.strip(), priority=30, source="memory"),
            self._section("conversation", self._conversation_context(context), priority=50, source="transcript"),
            self._section("tool_observations", self._tool_observations(context), priority=40, source="tool_runtime"),
            self._section("evidence", self._evidence_context(context), priority=35, source="knowledge_runtime"),
            self._section("current_user", context.request.message.strip(), priority=5, source="request"),
            self._section("runtime", self._runtime_context(context), priority=60, source="runtime"),
        ]
        return self.budgeter.apply([
            section
            for section in sections
            if section.name in included and section.content.strip()
        ])

    async def _build_pack_with_plan_async(self, context: AgentContext, plan: ContextReadPlan) -> ContextPack:
        included = set(plan.included_sections)
        context.trace.add_event(
            "context.read_plan",
            "planned context sections",
            included_sections=plan.included_sections,
            reasons=plan.reasons,
            source=plan.source,
            confidence=plan.confidence,
        )
        sections = [
            self._section("system_policy", self._system_policy(), priority=10, source="runtime"),
            self._section("workflow", self._workflow_snapshot(context), priority=20, source="trace"),
            self._section("memory", context.memory_context.strip(), priority=30, source="memory"),
            self._section("conversation", await self._conversation_context_async(context), priority=50, source="transcript"),
            self._section("tool_observations", self._tool_observations(context), priority=40, source="tool_runtime"),
            self._section("evidence", await self._evidence_context_async(context), priority=35, source="knowledge_runtime"),
            self._section("current_user", context.request.message.strip(), priority=5, source="request"),
            self._section("runtime", self._runtime_context(context), priority=60, source="runtime"),
        ]
        return self.budgeter.apply([
            section
            for section in sections
            if section.name in included and section.content.strip()
        ])

    def _section(self, name: str, content: str, *, priority: int, source: str) -> ContextSection:
        budget = self.section_budgets.get(name)
        return ContextSection(
            name=name,
            content=content,
            priority=priority,
            source=source,
            char_budget=budget,
            token_estimate=_estimate_tokens(content),
        )

    @staticmethod
    def _system_policy() -> str:
        return (
            "你是 nikon0 企业助手。必须基于可见上下文和工具证据回答；"
            "不能编造业务动作、订单状态、退款、维修结论或手册内容。"
        )

    @staticmethod
    def _workflow_snapshot(context: AgentContext) -> str:
        for event in reversed(context.trace.events):
            if event.stage == "workflow.decision":
                return _json_dumps(event.payload)
        return ""

    def _conversation_context(self, context: AgentContext) -> str:
        active_issue_summary = ""
        if context.session_state is not None:
            thread = context.session_state.active_thread()
            if thread is not None:
                active_issue_summary = thread.summary or thread.user_goal or thread.issue_type
        compacted = self.conversation_compactor.compact(
            context.transcript_context,
            active_issue_summary=active_issue_summary,
        )
        if compacted.compacted:
            context.trace.add_event(
                "context.conversation_compact",
                "compacted transcript for prompt context",
                original_chars=compacted.original_chars,
                rendered_chars=compacted.rendered_chars,
                recent_line_count=len(compacted.raw_recent_lines),
                summary_line_count=len(compacted.summary_lines),
            )
        return compacted.render()

    async def _conversation_context_async(self, context: AgentContext) -> str:
        active_issue_summary = self._active_issue_summary(context)
        if hasattr(self.conversation_compactor, "acompact"):
            compacted = await self.conversation_compactor.acompact(  # type: ignore[attr-defined]
                context.transcript_context,
                active_issue_summary=active_issue_summary,
            )
        else:
            compacted = self.conversation_compactor.compact(
                context.transcript_context,
                active_issue_summary=active_issue_summary,
            )
        if compacted.compacted:
            context.trace.add_event(
                "context.conversation_compact",
                "compacted transcript for prompt context",
                original_chars=compacted.original_chars,
                rendered_chars=compacted.rendered_chars,
                recent_line_count=len(compacted.raw_recent_lines),
                summary_line_count=len(compacted.summary_lines),
                source="llm" if hasattr(self.conversation_compactor, "acompact") else "deterministic",
            )
        return compacted.render()

    @staticmethod
    def _active_issue_summary(context: AgentContext) -> str:
        if context.session_state is not None:
            thread = context.session_state.active_thread()
            if thread is not None:
                return thread.summary or thread.user_goal or thread.issue_type
        return ""

    def _tool_observations(self, context: AgentContext) -> str:
        if not context.tool_results:
            return ""
        pack = self.tool_observation_manager.build(
            context.tool_results,
            trace_id=context.trace.trace_id,
        )
        context.trace.add_event(
            "context.tool_observations",
            "built prompt tool observations",
            item_count=len(pack.items),
            raw_result_refs=[item.raw_result_ref for item in pack.items],
        )
        return pack.render_json()

    def _evidence_context(self, context: AgentContext) -> str:
        if context.evidence_context:
            pack = self.evidence_manager.build(
                query=context.request.message,
                evidence=context.evidence_context,
            )
            context.trace.add_event(
                "context.evidence_pack",
                "built prompt evidence pack",
                **pack.usage,
                item_count=len(pack.items),
                source="deterministic",
            )
            return pack.render_json()
        if not context.trace.knowledge_calls:
            return ""
        calls = []
        for call in context.trace.knowledge_calls[-3:]:
            calls.append(
                {
                    "query": call.get("query"),
                    "intent": call.get("intent"),
                    "evidence_count": call.get("evidence_count"),
                    "product_resolution": call.get("product_resolution"),
                    "backend_trace": call.get("backend_trace"),
                }
            )
        return _json_dumps(calls)

    async def _evidence_context_async(self, context: AgentContext) -> str:
        if context.evidence_context:
            if hasattr(self.evidence_manager, "abuild"):
                pack = await self.evidence_manager.abuild(
                    query=context.request.message,
                    evidence=context.evidence_context,
                )
            else:
                pack = self.evidence_manager.build(
                    query=context.request.message,
                    evidence=context.evidence_context,
                )
            context.trace.add_event(
                "context.evidence_pack",
                "built prompt evidence pack",
                **pack.usage,
                item_count=len(pack.items),
                source="llm_span_selector" if hasattr(self.evidence_manager, "span_selector") and getattr(self.evidence_manager, "span_selector", None) is not None else "deterministic",
            )
            return pack.render_json()
        return self._evidence_context(context)

    @staticmethod
    def _runtime_context(context: AgentContext) -> str:
        payload = {
            "channel": context.request.channel,
            "image_count": len(context.request.images),
            "available_tool_count": len(context.available_tools),
            "selected_skill": context.selected_skill,
            "loop_turn": context.loop_turn,
        }
        return _json_dumps(payload)

def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _estimate_tokens(content: str) -> int:
    return max(1, len(content) // 4) if content else 0
