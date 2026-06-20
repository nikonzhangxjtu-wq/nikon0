"""Supervisor agent for the first nikon0 runtime loop."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import AgentMatch, AgentResult, SkillResult, StateUpdate
from nikon0.llm.generation import LlmAnswerGenerator
from nikon0.skills.base import SkillRegistry


class SupervisorAgent:
    name = "supervisor"
    description = "Coordinates skills and composes the first runtime result."

    DEFAULT_SKILL_CONFIDENCE_THRESHOLD = 0.75
    SKILL_CONFIDENCE_THRESHOLDS: dict[str, float] = {
        # product_support is backed by RAG + rule signals; allow model/planner/fallback at 0.55+
        "product_support": 0.55,
    }

    def __init__(
        self,
        skill_registry: SkillRegistry,
        *,
        skill_confidence_threshold: float = DEFAULT_SKILL_CONFIDENCE_THRESHOLD,
        skill_confidence_thresholds: dict[str, float] | None = None,
        answer_generator: LlmAnswerGenerator | None = None,
    ) -> None:
        self.skill_registry = skill_registry
        self.skill_confidence_threshold = skill_confidence_threshold
        self.skill_confidence_thresholds = {
            **self.SKILL_CONFIDENCE_THRESHOLDS,
            **(skill_confidence_thresholds or {}),
        }
        self.answer_generator = answer_generator

    def _confidence_threshold_for(self, skill_name: str | None) -> float:
        if skill_name:
            return self.skill_confidence_thresholds.get(skill_name, self.skill_confidence_threshold)
        return self.skill_confidence_threshold

    async def can_handle(self, context: AgentContext) -> AgentMatch:
        return AgentMatch(
            matched=True,
            confidence=1.0,
            reason="supervisor is the default orchestrator",
        )

    async def run(self, context: AgentContext) -> AgentResult:
        skill, match, selection = await self.skill_registry.select_best(context)
        threshold = self._confidence_threshold_for(skill.name if skill is not None else None)
        if skill is None or match.confidence < threshold:
            context.selected_skill = None
            if skill is not None:
                context.skill_selection = selection.model_copy(
                    update={
                        "selected_skill": None,
                        "source": "none",
                        "reason": (
                            f"no high-confidence skill selected; best={selection.selected_skill or 'none'} "
                            f"confidence={match.confidence}"
                        ),
                        "confidence": match.confidence,
                    }
                )
            else:
                context.skill_selection = selection
            context.trace.add_event(
                "agent.general_handle",
                "no high-confidence skill selected",
                best_confidence=match.confidence,
                reason=match.reason,
                selection=context.skill_selection.model_dump(),
            )
            return await self._general_handle(context)

        context.selected_skill = skill.name
        context.retry_tool_errors = bool(skill.manifest.fallback_policy.retry_on_tool_error)
        if skill.name not in context.trace.selected_skills:
            context.trace.selected_skills.append(skill.name)
        context.trace.add_event(
            "skill.select",
            f"selected {skill.name}",
            confidence=match.confidence,
            reason=match.reason,
            selection_source=selection.source,
        )
        try:
            result = await skill.run(context)
        except Exception as exc:  # noqa: BLE001
            context.trace.add_event(
                "skill.exception",
                f"{skill.name} raised {type(exc).__name__}",
                skill=skill.name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            result = SkillResult(
                status="failed",
                answer_draft=f"{skill.name} 暂时不可用，请稍后再试或转人工处理。",
                risk_level=getattr(skill, "risk_level", "low"),
                handoff_reason=f"{skill.name} failed with {type(exc).__name__}",
            )
        result = self._apply_fallback_policy(context, skill, result)
        if selection.source == "sticky":
            result.state_updates.append(self._sticky_turn_update(context, skill.name))
        context.trace.add_event(
            "skill.run",
            f"{skill.name} returned {result.status}",
            risk_level=result.risk_level,
            evidence_count=len(result.evidence),
            state_update_count=len(result.state_updates),
        )
        return AgentResult(
            status=result.status,
            answer_draft=result.answer_draft,
            evidence=result.evidence,
            tool_calls=result.tool_calls,
            state_updates=result.state_updates,
            risk_level=result.risk_level,
            selected_skills=[skill.name],
            handoff_reason=result.handoff_reason,
        )

    def _apply_fallback_policy(self, context: AgentContext, skill, result: SkillResult) -> SkillResult:
        if result.status != "failed":
            return result
        policy = skill.manifest.fallback_policy
        context.trace.add_event(
            "skill.fallback",
            f"applying fallback policy for {skill.name}",
            skill=skill.name,
            allow_general_fallback=policy.allow_general_fallback,
            allow_handoff=policy.allow_handoff,
            retry_on_tool_error=policy.retry_on_tool_error,
            original_status=result.status,
        )
        if policy.allow_handoff:
            return result.model_copy(
                update={
                    "status": "handoff_required",
                    "answer_draft": result.answer_draft or "当前服务暂时不可用，已为你转人工处理。",
                    "handoff_reason": result.handoff_reason or f"{skill.name} failed and requires handoff",
                }
            )
        if policy.allow_general_fallback:
            return result.model_copy(
                update={
                    "status": "success",
                    "answer_draft": result.answer_draft or "当前能力暂时不可用，请稍后再试或换一种说法。",
                }
            )
        return result

    @staticmethod
    def _sticky_turn_update(context: AgentContext, skill_name: str) -> StateUpdate:
        current: dict[str, int] = {}
        if context.session_state is not None:
            raw = context.session_state.flat_state.get("_sticky_turns")
            if isinstance(raw, dict):
                for key, value in raw.items():
                    try:
                        current[str(key)] = max(0, int(value))
                    except (TypeError, ValueError):
                        current[str(key)] = 0
        current[skill_name] = current.get(skill_name, 0) + 1
        return StateUpdate(
            key="_sticky_turns",
            value=current,
            reason=f"record sticky continuation for {skill_name}",
        )

    async def _general_handle(self, context: AgentContext) -> AgentResult:
        if context.tool_results:
            return AgentResult(
                status="success",
                answer_draft="已根据工具结果完成处理。",
                risk_level="low",
                selected_skills=[],
            )
        fallback_answer = (
            "nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，"
            "因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。"
        )
        answer = fallback_answer
        if self.answer_generator is not None:
            answer = await self.answer_generator.general_answer(
                context=context,
                fallback_answer=fallback_answer,
            )
        return AgentResult(
            status="success",
            answer_draft=answer,
            risk_level="low",
            selected_skills=[],
        )
