"""Skill protocol and registry."""

from __future__ import annotations

from typing import Protocol

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import (
    FallbackPolicy,
    RejectedSkill,
    SkillCandidate,
    SkillManifest,
    SkillMatch,
    SkillResult,
    SkillSelection,
    StickyPolicy,
)


class Skill(Protocol):
    name: str
    description: str
    risk_level: str
    manifest: SkillManifest

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        ...

    async def run(self, context: AgentContext) -> SkillResult:
        ...


class ManifestDrivenSkillSelector:
    """Replaceable model-facing selector interface.

    Production implementations can call an LLM. Tests and local runtimes can
    return deterministic structured decisions through build_selection().
    """

    async def select(self, context: AgentContext, manifests: tuple[SkillManifest, ...]) -> SkillSelection:
        _ = (context, manifests)
        return SkillSelection(source="none", reason="no model selector configured")

    def build_selection(
        self,
        *,
        selected_skill: str | None,
        reason: str,
        confidence: float,
        manifests: tuple[SkillManifest, ...],
    ) -> SkillSelection:
        candidates = [
            SkillCandidate(
                name=manifest.name,
                matched=manifest.name == selected_skill,
                confidence=confidence if manifest.name == selected_skill else 0.0,
                reason=reason if manifest.name == selected_skill else "not selected by model",
                source="model",
            )
            for manifest in manifests
        ]
        rejected = [
            RejectedSkill(name=item.name, reason=item.reason, confidence=item.confidence)
            for item in candidates
            if item.name != selected_skill
        ]
        return SkillSelection(
            selected_skill=selected_skill,
            candidates=candidates,
            source="model" if selected_skill else "none",
            reason=reason,
            confidence=confidence,
            rejected=rejected,
        )


class SkillRegistry:
    """Immutable-ish registry snapshot for one runtime instance."""

    def __init__(
        self,
        skills: list[Skill] | None = None,
        *,
        selector: ManifestDrivenSkillSelector | None = None,
    ) -> None:
        self._skills = tuple(skills or [])
        self.selector = selector

    def list(self) -> tuple[Skill, ...]:
        return self._skills

    def manifests(self) -> tuple[SkillManifest, ...]:
        return tuple(getattr(skill, "manifest", self._default_manifest(skill)) for skill in self._skills)

    def get(self, name: str) -> Skill | None:
        lowered = name.lower()
        for skill in self._skills:
            if skill.name.lower() == lowered:
                return skill
        return None

    async def best_match(self, context: AgentContext) -> tuple[Skill | None, SkillMatch]:
        skill, match, _selection = await self.select_best(context)
        return skill, match

    async def select_best(self, context: AgentContext) -> tuple[Skill | None, SkillMatch, SkillSelection]:
        sticky = await self._sticky_match(context)
        if sticky[0] is not None:
            sticky = self._validate_selected_skill(context, *sticky)
            context.skill_selection = sticky[2]
            return sticky

        model = await self._model_match(context)
        if model[0] is not None:
            model = self._validate_selected_skill(context, *model)
            context.skill_selection = model[2]
            return model
        if "unknown skill" in model[2].reason:
            context.skill_selection = model[2]
            return model

        planned = await self._planned_match(context)
        if planned[0] is not None:
            planned = self._validate_selected_skill(context, *planned)
            context.skill_selection = planned[2]
            return planned

        best_skill: Skill | None = None
        best_match = SkillMatch(matched=False, confidence=0.0, reason="no skill matched")
        candidates: list[SkillCandidate] = []
        rejected: list[RejectedSkill] = []
        for skill in self._skills:
            match = await skill.can_handle(context)
            candidates.append(
                SkillCandidate(
                    name=skill.name,
                    matched=match.matched,
                    confidence=match.confidence,
                    reason=match.reason,
                    source="rule_fallback",
                )
            )
            context.trace.add_event(
                "skill.match",
                f"{skill.name}: {match.reason}",
                matched=match.matched,
                confidence=match.confidence,
            )
            if match.matched and match.confidence > best_match.confidence:
                best_skill = skill
                best_match = match
            elif match.matched:
                rejected.append(RejectedSkill(name=skill.name, reason="lower confidence than selected skill", confidence=match.confidence))
        if best_skill is not None:
            rejected.extend(
                RejectedSkill(name=item.name, reason=item.reason, confidence=item.confidence)
                for item in candidates
                if not item.matched
            )
            selection = SkillSelection(
                selected_skill=best_skill.name,
                candidates=candidates,
                source="rule_fallback",
                reason=best_match.reason,
                confidence=best_match.confidence,
                rejected=rejected,
            )
        else:
            selection = SkillSelection(
                selected_skill=None,
                candidates=candidates,
                source="none",
                reason=best_match.reason,
                confidence=best_match.confidence,
                rejected=[
                    RejectedSkill(name=item.name, reason=item.reason, confidence=item.confidence)
                    for item in candidates
                ],
            )
        context.skill_selection = selection
        context.trace.add_event(
            "skill.selection",
            f"selection source={selection.source}",
            **selection.model_dump(),
        )
        return self._validate_selected_skill(context, best_skill, best_match, selection)

    async def _model_match(self, context: AgentContext) -> tuple[Skill | None, SkillMatch, SkillSelection]:
        if self.selector is None:
            match = SkillMatch(matched=False, confidence=0.0, reason="no model selector configured")
            return None, match, SkillSelection(source="none", reason=match.reason)
        selection = await self.selector.select(context, self.manifests())
        if not selection.selected_skill:
            match = SkillMatch(matched=False, confidence=selection.confidence, reason=selection.reason)
            context.trace.add_event(
                "skill.selection",
                f"selection source={selection.source}",
                **selection.model_dump(),
            )
            return None, match, selection
        skill = self.get(selection.selected_skill)
        if skill is None:
            selection = selection.model_copy(
                update={
                    "selected_skill": None,
                    "source": "none",
                    "reason": f"model selected unknown skill: {selection.selected_skill}",
                    "confidence": 0.0,
                }
            )
            match = SkillMatch(matched=False, confidence=0.0, reason=selection.reason)
            context.trace.add_event(
                "skill.selection",
                f"selection source={selection.source}",
                **selection.model_dump(),
            )
            return None, match, selection
        model_match = SkillMatch(
            matched=True,
            confidence=selection.confidence,
            reason=f"model selected {skill.name}; {selection.reason}",
        )
        selection = selection.model_copy(
            update={
                "selected_skill": skill.name,
                "source": "model",
                "reason": model_match.reason,
                "confidence": model_match.confidence,
            }
        )
        context.trace.add_event(
            "skill.selection",
            f"selection source={selection.source}",
            **selection.model_dump(),
        )
        return skill, model_match, selection

    async def _planned_match(self, context: AgentContext) -> tuple[Skill | None, SkillMatch, SkillSelection]:
        if context.plan is None or not context.plan.recommended_skill:
            match = SkillMatch(matched=False, confidence=0.0, reason="no planned skill")
            return None, match, SkillSelection(source="none", reason=match.reason)
        skill = self.get(context.plan.recommended_skill)
        if skill is None:
            context.trace.add_event(
                "skill.planned_missing",
                f"planned skill {context.plan.recommended_skill} is not registered",
            )
            match = SkillMatch(matched=False, confidence=0.0, reason="planned skill not registered")
            return None, match, SkillSelection(source="none", reason=match.reason)
        boosted = 0.8
        planned_match = SkillMatch(
            matched=True,
            confidence=boosted,
            reason=f"planner recommended {skill.name}",
        )
        context.trace.add_event(
            "skill.planned_match",
            f"planner selected {skill.name}",
            confidence=planned_match.confidence,
            reason=planned_match.reason,
        )
        selection = SkillSelection(
            selected_skill=skill.name,
            candidates=[
                SkillCandidate(
                    name=skill.name,
                    matched=True,
                    confidence=planned_match.confidence,
                    reason=planned_match.reason,
                    source="planned",
                )
            ],
            source="planned",
            reason=planned_match.reason,
            confidence=planned_match.confidence,
        )
        context.trace.add_event(
            "skill.selection",
            f"selection source={selection.source}",
            **selection.model_dump(),
        )
        return skill, planned_match, selection

    async def _sticky_match(self, context: AgentContext) -> tuple[Skill | None, SkillMatch, SkillSelection]:
        sticky_candidates = []
        for skill in self._skills:
            manifest = getattr(skill, "manifest", self._default_manifest(skill))
            if manifest.sticky_policy.enabled:
                sticky_candidates.append((skill, manifest.sticky_policy))
        sticky_candidates.sort(key=lambda item: item[1].priority)
        for skill, policy in sticky_candidates:
            status = self._sticky_status(context, skill.name)
            if status is None or status in policy.exit_when or status not in policy.continue_when:
                continue
            sticky_turns = self._sticky_turns(context, skill.name)
            if sticky_turns >= policy.max_turns:
                context.trace.add_event(
                    "skill.sticky_overstay",
                    f"sticky policy exceeded max_turns for {skill.name}",
                    skill=skill.name,
                    sticky_turns=sticky_turns,
                    max_turns=policy.max_turns,
                    status=status,
                )
                continue
            sticky_match = SkillMatch(
                matched=True,
                confidence=0.88,
                reason=f"sticky policy continues {skill.name} while status={status}",
            )
            selection = SkillSelection(
                selected_skill=skill.name,
                candidates=[
                    SkillCandidate(
                        name=skill.name,
                        matched=True,
                        confidence=sticky_match.confidence,
                        reason=sticky_match.reason,
                        source="sticky",
                    )
                ],
                source="sticky",
                reason=sticky_match.reason,
                confidence=sticky_match.confidence,
            )
            context.trace.add_event(
                "skill.selection",
                f"selection source={selection.source}",
                **selection.model_dump(),
            )
            return skill, sticky_match, selection
        return None, SkillMatch(matched=False, confidence=0.0, reason="no sticky skill"), SkillSelection(
            source="none",
            reason="no sticky skill",
        )

    def _validate_selected_skill(
        self,
        context: AgentContext,
        skill: Skill | None,
        match: SkillMatch,
        selection: SkillSelection,
    ) -> tuple[Skill | None, SkillMatch, SkillSelection]:
        if skill is None:
            return skill, match, selection
        missing_tools = self._missing_required_tools(context, skill)
        if not missing_tools:
            return skill, match, selection
        reason = f"missing required tools for {skill.name}: {', '.join(missing_tools)}"
        blocked = selection.model_copy(
            update={
                "selected_skill": None,
                "source": "none",
                "reason": reason,
                "confidence": 0.0,
                "rejected": selection.rejected + [
                    RejectedSkill(name=skill.name, reason=reason, confidence=match.confidence)
                ],
            }
        )
        context.trace.add_event(
            "skill.selection_rejected",
            reason,
            skill=skill.name,
            missing_tools=missing_tools,
        )
        return None, SkillMatch(matched=False, confidence=0.0, reason=reason), blocked

    @staticmethod
    def _missing_required_tools(context: AgentContext, skill: Skill) -> list[str]:
        manifest = getattr(skill, "manifest", SkillRegistry._default_manifest(skill))
        available = {f"{spec.service_id}.{spec.tool_name}" for spec in context.available_tools}
        return [tool_name for tool_name in manifest.required_tools if tool_name not in available]

    @staticmethod
    def _sticky_status(context: AgentContext, skill_name: str) -> str | None:
        if context.session_state is None:
            return None
        state = context.session_state.flat_state.get(skill_name)
        if not isinstance(state, dict):
            return None
        status = state.get("status")
        return str(status) if status is not None else None

    @staticmethod
    def _sticky_turns(context: AgentContext, skill_name: str) -> int:
        if context.session_state is None:
            return 0
        sticky_state = context.session_state.flat_state.get("_sticky_turns")
        if not isinstance(sticky_state, dict):
            return 0
        value = sticky_state.get(skill_name, 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _default_manifest(skill: Skill) -> SkillManifest:
        return SkillManifest(
            name=skill.name,
            title=skill.name,
            description=getattr(skill, "description", ""),
            risk_level=getattr(skill, "risk_level", "low"),
            sticky_policy=StickyPolicy(),
            fallback_policy=FallbackPolicy(),
        )
