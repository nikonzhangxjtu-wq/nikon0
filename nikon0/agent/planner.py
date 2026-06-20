"""Rule-based planner for the first nikon0 AgentLoop."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.planner import CapabilityCandidate, IntentSignal, PlannerResult, PlanStep
from nikon0.skills.routing_signals import (
    has_approval_signal,
    has_handoff_signal,
    looks_like_case_intake,
    looks_like_product_support,
)


class RuleBasedPlanner:
    """Small deterministic planner that exposes composite intent structure."""

    def plan(self, context: AgentContext) -> PlannerResult:
        text = context.request.message.strip().lower()
        zh_text = context.request.message.strip()
        intents: list[IntentSignal] = [] # 可多意图（如 product_support + refund）
        candidates: list[CapabilityCandidate] = []
        steps: list[PlanStep] = [] # 规划步骤（含尚未实现的 product_support、refund_policy）

        def add_intent(intent: str, confidence: float, reason: str) -> None:
            intents.append(IntentSignal(intent=intent, confidence=confidence, reason=reason))

        if looks_like_case_intake(zh_text):
            add_intent("case_intake", 0.9, "matched repair/service intake signal")
            candidates.append(CapabilityCandidate(kind="skill", name="case_intake", confidence=0.9, reason="service intake"))
            steps.append(PlanStep(step_id="case_intake", capability="case_intake", purpose="collect service case fields"))

        if has_approval_signal(zh_text):
            add_intent("refund", 0.88, "matched refund/return signal")
            candidates.append(CapabilityCandidate(kind="skill", name="case_intake", confidence=0.86, reason="refund intake still uses case intake"))
            steps.append(PlanStep(step_id="refund_policy", capability="refund_policy", purpose="assess refund or return policy"))

        if has_handoff_signal(zh_text):
            add_intent("complaint", 0.9, "matched complaint or handoff signal")
            candidates.append(CapabilityCandidate(kind="skill", name="case_intake", confidence=0.84, reason="complaint intake"))
            steps.append(PlanStep(step_id="complaint_escalation", capability="complaint_escalation", purpose="prepare human escalation"))

        if looks_like_product_support(zh_text)[0]:
            add_intent("product_support", 0.82, "matched product support or troubleshooting signal")
            candidates.append(CapabilityCandidate(kind="skill", name="product_support", confidence=0.8, reason="manual QA via product_support"))
            steps.append(PlanStep(step_id="product_support", capability="product_support", purpose="answer product troubleshooting question"))

        if "tool echo" in text or "工具回声" in zh_text:
            add_intent("tool_echo", 0.95, "matched tool echo verification signal")
            candidates.append(CapabilityCandidate(kind="skill", name="tool_echo", confidence=0.95, reason="tool echo test"))
            steps.append(PlanStep(step_id="tool_echo", capability="tool_echo", purpose="verify tool loop"))

        if not intents:
            add_intent("general", 0.4, "no domain-specific signal")

        recommended_skill = self._recommend_skill(candidates)
        risk_level = "high" if any(intent.intent in {"refund", "complaint"} for intent in intents) else "low"
        return PlannerResult(
            intents=intents,
            candidates=candidates,
            steps=steps,
            recommended_skill=recommended_skill,
            risk_level=risk_level,
            needs_general_handle=recommended_skill is None,
            is_composite=len({intent.intent for intent in intents if intent.intent != "general"}) > 1, # 是否复合意图（如 product_support + refund）
        )

    @staticmethod
    def _recommend_skill(candidates: list[CapabilityCandidate]) -> str | None:
        priority = {"case_intake": 0, "tool_echo": 1, "product_support": 2}
        runnable = [
            candidate
            for candidate in candidates
            if candidate.kind == "skill" and candidate.name in priority
        ]
        if not runnable:
            return None
        runnable.sort(key=lambda item: (priority[item.name], -item.confidence, item.name))
        return runnable[0].name
