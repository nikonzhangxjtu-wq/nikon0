"""Context section read planners."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from nikon0.app.schemas.agent import AgentContext
from nikon0.context.read_planner_prompt import CONTEXT_READ_PLANNER_SYSTEM, build_context_read_planner_user_prompt


ALL_CONTEXT_SECTIONS = (
    "system_policy",
    "workflow",
    "memory",
    "conversation",
    "tool_observations",
    "evidence",
    "current_user",
    "runtime",
)

BASE_CONTEXT_SECTIONS = ("system_policy", "current_user", "memory", "conversation", "runtime")


class ContextPlannerClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class ContextReadPlan(BaseModel):
    included_sections: list[str] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    source: str = "deterministic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def normalized(self) -> "ContextReadPlan":
        seen: set[str] = set()
        included: list[str] = []
        for section in self.included_sections:
            if section in ALL_CONTEXT_SECTIONS and section not in seen:
                included.append(section)
                seen.add(section)
        for section in ("system_policy", "current_user", "runtime"):
            if section not in seen:
                included.append(section)
                seen.add(section)
        return self.model_copy(update={"included_sections": included})


class DeterministicContextReadPlanner:
    """Fast, conservative read planner used as default and fallback."""

    def plan(self, context: AgentContext) -> ContextReadPlan:
        message = context.request.message.strip()
        sections = list(BASE_CONTEXT_SECTIONS)
        reasons = {
            "system_policy": "always needed for answer boundaries",
            "current_user": "current user message is required",
            "memory": "session focus and active issue help resolve references",
            "conversation": "recent conversation helps resolve follow-ups",
            "runtime": "runtime metadata is safe and small",
        }
        if _looks_like_product_support(message):
            sections.append("evidence")
            reasons["evidence"] = "product/manual question may require grounded evidence"
        if _looks_like_workflow(message) or _has_case_state(context):
            for section in ("workflow", "tool_observations"):
                if section not in sections:
                    sections.append(section)
            reasons["workflow"] = "service/refund/complaint flow needs workflow state"
            reasons["tool_observations"] = "business flow may need previous tool observations"
        if context.tool_results and "tool_observations" not in sections:
            sections.append("tool_observations")
            reasons["tool_observations"] = "tool results are available for this turn"
        if context.evidence_context and "evidence" not in sections and _looks_like_followup(message):
            sections.append("evidence")
            reasons["evidence"] = "available evidence may ground follow-up answer"
        return ContextReadPlan(included_sections=sections, reasons=reasons, source="deterministic").normalized()


class LlmContextReadPlanner:
    """LLM planner with strict JSON parsing and deterministic fallback."""

    def __init__(
        self,
        client: ContextPlannerClient,
        *,
        fallback: DeterministicContextReadPlanner | None = None,
        min_confidence: float = 0.4,
    ) -> None:
        self.client = client
        self.fallback = fallback or DeterministicContextReadPlanner()
        self.min_confidence = max(0.0, min(1.0, min_confidence))

    async def aplan(self, context: AgentContext) -> ContextReadPlan:
        messages = [
            {"role": "system", "content": CONTEXT_READ_PLANNER_SYSTEM},
            {
                "role": "user",
                "content": build_context_read_planner_user_prompt(
                    message=context.request.message,
                    memory_preview=context.memory_context[-800:],
                    transcript_preview=context.transcript_context[-1200:],
                ),
            },
        ]
        try:
            raw = await self.client.complete(messages)
            payload = _parse_json(raw)
            sections = payload.get("included_sections")
            if not isinstance(sections, list):
                raise ValueError("included_sections must be a list")
            confidence = _coerce_confidence(payload.get("confidence"))
            if confidence < self.min_confidence:
                raise ValueError("planner confidence below threshold")
            reasons = payload.get("reasons")
            plan = ContextReadPlan(
                included_sections=[str(item) for item in sections],
                reasons={str(k): str(v) for k, v in reasons.items()} if isinstance(reasons, dict) else {},
                source="llm",
                confidence=confidence,
            ).normalized()
            if not plan.included_sections:
                raise ValueError("planner returned no valid sections")
            return plan
        except Exception as exc:  # noqa: BLE001
            fallback = self.fallback.plan(context)
            fallback.reasons["llm_failed"] = f"{type(exc).__name__}: {exc}"
            return fallback


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("planner output must be an object")
    return data


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _looks_like_product_support(message: str) -> bool:
    keywords = (
        "手册",
        "说明书",
        "怎么用",
        "怎么处理",
        "故障码",
        "显示",
        "清洁",
        "安装",
        "滤网",
        "电池",
        "充电",
        "参数",
        "模式",
        "e2",
        "E2",
        "AC900",
    )
    return any(keyword in message for keyword in keywords)


def _looks_like_workflow(message: str) -> bool:
    keywords = ("报修", "售后", "退款", "退货", "投诉", "转人工", "工单", "审批", "维修")
    return any(keyword in message for keyword in keywords)


def _looks_like_followup(message: str) -> bool:
    keywords = ("继续", "刚才", "那个", "还是", "不行", "然后", "下一步")
    return any(keyword in message for keyword in keywords)


def _has_case_state(context: AgentContext) -> bool:
    if context.session_state is None:
        return False
    state = context.session_state.flat_state.get("case_intake")
    return isinstance(state, dict) and bool(state)
