"""Agent evaluation runner and metric aggregation."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from pydantic import BaseModel, Field

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest, AgentResponse
from nikon0.eval.agent_dataset import AgentEvalCase


class AgentEvalCaseResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    trace_id: str
    selected_skill: str | None
    selection_source: str
    tool_names: list[str] = Field(default_factory=list)
    risk_level: str
    approval_required: bool
    handoff_required: bool
    evidence_count: int
    loop_turns: int
    fallback_count: int
    guard_rejected_count: int
    sticky_overstay_count: int
    failures: list[str] = Field(default_factory=list)


class AgentMetricReport(BaseModel):
    total: int
    passed: int
    pass_rate: float
    skill_accuracy: float
    tool_accuracy: float
    safety_accuracy: float
    evidence_accuracy: float
    fallback_rate: float
    guard_rejection_rate: float
    sticky_overstay_rate: float
    approval_rate: float
    handoff_rate: float
    avg_loop_turns: float
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[AgentEvalCaseResult] = Field(default_factory=list)


class AgentEvaluationHarness:
    """Runs golden cases through AgentRuntime and computes platform metrics."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run_cases(self, cases: list[AgentEvalCase]) -> AgentMetricReport:
        results: list[AgentEvalCaseResult] = []
        for index, case in enumerate(cases, start=1):
            response = await self.runtime.run(
                AgentRequest(
                    session_id=case.session_id or f"agent-eval-{case.case_id}-{index}",
                    user_id=case.user_id,
                    message=case.message,
                    images=case.images,
                    metadata=case.metadata,
                )
            )
            results.append(_score_case(case, response))
        return _aggregate(results)


def _score_case(case: AgentEvalCase, response: AgentResponse) -> AgentEvalCaseResult:
    debug = response.debug or {}
    trace = debug.get("trace") or {}
    selection = debug.get("skill_selection") or {}
    loop = debug.get("loop") or {}
    events = trace.get("events") or []
    actions = [action.model_dump() for action in response.actions]

    selected_skill = selection.get("selected_skill")
    tool_names = [
        action["name"]
        for action in actions
        if action.get("kind") == "tool"
    ]
    approval_required = any(action.get("kind") == "approval" for action in actions)
    handoff_required = any(action.get("kind") == "handoff" for action in actions)
    evidence_count = _evidence_count(trace)
    fallback_count = _event_count(events, "skill.fallback")
    guard_rejected_count = _event_count(events, "skill.guard_rejected")
    sticky_overstay_count = _event_count(events, "skill.sticky_overstay")
    failures: list[str] = []

    exp = case.expected
    if exp.acceptable_skills and selected_skill not in exp.acceptable_skills:
        failures.append(f"skill expected one of {exp.acceptable_skills}, got {selected_skill}")
    missing_tools = [tool for tool in exp.required_tools if tool not in tool_names]
    if missing_tools:
        failures.append(f"missing tools: {', '.join(missing_tools)}")
    if exp.risk_level and response.risk_level != exp.risk_level:
        failures.append(f"risk expected {exp.risk_level}, got {response.risk_level}")
    if exp.approval_required != approval_required:
        failures.append(f"approval expected {exp.approval_required}, got {approval_required}")
    if exp.handoff_required != handoff_required:
        failures.append(f"handoff expected {exp.handoff_required}, got {handoff_required}")
    if evidence_count < exp.min_evidence_count:
        failures.append(f"evidence expected >= {exp.min_evidence_count}, got {evidence_count}")
    for keyword in exp.answer_must_contain:
        if keyword not in response.answer:
            failures.append(f"answer missing keyword: {keyword}")
    for keyword in exp.answer_must_not_contain:
        if keyword in response.answer:
            failures.append(f"answer contains forbidden keyword: {keyword}")

    return AgentEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        passed=not failures,
        trace_id=response.trace_id,
        selected_skill=selected_skill,
        selection_source=str(selection.get("source") or "none"),
        tool_names=tool_names,
        risk_level=response.risk_level,
        approval_required=approval_required,
        handoff_required=handoff_required,
        evidence_count=evidence_count,
        loop_turns=int(loop.get("turn_count") or 0),
        fallback_count=fallback_count,
        guard_rejected_count=guard_rejected_count,
        sticky_overstay_count=sticky_overstay_count,
        failures=failures,
    )


def _aggregate(results: list[AgentEvalCaseResult]) -> AgentMetricReport:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({result.category for result in results})
    for category in categories:
        subset = [result for result in results if result.category == category]
        by_category[category] = {
            "total": float(len(subset)),
            "pass_rate": _ratio(sum(1 for result in subset if result.passed), len(subset)),
        }

    expected_skill_cases = [result for result in results if result.selected_skill is not None or result.passed]
    _ = expected_skill_cases
    return AgentMetricReport(
        total=total,
        passed=passed,
        pass_rate=_ratio(passed, total),
        skill_accuracy=_ratio(sum(1 for result in results if not _has_failure(result, "skill expected")), total),
        tool_accuracy=_ratio(sum(1 for result in results if not _has_failure(result, "missing tools")), total),
        safety_accuracy=_ratio(
            sum(
                1
                for result in results
                if not _has_failure(result, "risk expected")
                and not _has_failure(result, "approval expected")
                and not _has_failure(result, "handoff expected")
            ),
            total,
        ),
        evidence_accuracy=_ratio(sum(1 for result in results if not _has_failure(result, "evidence expected")), total),
        fallback_rate=_ratio(sum(1 for result in results if result.fallback_count > 0), total),
        guard_rejection_rate=_ratio(sum(1 for result in results if result.guard_rejected_count > 0), total),
        sticky_overstay_rate=_ratio(sum(1 for result in results if result.sticky_overstay_count > 0), total),
        approval_rate=_ratio(sum(1 for result in results if result.approval_required), total),
        handoff_rate=_ratio(sum(1 for result in results if result.handoff_required), total),
        avg_loop_turns=mean([result.loop_turns for result in results]) if results else 0.0,
        by_category=by_category,
        results=results,
    )


def _evidence_count(trace: dict) -> int:
    knowledge = trace.get("knowledge_calls") or []
    if knowledge:
        return max(int(item.get("evidence_count") or 0) for item in knowledge)
    return len(trace.get("memory_updates") or [])


def _event_count(events: list[dict], stage: str) -> int:
    return sum(1 for event in events if event.get("stage") == stage)


def _has_failure(result: AgentEvalCaseResult, prefix: str) -> bool:
    return any(item.startswith(prefix) for item in result.failures)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
