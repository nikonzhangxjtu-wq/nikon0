"""Agent evaluation runner for nikon0."""

from __future__ import annotations

import asyncio
import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest, AgentResponse
from nikon0.app.schemas.capability import ToolCallRequest, ToolSpec
from nikon0.eval.agent_dataset import AgentEvalCase, AgentEvalTurn, ExpectedOutcome, load_jsonl_dataset
from nikon0.eval.runtime_profiles import (
    EvalRuntimeProfile,
    RuntimeProfileAudit,
    build_profiled_eval_runtime,
    coerce_runtime_profile,
)
from nikon0.llm import BailianOllamaChatClient, LlmAnswerGenerator
from nikon0.knowledge.runtime import EnterpriseRagBackend, KnowledgeRuntime, StructuredManualBackend
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.model_selector import BailianOllamaSkillSelectionClient, LlmSkillSelector
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.skills.tool_echo import ToolEchoSkill
from nikon0.tools.runtime import EchoTool, ToolRegistry, ToolRuntime


class EvalTurnResult(BaseModel):
    turn_index: int
    message: str
    answer: str
    trace_id: str
    selected_skill: str | None = None
    selection_source: str = ""
    tool_names: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    evidence_count: int = 0
    loop_turns: int = 0
    context_section_names: list[str] = Field(default_factory=list)
    memory_write_outcomes: list[str] = Field(default_factory=list)
    memory_read_source: str = ""
    memory_thread_action: str = ""
    memory_degraded: bool = False


class EvalCaseResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    trace_id: str
    selected_skill: str | None = None
    selection_source: str = ""
    tool_names: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    approval_required: bool = False
    handoff_required: bool = False
    evidence_count: int = 0
    evidence_manual_names: list[str] = Field(default_factory=list)
    rag_backend: str = ""
    rag_backend_ok: bool = False
    rag_fallback_used: bool = False
    rag_backend_trace: list[dict[str, Any]] = Field(default_factory=list)
    fact_coverage_score: float = 1.0
    evidence_alignment_score: float = 1.0
    loop_turns: int = 0
    answer: str = ""
    golden_answer: str = ""
    turn_results: list[EvalTurnResult] = Field(default_factory=list)
    context_section_names: list[str] = Field(default_factory=list)
    expected_context_sections: list[str] = Field(default_factory=list)
    missing_context_sections: list[str] = Field(default_factory=list)
    evidence_section_present: bool = False
    tool_observations_present: bool = False
    llm_context_read_plan_used: bool = False
    llm_context_compactor_used: bool = False
    llm_context_span_selector_used: bool = False
    mock_tool_names: list[str] = Field(default_factory=list)
    memory_write_outcomes: list[str] = Field(default_factory=list)
    memory_read_source: str = ""
    memory_thread_action: str = ""
    memory_degraded: bool = False


class EvalRunReport(BaseModel):
    runtime_profile: str = "custom"
    context_profile: str = "unknown"
    runtime_profile_description: dict[str, Any] = Field(default_factory=dict)
    mock_skill_enabled: bool = False
    mock_tool_usage_count: int = 0
    mock_tool_names: list[str] = Field(default_factory=list)
    context_governance_enabled: bool = True
    llm_context_components_enabled: dict[str, bool] = Field(default_factory=dict)
    eval_runtime_matches_production: bool = False
    production_mismatch_reasons: list[str] = Field(default_factory=list)
    rag_backend_policy: dict[str, Any] = Field(default_factory=dict)
    case_intake_tool_mode: str = ""
    total: int
    passed: int
    pass_rate: float
    skill_accuracy: float
    tool_accuracy: float
    safety_accuracy: float
    evidence_accuracy: float
    answer_constraint_accuracy: float
    fact_coverage_accuracy: float
    evidence_alignment_accuracy: float
    rag_backend_accuracy: float
    enterprise_rag_ok_rate: float
    rag_fallback_rate: float
    fallback_rate: float
    guard_rejection_rate: float
    sticky_overstay_rate: float
    approval_rate: float
    handoff_rate: float
    avg_loop_turns: float
    context_section_miss_rate: float = 0.0
    evidence_omitted_rate: float = 0.0
    tool_observation_omitted_rate: float = 0.0
    memory_write_accept_rate: float = 0.0
    memory_write_reject_rate: float = 0.0
    memory_confirmation_rate: float = 0.0
    memory_degraded_write_rate: float = 0.0
    memory_read_fallback_rate: float = 0.0
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[EvalCaseResult] = Field(default_factory=list)


async def run_agent_eval(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    manual_dir: str | Path = "/Users/nikonzhang/compeletion/手册",
    run_id: str | None = None,
    use_real_llm: bool = True,
    local_rag: bool = False,
    runtime_profile: str | EvalRuntimeProfile = EvalRuntimeProfile.PRODUCTION_LIKE,
    mock_case_intake_tool: bool | None = None,
    show_progress: bool = False,
    runtime: AgentRuntime | None = None,
) -> EvalRunReport:
    cases = load_jsonl_dataset(dataset_path)
    profile_audit: RuntimeProfileAudit
    if runtime is None:
        profiled = build_profiled_eval_runtime(
            runtime_profile=runtime_profile,
            manual_dir=manual_dir,
            use_real_llm=use_real_llm,
            local_rag=local_rag,
            mock_case_intake_tool=mock_case_intake_tool,
        )
        runtime = profiled.runtime
        profile_audit = profiled.audit
        setattr(runtime, "_eval_profile_audit", profile_audit)
    else:
        profile_audit = _runtime_audit_from_runtime(runtime)
    run_dir = Path(output_dir) / (run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalCaseResult] = []
    answer_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_cases = len(cases)
    if show_progress:
        _print_progress(0, total_cases, passed=0, failed=0, case_label="starting")
    for index, case in enumerate(cases, start=1):
        case_result = await _run_case(runtime, case, profile_audit=profile_audit)
        results.append(case_result)
        row = case_result.model_dump()
        row["expected"] = case.expected.model_dump()
        row["golden_answer"] = case.golden_answer
        row["case_metadata"] = case.metadata
        answer_rows.append(row)
        if not case_result.passed:
            failure_rows.append(row)
        if show_progress:
            _print_progress(
                index,
                total_cases,
                passed=sum(1 for result in results if result.passed),
                failed=sum(1 for result in results if not result.passed),
                case_label=case.case_id,
            )
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    report = _aggregate(results, profile_audit=profile_audit)
    (run_dir / "answers.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in answer_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "failures.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in failure_rows) + ("\n" if failure_rows else ""),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(report.model_dump_json(indent=2, exclude={"results"}), encoding="utf-8")
    (run_dir / "metrics.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


async def _run_case(
    runtime: AgentRuntime,
    case: AgentEvalCase,
    *,
    profile_audit: RuntimeProfileAudit,
) -> EvalCaseResult:
    turns = case.turns or [AgentEvalTurn(message=case.message, images=case.images, metadata=case.metadata)]
    session_id = case.session_id or f"eval-{case.case_id}"
    turn_results: list[EvalTurnResult] = []
    final_response: AgentResponse | None = None
    for index, turn in enumerate(turns, start=1):
        response = await runtime.run(
            AgentRequest(
                session_id=session_id,
                user_id=case.user_id,
                message=turn.message,
                images=turn.images,
                metadata={**case.metadata, **turn.metadata},
            )
        )
        final_response = response
        turn_results.append(_summarize_turn(index, turn, response))
    assert final_response is not None
    failures = _score_case(case, final_response)
    fact_coverage_score = _fact_coverage_score(case, final_response)
    evidence_alignment_score = _evidence_alignment_score(case, final_response)
    rag_audit = _rag_audit(final_response)
    context_audit = _context_audit(case, final_response, turn_count=len(turns))
    mock_tool_names = [name for name in _tool_names(final_response) if name in set(profile_audit.mock_tool_names)]
    memory_audit = _memory_audit(final_response)
    return EvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        passed=not failures,
        failures=failures,
        trace_id=final_response.trace_id,
        selected_skill=_selection(final_response).get("selected_skill"),
        selection_source=str(_selection(final_response).get("source") or ""),
        tool_names=_tool_names(final_response),
        risk_level=final_response.risk_level,
        approval_required=any(item.kind == "approval" for item in final_response.actions),
        handoff_required=any(item.kind == "handoff" for item in final_response.actions),
        evidence_count=_evidence_count(final_response),
        evidence_manual_names=_evidence_manual_names(final_response),
        rag_backend=rag_audit["backend"],
        rag_backend_ok=bool(rag_audit["ok"]),
        rag_fallback_used=bool(rag_audit["fallback_used"]),
        rag_backend_trace=rag_audit["backend_trace"],
        fact_coverage_score=fact_coverage_score,
        evidence_alignment_score=evidence_alignment_score,
        loop_turns=int((final_response.debug.get("loop") or {}).get("turn_count") or 0),
        answer=final_response.answer,
        golden_answer=case.golden_answer,
        turn_results=turn_results,
        context_section_names=context_audit["section_names"],
        expected_context_sections=context_audit["expected_sections"],
        missing_context_sections=context_audit["missing_sections"],
        evidence_section_present=context_audit["evidence_section_present"],
        tool_observations_present=context_audit["tool_observations_present"],
        llm_context_read_plan_used=context_audit["llm_read_plan_used"],
        llm_context_compactor_used=context_audit["llm_compactor_used"],
        llm_context_span_selector_used=context_audit["llm_span_selector_used"],
        mock_tool_names=mock_tool_names,
        memory_write_outcomes=memory_audit["write_outcomes"],
        memory_read_source=memory_audit["read_source"],
        memory_thread_action=memory_audit["thread_action"],
        memory_degraded=memory_audit["degraded"],
    )


def build_eval_runtime(
    *,
    manual_dir: str | Path,
    use_real_llm: bool = True,
    local_rag: bool = False,
    runtime_profile: str | EvalRuntimeProfile = EvalRuntimeProfile.PRODUCTION_LIKE,
    mock_case_intake_tool: bool | None = None,
) -> AgentRuntime:
    profiled = build_profiled_eval_runtime(
        runtime_profile=runtime_profile,
        manual_dir=manual_dir,
        use_real_llm=use_real_llm,
        local_rag=local_rag,
        mock_case_intake_tool=mock_case_intake_tool,
    )
    setattr(profiled.runtime, "_eval_profile_audit", profiled.audit)
    return profiled.runtime


def _build_product_knowledge_runtime(*, manual_dir: str | Path, local_rag: bool = False) -> KnowledgeRuntime:
    local_backend = StructuredManualBackend(manual_dir)
    if local_rag:
        return KnowledgeRuntime(local_backend)
    return KnowledgeRuntime(EnterpriseRagBackend(fallback_backend=local_backend))


def _build_answer_generator(*, use_real_llm: bool) -> LlmAnswerGenerator | None:
    if not use_real_llm:
        return None
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return None
    model = getattr(settings, "simple_llm_model", "") or getattr(settings, "gen_model", "")
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


def _build_selector(*, answer_generator: LlmAnswerGenerator | None, enabled: bool):
    if not enabled:
        return None
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return None
    if not bool(getattr(settings, "router_llm_enabled", False)):
        return None
    model = (
        getattr(settings, "router_llm_model", "")
        or getattr(settings, "simple_llm_model", "")
        or getattr(settings, "gen_model", "")
    )
    if not model:
        return None
    return LlmSkillSelector(
        BailianOllamaSkillSelectionClient(
            model=model,
            temperature=0.0,
            max_tokens=256,
            timeout=15,
        )
    )


class _EvalCaseIntakeTool:
    def __init__(self, tool_name: str) -> None:
        self.spec = ToolSpec(
            service_id="case-intake",
            tool_name=tool_name,
            description="Deterministic eval-time intake tool.",
            risk_level="medium",
        )

    async def call(self, request: ToolCallRequest):
        from nikon0.app.schemas.capability import ToolCallResult

        question = str(request.arguments.get("question") or "")
        if request.tool_name == "try_cancel_case_intake":
            return ToolCallResult(
                ok=True,
                service_id=request.service_id,
                tool_name=request.tool_name,
                data={
                    "completed": False,
                    "exited": True,
                    "reply_text": "好的，已取消当前工单收集。",
                    "missing_slots": [],
                    "ticket_payload": {},
                    "context_block": "[工单收集状态]\nstatus: cancelled",
                },
            )
        if "型号" in question and "电话" in question:
            payload = {
                "completed": True,
                "exited": False,
                "reply_text": "已为你完成售后受理信息收集。",
                "missing_slots": [],
                "ticket_payload": {
                    "intent": "repair",
                    "product_model": "AC900",
                    "issue": question,
                    "contact_phone": "13800138000",
                    "priority": "medium",
                    "status": "ready",
                },
                "context_block": "[工单收集状态]\nstatus: ready",
            }
        elif "退款" in question or "退货" in question or "换货" in question:
            payload = {
                "completed": False,
                "exited": False,
                "reply_text": "为处理退款，请提供订单号和联系电话。",
                "missing_slots": ["order_id", "contact_phone"],
                "ticket_payload": {"intent": "refund", "status": "collecting"},
                "context_block": "[工单收集状态]\nintent: refund\nstatus: collecting",
            }
        else:
            payload = {
                "completed": False,
                "exited": False,
                "reply_text": "为尽快处理，请提供产品型号和联系电话。",
                "missing_slots": ["product_model", "contact_phone"],
                "ticket_payload": {"intent": "repair", "status": "collecting"},
                "context_block": "[工单收集状态]\nintent: repair\nstatus: collecting",
            }
        return ToolCallResult(ok=True, service_id=request.service_id, tool_name=request.tool_name, data=payload)


def _summarize_turn(index: int, turn: AgentEvalTurn, response: AgentResponse) -> EvalTurnResult:
    selection = _selection(response)
    context_debug = response.debug.get("context_debug") or {}
    memory_audit = _memory_audit(response)
    return EvalTurnResult(
        turn_index=index,
        message=turn.message,
        answer=response.answer,
        trace_id=response.trace_id,
        selected_skill=selection.get("selected_skill"),
        selection_source=str(selection.get("source") or ""),
        tool_names=_tool_names(response),
        risk_level=response.risk_level,
        evidence_count=_evidence_count(response),
        loop_turns=int((response.debug.get("loop") or {}).get("turn_count") or 0),
        context_section_names=[str(item) for item in context_debug.get("section_names") or []],
        memory_write_outcomes=memory_audit["write_outcomes"],
        memory_read_source=memory_audit["read_source"],
        memory_thread_action=memory_audit["thread_action"],
        memory_degraded=memory_audit["degraded"],
    )


def _score_case(case: AgentEvalCase, response: AgentResponse) -> list[str]:
    failures: list[str] = []
    selection = _selection(response)
    selected_skill = selection.get("selected_skill")
    tool_names = _tool_names(response)
    evidence_count = _evidence_count(response)
    approval_required = any(item.kind == "approval" for item in response.actions)
    handoff_required = any(item.kind == "handoff" for item in response.actions)

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
    if case.category == "product_support":
        rag_audit = _rag_audit(response)
        if rag_audit["backend"] != "enterprise_rag" or not rag_audit["ok"]:
            failures.append(
                f"rag backend expected enterprise_rag ok=True, got {rag_audit['backend']} ok={rag_audit['ok']}"
            )
        if rag_audit["fallback_used"]:
            failures.append("rag fallback used: structured_manual")
        fact_score = _fact_coverage_score(case, response)
        evidence_score = _evidence_alignment_score(case, response)
        if fact_score < 1.0:
            failures.append(f"fact coverage below threshold: {fact_score:.2f}")
        if evidence_score < 1.0:
            failures.append(f"evidence alignment below threshold: {evidence_score:.2f}")
    else:
        for keyword in exp.answer_must_contain:
            if keyword not in response.answer:
                failures.append(f"answer missing keyword: {keyword}")
    for keyword in exp.answer_must_not_contain:
        if keyword in response.answer:
            failures.append(f"answer contains forbidden keyword: {keyword}")
    return failures


def _fact_coverage_score(case: AgentEvalCase, response: AgentResponse) -> float:
    if case.category != "product_support":
        return 1.0
    checks = case.expected.answer_must_contain or _metadata_strings(case, "source_facts")
    if not checks:
        return 1.0
    covered = sum(1 for item in checks if _phrase_semantically_covered(item, response.answer))
    return _ratio(covered, len(checks))


def _evidence_alignment_score(case: AgentEvalCase, response: AgentResponse) -> float:
    if case.category != "product_support":
        return 1.0
    expected = _expected_manual_name(case)
    if not expected:
        return 1.0
    actual = _evidence_manual_names(response)
    if not actual:
        return 0.0
    expected_norm = _normalize_eval_text(expected)
    return 1.0 if any(expected_norm and expected_norm in _normalize_eval_text(name) for name in actual) else 0.0


def _selection(response: AgentResponse) -> dict[str, Any]:
    selection = response.debug.get("skill_selection") or {}
    return selection if isinstance(selection, dict) else {}


def _tool_names(response: AgentResponse) -> list[str]:
    return [action.name for action in response.actions if action.kind == "tool"]


def _evidence_count(response: AgentResponse) -> int:
    trace = response.debug.get("trace") or {}
    knowledge = trace.get("knowledge_calls") or []
    if knowledge:
        try:
            return max(int(item.get("evidence_count") or 0) for item in knowledge)
        except ValueError:
            return 0
    return len(trace.get("memory_updates") or [])


def _evidence_manual_names(response: AgentResponse) -> list[str]:
    names: set[str] = set()
    trace = response.debug.get("trace") or {}
    for update in trace.get("memory_updates") or []:
        if not isinstance(update, dict):
            continue
        value = update.get("value")
        if not isinstance(value, dict):
            continue
        for name in value.get("manual_names") or []:
            if str(name).strip():
                names.add(str(name).strip())
    return sorted(names)


def _rag_audit(response: AgentResponse) -> dict[str, Any]:
    backend_trace = _rag_backend_trace(response)
    if not backend_trace:
        return {"backend": "", "ok": False, "fallback_used": False, "backend_trace": []}
    primary = backend_trace[0]
    backend = str(primary.get("backend") or "")
    ok = bool(primary.get("ok", backend == "structured_manual"))
    fallback_used = bool(primary.get("fallback") or any(item.get("backend") == "structured_manual" for item in backend_trace[1:]))
    return {
        "backend": backend,
        "ok": ok,
        "fallback_used": fallback_used,
        "backend_trace": backend_trace,
    }


def _rag_backend_trace(response: AgentResponse) -> list[dict[str, Any]]:
    trace = response.debug.get("trace") or {}
    knowledge = trace.get("knowledge_calls") or []
    backend_trace: list[dict[str, Any]] = []
    for call in knowledge:
        if not isinstance(call, dict):
            continue
        for item in call.get("backend_trace") or []:
            if isinstance(item, dict):
                backend_trace.append(item)
    return backend_trace


def _context_audit(case: AgentEvalCase, response: AgentResponse, *, turn_count: int) -> dict[str, Any]:
    context_debug = response.debug.get("context_debug") or {}
    section_names = [str(item) for item in context_debug.get("section_names") or []]
    section_set = set(section_names)
    expected_sections = _expected_context_sections(case, response, turn_count=turn_count)
    missing_sections = [section for section in expected_sections if section not in section_set]
    trace = response.debug.get("trace") or {}
    events = [event for event in trace.get("events") or [] if isinstance(event, dict)]
    return {
        "section_names": section_names,
        "expected_sections": expected_sections,
        "missing_sections": missing_sections,
        "evidence_section_present": "evidence" in section_set,
        "tool_observations_present": "tool_observations" in section_set,
        "llm_read_plan_used": any(
            event.get("stage") == "context.read_plan"
            and (event.get("payload") or {}).get("source") == "llm"
            for event in events
        ),
        "llm_compactor_used": any(
            event.get("stage") == "context.conversation_compact"
            and (event.get("payload") or {}).get("source") == "llm"
            for event in events
        ),
        "llm_span_selector_used": any(
            event.get("stage") == "context.evidence_pack"
            and (event.get("payload") or {}).get("source") == "llm_span_selector"
            for event in events
        ),
    }


def _memory_audit(response: AgentResponse) -> dict[str, Any]:
    governance = response.debug.get("memory_governance") or {}
    decisions = governance.get("write_decisions") or []
    return {
        "write_outcomes": [str(item.get("outcome")) for item in decisions if isinstance(item, dict)],
        "read_source": str((governance.get("read_plan") or {}).get("source") or ""),
        "thread_action": str((governance.get("thread_decision") or {}).get("action") or ""),
        "degraded": bool(governance.get("degraded")),
    }


def _expected_context_sections(case: AgentEvalCase, response: AgentResponse, *, turn_count: int) -> list[str]:
    expected = ["system_policy", "current_user", "memory", "runtime"]
    if turn_count > 1:
        expected.append("conversation")
    if case.category == "product_support" and _evidence_count(response) >= case.expected.min_evidence_count:
        expected.append("evidence")
    if _tool_names(response):
        expected.append("tool_observations")
    return expected


def _runtime_audit_from_runtime(runtime: AgentRuntime) -> RuntimeProfileAudit:
    audit = getattr(runtime, "_eval_profile_audit", None)
    if isinstance(audit, RuntimeProfileAudit):
        return audit
    return RuntimeProfileAudit(
        runtime_profile="custom",
        context_profile="unknown",
        context_governance_enabled=runtime.context_governance is not None,
        mock_skill_enabled=any(skill.name == "mock_enterprise_assistant" for skill in runtime.skill_registry.list()),
        eval_runtime_matches_production=False,
        production_mismatch_reasons=["custom runtime was injected; production match not inferred"],
    )


def _expected_manual_name(case: AgentEvalCase) -> str:
    source = case.metadata.get("source_manual")
    if not source:
        return ""
    return Path(str(source)).stem


def _metadata_strings(case: AgentEvalCase, key: str) -> list[str]:
    raw = case.metadata.get(key)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _phrase_semantically_covered(expected: str, answer: str) -> bool:
    expected_norm = _normalize_eval_text(expected)
    answer_norm = _normalize_eval_text(answer)
    if not expected_norm:
        return True
    if expected_norm in answer_norm:
        return True
    numbers = re.findall(r"±?\d+(?:\.\d+)?|[a-z]+-\d+|[a-z]+\d+", expected_norm)
    if numbers and not all(number in answer_norm for number in numbers):
        return False
    required_terms = _required_terms(expected_norm)
    if required_terms:
        return all(term in answer_norm for term in required_terms)
    return False


def _required_terms(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9+\-]*|[\u4e00-\u9fff]{2,}", text):
        if token in _EVAL_STOPWORDS:
            continue
        cleaned = token
        for stopword in _EVAL_STOPWORDS:
            cleaned = cleaned.replace(stopword, "")
        if len(cleaned) >= 2 and not cleaned.isdigit():
            tokens.append(cleaned)
    return tokens[:4]


def _normalize_eval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    replacements = {
        "六到十二": "6-12",
        "六至十二": "6-12",
        "六十二": "62",
        "一到二": "1-2",
        "一至二": "1-2",
        "十二": "12",
        "十": "10",
        "九": "9",
        "六": "6",
        "三": "3",
    }
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("—", "-").replace("–", "-").replace("~", "-")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _aggregate(results: list[EvalCaseResult], *, profile_audit: RuntimeProfileAudit) -> EvalRunReport:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    product_results = [result for result in results if result.category == "product_support"]
    product_total = len(product_results)
    evidence_expected = [result for result in results if "evidence" in result.expected_context_sections]
    tool_observation_expected = [result for result in results if "tool_observations" in result.expected_context_sections]
    memory_decisions = [outcome for result in results for outcome in result.memory_write_outcomes]
    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({result.category for result in results})
    for category in categories:
        subset = [result for result in results if result.category == category]
        by_category[category] = {
            "total": float(len(subset)),
            "pass_rate": _ratio(sum(1 for result in subset if result.passed), len(subset)),
        }
    return EvalRunReport(
        runtime_profile=profile_audit.runtime_profile,
        context_profile=profile_audit.context_profile,
        runtime_profile_description=profile_audit.model_dump(),
        mock_skill_enabled=profile_audit.mock_skill_enabled,
        mock_tool_usage_count=sum(len(result.mock_tool_names) for result in results),
        mock_tool_names=profile_audit.mock_tool_names,
        context_governance_enabled=profile_audit.context_governance_enabled,
        llm_context_components_enabled=profile_audit.llm_context_components_enabled,
        eval_runtime_matches_production=profile_audit.eval_runtime_matches_production,
        production_mismatch_reasons=profile_audit.production_mismatch_reasons,
        rag_backend_policy=profile_audit.rag_backend_policy,
        case_intake_tool_mode=profile_audit.case_intake_tool_mode,
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
        answer_constraint_accuracy=_ratio(sum(1 for result in results if not _answer_failure(result)), total),
        fact_coverage_accuracy=_ratio(sum(1 for result in results if not _has_failure(result, "fact coverage")), total),
        evidence_alignment_accuracy=_ratio(
            sum(1 for result in results if not _has_failure(result, "evidence alignment")),
            total,
        ),
        rag_backend_accuracy=_ratio(sum(1 for result in product_results if not _has_failure(result, "rag backend")), product_total),
        enterprise_rag_ok_rate=_ratio(
            sum(1 for result in product_results if result.rag_backend == "enterprise_rag" and result.rag_backend_ok),
            product_total,
        ),
        rag_fallback_rate=_ratio(sum(1 for result in product_results if result.rag_fallback_used), product_total),
        fallback_rate=_ratio(sum(1 for result in results if any(item.startswith("skill expected") for item in result.failures) and result.selection_source == "none"), total),
        guard_rejection_rate=_ratio(sum(1 for result in results if result.selection_source == "none" and result.selected_skill is None), total),
        sticky_overstay_rate=_ratio(sum(1 for result in results if any("sticky" in item for item in result.failures)), total),
        approval_rate=_ratio(sum(1 for result in results if result.approval_required), total),
        handoff_rate=_ratio(sum(1 for result in results if result.handoff_required), total),
        avg_loop_turns=mean([result.loop_turns for result in results]) if results else 0.0,
        context_section_miss_rate=_ratio(sum(1 for result in results if result.missing_context_sections), total),
        evidence_omitted_rate=_ratio(
            sum(1 for result in evidence_expected if "evidence" in result.missing_context_sections),
            len(evidence_expected),
        ),
        tool_observation_omitted_rate=_ratio(
            sum(1 for result in tool_observation_expected if "tool_observations" in result.missing_context_sections),
            len(tool_observation_expected),
        ),
        memory_write_accept_rate=_ratio(sum(1 for item in memory_decisions if item == "accept"), len(memory_decisions)),
        memory_write_reject_rate=_ratio(sum(1 for item in memory_decisions if item == "reject"), len(memory_decisions)),
        memory_confirmation_rate=_ratio(sum(1 for item in memory_decisions if item == "needs_confirmation"), len(memory_decisions)),
        memory_degraded_write_rate=_ratio(sum(1 for result in results if result.memory_degraded), total),
        memory_read_fallback_rate=_ratio(sum(1 for result in results if result.memory_read_source == "deterministic"), total),
        by_category=by_category,
        results=results,
    )


def _answer_failure(result: EvalCaseResult) -> bool:
    return any(item.startswith("answer ") for item in result.failures)


def _has_failure(result: EvalCaseResult, prefix: str) -> bool:
    return any(item.startswith(prefix) for item in result.failures)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _render_markdown(report: EvalRunReport) -> str:
    lines = [
        "# nikon0 Eval Report",
        "",
        f"- runtime_profile: {report.runtime_profile}",
        f"- context_profile: {report.context_profile}",
        f"- eval_runtime_matches_production: {str(report.eval_runtime_matches_production).lower()}",
        f"- production_mismatch_reasons: {', '.join(report.production_mismatch_reasons) if report.production_mismatch_reasons else 'none'}",
        f"- mock_skill_enabled: {str(report.mock_skill_enabled).lower()}",
        f"- mock_tool_usage_count: {report.mock_tool_usage_count}",
        f"- mock_tool_names: {', '.join(report.mock_tool_names) if report.mock_tool_names else 'none'}",
        f"- context_governance_enabled: {str(report.context_governance_enabled).lower()}",
        f"- llm_context_components_enabled: {json.dumps(report.llm_context_components_enabled, ensure_ascii=False, sort_keys=True)}",
        f"- case_intake_tool_mode: {report.case_intake_tool_mode}",
        "",
        f"- total: {report.total}",
        f"- passed: {report.passed}",
        f"- pass_rate: {report.pass_rate:.4f}",
        f"- skill_accuracy: {report.skill_accuracy:.4f}",
        f"- tool_accuracy: {report.tool_accuracy:.4f}",
        f"- safety_accuracy: {report.safety_accuracy:.4f}",
        f"- evidence_accuracy: {report.evidence_accuracy:.4f}",
        f"- answer_constraint_accuracy: {report.answer_constraint_accuracy:.4f}",
        f"- fact_coverage_accuracy: {report.fact_coverage_accuracy:.4f}",
        f"- evidence_alignment_accuracy: {report.evidence_alignment_accuracy:.4f}",
        f"- rag_backend_accuracy: {report.rag_backend_accuracy:.4f}",
        f"- enterprise_rag_ok_rate: {report.enterprise_rag_ok_rate:.4f}",
        f"- rag_fallback_rate: {report.rag_fallback_rate:.4f}",
        f"- fallback_rate: {report.fallback_rate:.4f}",
        f"- guard_rejection_rate: {report.guard_rejection_rate:.4f}",
        f"- sticky_overstay_rate: {report.sticky_overstay_rate:.4f}",
        f"- approval_rate: {report.approval_rate:.4f}",
        f"- handoff_rate: {report.handoff_rate:.4f}",
        f"- avg_loop_turns: {report.avg_loop_turns:.2f}",
        f"- context_section_miss_rate: {report.context_section_miss_rate:.4f}",
        f"- evidence_omitted_rate: {report.evidence_omitted_rate:.4f}",
        f"- tool_observation_omitted_rate: {report.tool_observation_omitted_rate:.4f}",
        f"- memory_write_accept_rate: {report.memory_write_accept_rate:.4f}",
        f"- memory_write_reject_rate: {report.memory_write_reject_rate:.4f}",
        f"- memory_confirmation_rate: {report.memory_confirmation_rate:.4f}",
        f"- memory_degraded_write_rate: {report.memory_degraded_write_rate:.4f}",
        f"- memory_read_fallback_rate: {report.memory_read_fallback_rate:.4f}",
        "",
        "| category | total | pass_rate |",
        "|---|---:|---:|",
    ]
    for category, stats in report.by_category.items():
        lines.append(f"| {category} | {int(stats['total'])} | {stats['pass_rate']:.4f} |")
    return "\n".join(lines) + "\n"


def _print_progress(current: int, total: int, *, passed: int, failed: int, case_label: str = "") -> None:
    width = 28
    ratio = current / total if total else 1.0
    filled = min(width, max(0, int(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" current={case_label}" if case_label else ""
    sys.stderr.write(
        f"\r[eval] [{bar}] {current}/{total} pass={passed} fail={failed}{suffix}"
    )
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nikon0 agent eval.")
    parser.add_argument(
        "--dataset",
        default="/Users/nikonzhang/compeletion/nikon0/eval/datasets/agent_eval_150.jsonl",
        help="Path to the eval dataset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/nikonzhang/compeletion/nikon0/eval/reports",
        help="Directory where report folders will be written.",
    )
    parser.add_argument(
        "--manual-dir",
        default="/Users/nikonzhang/compeletion/手册",
        help="Manual directory used by product-support RAG.",
    )
    parser.add_argument("--run-id", default=None, help="Report subdirectory name.")
    parser.add_argument("--target-size", type=int, default=150, help="Dataset size when --build-dataset is set.")
    parser.add_argument("--build-dataset", action="store_true", help="Build the high-quality dataset before running.")
    parser.add_argument(
        "--runtime-profile",
        default=EvalRuntimeProfile.PRODUCTION_LIKE.value,
        choices=[item.value for item in EvalRuntimeProfile],
        help="Runtime profile used for eval.",
    )
    parser.add_argument("--no-real-llm", action="store_true", help="Disable real LLM calls for a local deterministic baseline.")
    parser.add_argument("--local-rag", action="store_true", help="Use StructuredManualBackend instead of EnterpriseRagBackend.")
    parser.add_argument("--mock-case-intake-tool", action="store_true", help="Use deterministic eval-time case-intake tools.")
    parser.add_argument("--progress", action="store_true", help="Show an in-terminal progress bar.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    manual_dir = Path(args.manual_dir)
    if args.build_dataset:
        from nikon0.eval.agent_dataset import build_high_quality_agent_dataset, write_jsonl_dataset

        cases = build_high_quality_agent_dataset(manual_dir=manual_dir, target_size=args.target_size)
        write_jsonl_dataset(cases, dataset_path)
        print(f"dataset={dataset_path}")

    report = asyncio.run(
        run_agent_eval(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            manual_dir=manual_dir,
            run_id=args.run_id,
            use_real_llm=not args.no_real_llm,
            local_rag=args.local_rag,
            runtime_profile=coerce_runtime_profile(args.runtime_profile),
            mock_case_intake_tool=True if args.mock_case_intake_tool else None,
            show_progress=args.progress,
        )
    )
    run_id = args.run_id or "latest timestamped run"
    print(f"run_id={run_id}")
    print(f"total={report.total} passed={report.passed} pass_rate={report.pass_rate:.4f}")
    print(f"runtime_profile={report.runtime_profile}")
    print(f"context_profile={report.context_profile}")
    print(f"eval_runtime_matches_production={report.eval_runtime_matches_production}")
    print(f"mock_tool_usage_count={report.mock_tool_usage_count}")
    print(f"fact_coverage_accuracy={report.fact_coverage_accuracy:.4f}")
    print(f"evidence_alignment_accuracy={report.evidence_alignment_accuracy:.4f}")
    print(f"rag_backend_accuracy={report.rag_backend_accuracy:.4f}")
    print(f"enterprise_rag_ok_rate={report.enterprise_rag_ok_rate:.4f}")
    print(f"rag_fallback_rate={report.rag_fallback_rate:.4f}")


_EVAL_STOPWORDS = {
    "一次",
    "以上",
    "以下",
    "建议",
    "手册",
    "表示",
    "进行",
    "可以",
    "不能",
    "不要",
    "请勿",
    "当前",
    "使用",
    "支持",
    "功能",
}


__all__ = ["run_agent_eval", "build_eval_runtime", "EvalRunReport", "EvalCaseResult", "EvalTurnResult"]


if __name__ == "__main__":
    main()
