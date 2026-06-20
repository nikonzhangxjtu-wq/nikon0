"""Context-specific evaluation harness and stress runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, model_validator

from nikon0.agent.context_governance import ContextGovernance
from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentContext, AgentRequest, AgentResponse
from nikon0.app.schemas.capability import Evidence
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.context.runtime import ContextRuntime
from nikon0.eval.agent_dataset import AgentEvalTurn
from nikon0.eval.runtime_profiles import (
    EvalRuntimeProfile,
    RuntimeProfileAudit,
    build_profiled_eval_runtime,
    coerce_runtime_profile,
)


class ContextEvalExpected(BaseModel):
    required_sections: list[str] = Field(default_factory=list)
    forbidden_sections: list[str] = Field(default_factory=list)
    required_prompt_strings: list[str] = Field(default_factory=list)
    forbidden_prompt_strings: list[str] = Field(default_factory=list)
    required_section_strings: dict[str, list[str]] = Field(default_factory=dict)
    protected_sections: list[str] = Field(default_factory=lambda: ["system_policy", "current_user"])
    max_total_chars: int | None = None
    require_budget_degradation: bool = False
    require_evidence_section: bool = False
    require_tool_observation_refs: bool = False


class ContextEvalCase(BaseModel):
    case_id: str
    message: str
    transcript_context: str = ""
    memory_context: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    expected: ContextEvalExpected
    total_char_budget: int = 9000
    section_budgets: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def validate_case(self) -> "ContextEvalCase":
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.message.strip():
            raise ValueError("message must be non-empty")
        return self


class ContextStressCase(BaseModel):
    case_id: str
    category: str
    turns: list[AgentEvalTurn]
    expected: ContextEvalExpected
    notes: str = ""

    @model_validator(mode="after")
    def validate_case(self) -> "ContextStressCase":
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.turns:
            raise ValueError("turns must be non-empty")
        return self


class ContextDebugSection(BaseModel):
    name: str
    source: str = "runtime"
    priority: int = 0
    chars: int = 0
    token_estimate: int = 0
    char_budget: int | None = None
    truncated: bool = False
    content: str = ""
    preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextDebugReport(BaseModel):
    rendered_chars: int = 0
    section_count: int = 0
    section_names: list[str] = Field(default_factory=list)
    sections: list[ContextDebugSection] = Field(default_factory=list)
    budget_report: dict[str, Any] = Field(default_factory=dict)
    read_plan: dict[str, Any] = Field(default_factory=dict)
    governed_context_preview: str = ""
    trace_context_events: list[dict[str, Any]] = Field(default_factory=list)


class ContextEvalResult(BaseModel):
    case_id: str
    case_type: str
    category: str = ""
    passed: bool
    failures: list[str] = Field(default_factory=list)
    required_sections_ok: bool = True
    forbidden_sections_ok: bool = True
    fact_retention_ok: bool = True
    tool_leak_free: bool = True
    evidence_raw_ok: bool = True
    budget_policy_ok: bool = True
    llm_read_plan_used: bool = False
    llm_compactor_used: bool = False
    llm_span_selector_used: bool = False
    debug_report: ContextDebugReport
    final_answer: str = ""


class ContextEvalReport(BaseModel):
    runtime_profile: str = "custom"
    context_profile: str = "unknown"
    context_governance_enabled: bool = True
    llm_context_components_enabled: dict[str, bool] = Field(default_factory=dict)
    mock_skill_enabled: bool = False
    mock_tool_usage_count: int = 0
    mock_tool_names: list[str] = Field(default_factory=list)
    eval_runtime_matches_production: bool = False
    production_mismatch_reasons: list[str] = Field(default_factory=list)
    total: int
    passed: int
    pass_rate: float
    required_section_accuracy: float
    forbidden_section_accuracy: float
    fact_retention_accuracy: float
    tool_leak_free_rate: float
    evidence_raw_accuracy: float
    budget_policy_accuracy: float
    llm_read_plan_usage_rate: float
    llm_compactor_usage_rate: float
    llm_span_selector_usage_rate: float
    avg_rendered_chars: float
    results: list[ContextEvalResult] = Field(default_factory=list)


def build_context_eval_dataset() -> list[ContextEvalCase]:
    raw_tool_blob = "RAW_TOOL_SECRET_" + ("X" * 240)
    return [
        ContextEvalCase(
            case_id="ctx_product_long_history",
            message="还是 E2，下一步怎么处理？",
            transcript_context="\n".join(
                [
                    "user: 我的 AC900 显示 E2。",
                    "assistant: E2 通常和滤网堵塞有关。",
                    "user: 我已经断电重启过一次。",
                    "assistant: 还需要结合手册步骤继续排查。",
                    "user: 型号就是 AC900 空气净化器。",
                    "assistant: 记住了，是 AC900。",
                ]
            ),
            memory_context="[Memory View]\nsession_id: s1\nactive_product:\n- display_name: AC900 空气净化器",
            evidence=[
                Evidence(
                    evidence_id="ev-ac900-e2",
                    source="manual",
                    text="AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源并清洁滤网，确认风道无遮挡后再重新启动。",
                    confidence=0.95,
                    payload={"manual_name": "空气净化器手册", "page": 12, "chunk_id": "ac900-e2"},
                )
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "evidence", "current_user", "runtime"],
                required_prompt_strings=["AC900", "E2"],
                required_section_strings={
                    "evidence": ["关闭电源并清洁滤网"],
                    "memory": ["AC900 空气净化器"],
                },
                require_evidence_section=True,
            ),
            notes="长对话商品问答必须保留产品事实和原文证据。",
        ),
        ContextEvalCase(
            case_id="ctx_chat_excludes_evidence",
            message="今天先随便聊聊，你介绍一下自己。",
            transcript_context="user: AC900 怎么清洁滤网？\nassistant: 可以继续帮你查手册。",
            memory_context="[Memory View]\nsession_id: s2\nactive_product:\n- display_name: AC900 空气净化器",
            evidence=[
                Evidence(evidence_id="ev-chat", source="manual", text="这段证据不应该进入闲聊 prompt。")
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "current_user", "runtime"],
                forbidden_sections=["evidence", "workflow", "tool_observations"],
                forbidden_prompt_strings=["这段证据不应该进入闲聊 prompt"],
            ),
            notes="闲聊不该带着 RAG 证据进窗口。",
        ),
        ContextEvalCase(
            case_id="ctx_workflow_tool_observation_no_raw_leak",
            message="我还是要退款，订单 O1001 上一轮已经查过了。",
            transcript_context="user: 我要退款。\nassistant: 请提供订单号。\nuser: 订单 O1001。",
            memory_context="[Memory View]\nsession_id: s3\nactive_issue:\n- issue_type: refund\n- summary: refund intake",
            tool_results=[
                {
                    "service_id": "order",
                    "tool_name": "query_order",
                    "ok": True,
                    "data": {
                        "summary": "订单 O1001 已签收，暂不满足自动退款条件。",
                        "order_id": "O1001",
                        "raw_detail": raw_tool_blob,
                    },
                    "raw": {"http_body": raw_tool_blob},
                }
            ],
            trace_events=[
                {
                    "stage": "workflow.decision",
                    "message": "refund workflow is active",
                    "payload": {"workflow_name": "refund_request", "workflow_status": "collecting", "intent": "refund"},
                }
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "workflow", "memory", "conversation", "tool_observations", "current_user", "runtime"],
                required_section_strings={"tool_observations": ["trace://", "order.query_order"]},
                forbidden_prompt_strings=[raw_tool_blob],
                require_tool_observation_refs=True,
            ),
            notes="tool raw payload 不能泄漏到 prompt，只能保留 ref 和摘要。",
        ),
        ContextEvalCase(
            case_id="ctx_budget_degradation",
            message="我要报修 AC900，电话 13800138000，还是刚才那个 E2 问题。",
            transcript_context="\n".join(f"user: 历史对话 {idx} AC900 E2" for idx in range(1, 15)),
            memory_context="[Memory View]\nactive_product:\n- display_name: AC900\nactive_issue:\n- summary: AC900 E2 报修\n- missing_info: ['address']\n" + ("关键记忆" * 80),
            tool_results=[
                {"service_id": "case-intake", "tool_name": "collect_case_intake", "ok": True, "data": {"summary": "已收集型号和电话。" * 60}}
            ],
            evidence=[
                Evidence(
                    evidence_id="ev-budget",
                    source="manual",
                    text="AC900 显示 E2 表示滤网堵塞。处理前关闭电源。若仍异常可申请售后检测。" * 40,
                    payload={"manual_name": "空气净化器手册", "page": 12},
                )
            ],
            trace_events=[
                {
                    "stage": "workflow.decision",
                    "message": "repair workflow is active",
                    "payload": {"workflow_name": "repair_request", "workflow_status": "collecting", "intent": "repair"},
                }
            ],
            total_char_budget=560,
            section_budgets={"current_user": 120, "system_policy": 140, "evidence": 180},
            expected=ContextEvalExpected(
                required_sections=["system_policy", "current_user"],
                required_prompt_strings=["13800138000", "E2"],
                max_total_chars=560,
                require_budget_degradation=True,
            ),
            notes="预算吃紧时优先保留核心 section，并给出明确降级记录。",
        ),
        ContextEvalCase(
            case_id="ctx_evidence_raw_excerpt",
            message="AC900 滤网要怎么清洁？",
            evidence=[
                Evidence(
                    evidence_id="ev-clean",
                    source="manual",
                    text="开始清洁前请先关闭电源。取出滤网后使用软刷或低压吸尘器清洁表面灰尘，切勿水洗或暴晒。",
                    confidence=0.9,
                    payload={"manual_name": "空气净化器手册", "page": 18, "chunk_id": "filter-clean"},
                )
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "evidence", "current_user", "runtime"],
                required_section_strings={"evidence": ["切勿水洗或暴晒", "raw_excerpt"]},
                require_evidence_section=True,
            ),
            notes="evidence 侧必须保留原文片段，而不是自由总结。",
        ),
    ]


def build_context_stress_dataset() -> list[ContextStressCase]:
    return [
        ContextStressCase(
            case_id="stress_product_followup",
            category="product_support",
            turns=[
                AgentEvalTurn(message="AC900 显示 E2 怎么处理？"),
                AgentEvalTurn(message="我已经断电过了，继续说下一步。"),
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "evidence", "current_user", "runtime"],
                required_prompt_strings=["E2"],
                require_evidence_section=True,
            ),
            notes="商品问答多轮下 evidence 和 conversation 需要同时在场。",
        ),
        ContextStressCase(
            case_id="stress_case_intake_repair",
            category="case_intake",
            turns=[
                AgentEvalTurn(message="我要报修，机器坏了。"),
                AgentEvalTurn(message="型号 AC900，电话 13800138000。"),
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "tool_observations", "current_user", "runtime"],
                required_prompt_strings=["AC900", "13800138000"],
                require_tool_observation_refs=True,
            ),
            notes="报修多轮要保留已收集槽位和上一轮 tool observation。",
        ),
        ContextStressCase(
            case_id="stress_refund_followup",
            category="refund",
            turns=[
                AgentEvalTurn(message="我要退款。"),
                AgentEvalTurn(message="订单号 O1001，电话 13800138000。"),
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "tool_observations", "current_user", "runtime"],
                required_prompt_strings=["O1001", "13800138000"],
                require_tool_observation_refs=True,
            ),
            notes="退款链路要验证 tool raw 未泄漏且 budget 保持可解释。",
        ),
        ContextStressCase(
            case_id="stress_composite_handoff",
            category="composite",
            turns=[
                AgentEvalTurn(message="AC900 显示 E2，我还想直接退款。"),
                AgentEvalTurn(message="如果不能退就转人工。"),
            ],
            expected=ContextEvalExpected(
                required_sections=["system_policy", "memory", "conversation", "current_user", "runtime"],
                required_prompt_strings=["退款", "转人工"],
            ),
            notes="复合意图场景下，context 至少要保住当前诉求和历史连续性。",
        ),
    ]


async def run_context_eval(
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    cases: list[ContextEvalCase] | None = None,
    context_runtime: ContextRuntime | None = None,
    runtime_profile: str | EvalRuntimeProfile = EvalRuntimeProfile.DETERMINISTIC,
    show_progress: bool = False,
) -> ContextEvalReport:
    cases = cases or build_context_eval_dataset()
    profile = coerce_runtime_profile(runtime_profile)
    profile_audit = _context_profile_audit(profile, context_runtime=context_runtime)
    if context_runtime is None:
        context_runtime = build_profiled_eval_runtime(
            runtime_profile=profile,
            manual_dir="/Users/nikonzhang/compeletion/手册",
            use_real_llm=profile == EvalRuntimeProfile.PRODUCTION_LIKE,
        ).runtime.context_governance.context_runtime
    run_dir = Path(output_dir) / (run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[ContextEvalResult] = []
    debug_rows: list[dict[str, Any]] = []

    total = len(cases)
    if show_progress:
        _print_progress("context", 0, total, 0, 0, "starting")
    for index, case in enumerate(cases, start=1):
        runtime = _case_runtime(case, context_runtime)
        result = await _run_context_case(case, runtime=runtime)
        results.append(result)
        debug_rows.append(
            {
                "case_id": case.case_id,
                "case_type": "context",
                "notes": case.notes,
                "debug_report": result.debug_report.model_dump(),
                "failures": result.failures,
            }
        )
        if show_progress:
            _print_progress(
                "context",
                index,
                total,
                sum(1 for item in results if item.passed),
                sum(1 for item in results if not item.passed),
                case.case_id,
            )
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    report = _aggregate_context_results(results, profile_audit=profile_audit)
    (run_dir / "results.jsonl").write_text(
        "\n".join(result.model_dump_json() for result in results) + "\n",
        encoding="utf-8",
    )
    (run_dir / "context_debug.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in debug_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(report.model_dump_json(indent=2, exclude={"results"}), encoding="utf-8")
    (run_dir / "metrics.md").write_text(_render_context_markdown(report, title="nikon0 Context Eval Report"), encoding="utf-8")
    return report


async def run_context_stress_eval(
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    cases: list[ContextStressCase] | None = None,
    runtime: AgentRuntime | None = None,
    manual_dir: str | Path = "/Users/nikonzhang/compeletion/手册",
    use_real_llm: bool = True,
    local_rag: bool = False,
    runtime_profile: str | EvalRuntimeProfile = EvalRuntimeProfile.PRODUCTION_LIKE,
    mock_case_intake_tool: bool | None = None,
    show_progress: bool = False,
) -> ContextEvalReport:
    cases = cases or build_context_stress_dataset()
    profile = coerce_runtime_profile(runtime_profile)
    if runtime is None:
        profiled = build_profiled_eval_runtime(
            runtime_profile=profile,
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
    results: list[ContextEvalResult] = []
    debug_rows: list[dict[str, Any]] = []

    total = len(cases)
    if show_progress:
        _print_progress("stress", 0, total, 0, 0, "starting")
    for index, case in enumerate(cases, start=1):
        result = await _run_context_stress_case(case, runtime=runtime)
        results.append(result)
        debug_rows.append(
            {
                "case_id": case.case_id,
                "case_type": "stress",
                "category": case.category,
                "notes": case.notes,
                "debug_report": result.debug_report.model_dump(),
                "final_answer": result.final_answer,
                "failures": result.failures,
            }
        )
        if show_progress:
            _print_progress(
                "stress",
                index,
                total,
                sum(1 for item in results if item.passed),
                sum(1 for item in results if not item.passed),
                case.case_id,
            )
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    report = _aggregate_context_results(results, profile_audit=profile_audit)
    (run_dir / "results.jsonl").write_text(
        "\n".join(result.model_dump_json() for result in results) + "\n",
        encoding="utf-8",
    )
    (run_dir / "context_debug.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in debug_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(report.model_dump_json(indent=2, exclude={"results"}), encoding="utf-8")
    (run_dir / "metrics.md").write_text(
        _render_context_markdown(report, title="nikon0 Context Stress Report"),
        encoding="utf-8",
    )
    return report


async def _run_context_case(case: ContextEvalCase, *, runtime: ContextRuntime) -> ContextEvalResult:
    trace = ExecutionTrace(trace_id=f"trace-{case.case_id}", session_id=f"ctx-{case.case_id}", user_message=case.message)
    for event in case.trace_events:
        trace.add_event(
            str(event.get("stage") or "custom"),
            str(event.get("message") or ""),
            **(event.get("payload") if isinstance(event.get("payload"), dict) else {}),
        )
    context = AgentContext(
        request=AgentRequest(session_id=f"ctx-{case.case_id}", message=case.message),
        transcript_context=case.transcript_context,
        memory_context=case.memory_context,
        evidence_context=list(case.evidence),
        tool_results=[dict(item) for item in case.tool_results],
        trace=trace,
    )
    governed = await ContextGovernance(context_runtime=runtime).agovern(context)
    debug = _context_debug_from_context(governed)
    return _score_context_result(
        case_id=case.case_id,
        case_type="context",
        category="context",
        expected=case.expected,
        debug_report=debug,
    )


async def _run_context_stress_case(case: ContextStressCase, *, runtime: AgentRuntime) -> ContextEvalResult:
    response: AgentResponse | None = None
    session_id = f"context-stress-{case.case_id}"
    for turn in case.turns:
        response = await runtime.run(
            AgentRequest(
                session_id=session_id,
                message=turn.message,
                images=turn.images,
                metadata=turn.metadata,
            )
        )
    assert response is not None
    debug = _context_debug_from_response(response)
    result = _score_context_result(
        case_id=case.case_id,
        case_type="stress",
        category=case.category,
        expected=case.expected,
        debug_report=debug,
        final_answer=response.answer,
    )
    return result


def _score_context_result(
    *,
    case_id: str,
    case_type: str,
    category: str,
    expected: ContextEvalExpected,
    debug_report: ContextDebugReport,
    final_answer: str = "",
) -> ContextEvalResult:
    failures: list[str] = []
    section_names = set(debug_report.section_names)
    required_sections_ok = all(section in section_names for section in expected.required_sections)
    if not required_sections_ok:
        missing = [section for section in expected.required_sections if section not in section_names]
        failures.append(f"missing required sections: {', '.join(missing)}")
    forbidden_sections_ok = all(section not in section_names for section in expected.forbidden_sections)
    if not forbidden_sections_ok:
        present = [section for section in expected.forbidden_sections if section in section_names]
        failures.append(f"forbidden sections present: {', '.join(present)}")

    prompt_text = "\n".join(section.content for section in debug_report.sections)
    fact_retention_ok = True
    for needle in expected.required_prompt_strings:
        if needle not in prompt_text:
            fact_retention_ok = False
            failures.append(f"required prompt string missing: {needle}")
    for section_name, needles in expected.required_section_strings.items():
        section = next((item for item in debug_report.sections if item.name == section_name), None)
        if section is None:
            fact_retention_ok = False
            failures.append(f"required section missing for strings: {section_name}")
            continue
        for needle in needles:
            if needle not in section.content:
                fact_retention_ok = False
                failures.append(f"section {section_name} missing string: {needle}")

    tool_leak_free = True
    for needle in expected.forbidden_prompt_strings:
        if needle and needle in prompt_text:
            tool_leak_free = False
            failures.append(f"forbidden prompt string leaked: {needle[:40]}")

    evidence_raw_ok = True
    if expected.require_evidence_section:
        evidence = next((item for item in debug_report.sections if item.name == "evidence"), None)
        if evidence is None or "raw_excerpt" not in evidence.content:
            evidence_raw_ok = False
            failures.append("evidence section missing raw_excerpt payload")
    if expected.require_tool_observation_refs:
        observations = next((item for item in debug_report.sections if item.name == "tool_observations"), None)
        if observations is None or "trace://" not in observations.content:
            tool_leak_free = False
            failures.append("tool observations missing trace ref")

    budget_policy_ok = True
    report = debug_report.budget_report
    used_chars = int(report.get("used_chars") or 0)
    max_total_chars = expected.max_total_chars or int(report.get("total_budget") or 0)
    if max_total_chars and used_chars > max_total_chars:
        budget_policy_ok = False
        failures.append(f"budget exceeded: used={used_chars} limit={max_total_chars}")
    for protected in expected.protected_sections:
        if protected not in section_names:
            budget_policy_ok = False
            failures.append(f"protected section dropped: {protected}")
    if expected.require_budget_degradation and not report.get("degraded_sections"):
        budget_policy_ok = False
        failures.append("expected degradation did not happen")

    read_plan = debug_report.read_plan
    llm_read_plan_used = str(read_plan.get("source") or "") == "llm"
    llm_compactor_used = any(
        event.get("stage") == "context.conversation_compact" and event.get("payload", {}).get("source") == "llm"
        for event in debug_report.trace_context_events
    )
    llm_span_selector_used = any(
        event.get("stage") == "context.evidence_pack" and event.get("payload", {}).get("source") == "llm_span_selector"
        for event in debug_report.trace_context_events
    )

    return ContextEvalResult(
        case_id=case_id,
        case_type=case_type,
        category=category,
        passed=not failures,
        failures=failures,
        required_sections_ok=required_sections_ok,
        forbidden_sections_ok=forbidden_sections_ok,
        fact_retention_ok=fact_retention_ok,
        tool_leak_free=tool_leak_free,
        evidence_raw_ok=evidence_raw_ok,
        budget_policy_ok=budget_policy_ok,
        llm_read_plan_used=llm_read_plan_used,
        llm_compactor_used=llm_compactor_used,
        llm_span_selector_used=llm_span_selector_used,
        debug_report=debug_report,
        final_answer=final_answer,
    )


def _aggregate_context_results(
    results: list[ContextEvalResult],
    *,
    profile_audit: RuntimeProfileAudit,
) -> ContextEvalReport:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    return ContextEvalReport(
        runtime_profile=profile_audit.runtime_profile,
        context_profile=profile_audit.context_profile,
        context_governance_enabled=profile_audit.context_governance_enabled,
        llm_context_components_enabled=profile_audit.llm_context_components_enabled,
        mock_skill_enabled=profile_audit.mock_skill_enabled,
        mock_tool_usage_count=0,
        mock_tool_names=profile_audit.mock_tool_names,
        eval_runtime_matches_production=profile_audit.eval_runtime_matches_production,
        production_mismatch_reasons=profile_audit.production_mismatch_reasons,
        total=total,
        passed=passed,
        pass_rate=_ratio(passed, total),
        required_section_accuracy=_ratio(sum(1 for item in results if item.required_sections_ok), total),
        forbidden_section_accuracy=_ratio(sum(1 for item in results if item.forbidden_sections_ok), total),
        fact_retention_accuracy=_ratio(sum(1 for item in results if item.fact_retention_ok), total),
        tool_leak_free_rate=_ratio(sum(1 for item in results if item.tool_leak_free), total),
        evidence_raw_accuracy=_ratio(sum(1 for item in results if item.evidence_raw_ok), total),
        budget_policy_accuracy=_ratio(sum(1 for item in results if item.budget_policy_ok), total),
        llm_read_plan_usage_rate=_ratio(sum(1 for item in results if item.llm_read_plan_used), total),
        llm_compactor_usage_rate=_ratio(sum(1 for item in results if item.llm_compactor_used), total),
        llm_span_selector_usage_rate=_ratio(sum(1 for item in results if item.llm_span_selector_used), total),
        avg_rendered_chars=round(mean([item.debug_report.rendered_chars for item in results]), 2) if results else 0.0,
        results=results,
    )


def _context_debug_from_context(context: AgentContext) -> ContextDebugReport:
    pack = context.context_pack
    read_plan_event = next(
        (event for event in reversed(context.trace.events) if event.stage == "context.read_plan"),
        None,
    )
    sections: list[ContextDebugSection] = []
    if pack is not None:
        for section in pack.sections:
            sections.append(
                ContextDebugSection(
                    name=section.name,
                    source=section.source,
                    priority=section.priority,
                    chars=len(section.content),
                    token_estimate=section.token_estimate,
                    char_budget=section.char_budget,
                    truncated=section.truncated,
                    content=section.content,
                    preview=section.content[:240],
                    metadata=dict(section.metadata),
                )
            )
    return ContextDebugReport(
        rendered_chars=len(context.governed_context or ""),
        section_count=len(sections),
        section_names=[section.name for section in sections],
        sections=sections,
        budget_report=pack.budget_report.model_dump() if pack is not None else {},
        read_plan={
            "source": read_plan_event.payload.get("source") if read_plan_event is not None else "",
            "confidence": read_plan_event.payload.get("confidence") if read_plan_event is not None else 0,
            "included_sections": read_plan_event.payload.get("included_sections") if read_plan_event is not None else [],
            "reasons": read_plan_event.payload.get("reasons") if read_plan_event is not None else {},
        },
        governed_context_preview=(context.governed_context or "")[:1000],
        trace_context_events=[event.model_dump() for event in context.trace.events if event.stage.startswith("context.")],
    )


def _context_debug_from_response(response: AgentResponse) -> ContextDebugReport:
    payload = response.debug.get("context_debug") or {}
    trace = response.debug.get("trace") or {}
    read_plan_event = next(
        (
            event
            for event in reversed(trace.get("events") or [])
            if isinstance(event, dict) and event.get("stage") == "context.read_plan"
        ),
        None,
    )
    return ContextDebugReport(
        rendered_chars=int(payload.get("rendered_chars") or 0),
        section_count=int(payload.get("section_count") or 0),
        section_names=[str(item) for item in payload.get("section_names") or []],
        sections=[ContextDebugSection.model_validate(item) for item in payload.get("sections") or []],
        budget_report=payload.get("budget_report") or {},
        read_plan={
            "source": (read_plan_event or {}).get("payload", {}).get("source", ""),
            "confidence": (read_plan_event or {}).get("payload", {}).get("confidence", 0),
            "included_sections": (read_plan_event or {}).get("payload", {}).get("included_sections", []),
            "reasons": (read_plan_event or {}).get("payload", {}).get("reasons", {}),
        },
        governed_context_preview=str(payload.get("governed_context_preview") or ""),
        trace_context_events=[
            event for event in trace.get("events") or [] if isinstance(event, dict) and str(event.get("stage", "")).startswith("context.")
        ],
    )


def _context_profile_audit(
    runtime_profile: str | EvalRuntimeProfile,
    *,
    context_runtime: ContextRuntime | None,
) -> RuntimeProfileAudit:
    profile = coerce_runtime_profile(runtime_profile)
    if context_runtime is not None:
        flags = {
            "read_planner": context_runtime.read_planner.__class__.__name__.startswith("Llm"),
            "conversation_compactor": context_runtime.conversation_compactor.__class__.__name__.startswith("Llm"),
            "evidence_span_selector": bool(getattr(context_runtime.evidence_manager, "span_selector", None)),
        }
        return RuntimeProfileAudit(
            runtime_profile="custom_context",
            context_profile="production_like_llm" if any(flags.values()) else "deterministic",
            context_governance_enabled=True,
            llm_context_components_enabled=flags,
            eval_runtime_matches_production=False,
            production_mismatch_reasons=["custom context_runtime was injected"],
        )
    profiled = build_profiled_eval_runtime(
        runtime_profile=profile,
        manual_dir="/Users/nikonzhang/compeletion/手册",
        use_real_llm=profile == EvalRuntimeProfile.PRODUCTION_LIKE,
    )
    return profiled.audit


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


def _case_runtime(case: ContextEvalCase, base_runtime: ContextRuntime | None) -> ContextRuntime:
    template = base_runtime or ContextRuntime()
    merged_section_budgets = {**dict(template.section_budgets), **dict(case.section_budgets)}
    return ContextRuntime(
        total_char_budget=case.total_char_budget,
        section_budgets=merged_section_budgets,
        budgeter=None,
        read_planner=template.read_planner,
        conversation_compactor=template.conversation_compactor,
        evidence_manager=template.evidence_manager,
        tool_observation_manager=template.tool_observation_manager,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _render_context_markdown(report: ContextEvalReport, *, title: str) -> str:
    lines = [
        f"# {title}",
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
        "",
        f"- total: {report.total}",
        f"- passed: {report.passed}",
        f"- pass_rate: {report.pass_rate:.4f}",
        f"- required_section_accuracy: {report.required_section_accuracy:.4f}",
        f"- forbidden_section_accuracy: {report.forbidden_section_accuracy:.4f}",
        f"- fact_retention_accuracy: {report.fact_retention_accuracy:.4f}",
        f"- tool_leak_free_rate: {report.tool_leak_free_rate:.4f}",
        f"- evidence_raw_accuracy: {report.evidence_raw_accuracy:.4f}",
        f"- budget_policy_accuracy: {report.budget_policy_accuracy:.4f}",
        f"- llm_read_plan_usage_rate: {report.llm_read_plan_usage_rate:.4f}",
        f"- llm_compactor_usage_rate: {report.llm_compactor_usage_rate:.4f}",
        f"- llm_span_selector_usage_rate: {report.llm_span_selector_usage_rate:.4f}",
        f"- avg_rendered_chars: {report.avg_rendered_chars:.2f}",
        "",
        "| case_id | type | passed | sections | failures |",
        "|---|---|---:|---|---|",
    ]
    for result in report.results:
        failures = "; ".join(result.failures[:3])
        sections = ",".join(result.debug_report.section_names)
        lines.append(f"| {result.case_id} | {result.case_type} | {int(result.passed)} | {sections} | {failures} |")
    return "\n".join(lines) + "\n"


def _print_progress(kind: str, current: int, total: int, passed: int, failed: int, label: str) -> None:
    width = 28
    ratio = current / total if total else 1.0
    filled = min(width, max(0, int(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[{kind}] [{bar}] {current}/{total} pass={passed} fail={failed} current={label}")
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nikon0 context evaluation.")
    parser.add_argument(
        "--mode",
        choices=["context", "stress"],
        default="context",
        help="Run direct context module eval or end-to-end stress eval.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/nikonzhang/compeletion/nikon0/eval/reports/context",
        help="Directory where report folders will be written.",
    )
    parser.add_argument("--run-id", default=None, help="Report subdirectory name.")
    parser.add_argument(
        "--manual-dir",
        default="/Users/nikonzhang/compeletion/手册",
        help="Manual directory used by stress eval runtime.",
    )
    parser.add_argument("--progress", action="store_true", help="Show an in-terminal progress bar.")
    parser.add_argument(
        "--runtime-profile",
        default=EvalRuntimeProfile.PRODUCTION_LIKE.value,
        choices=[item.value for item in EvalRuntimeProfile],
        help="Runtime profile used by context stress eval, and context profile source for direct eval.",
    )
    parser.add_argument("--no-real-llm", action="store_true", help="Disable real LLM calls during stress eval.")
    parser.add_argument("--local-rag", action="store_true", help="Use local structured RAG during stress eval.")
    parser.add_argument("--mock-case-intake-tool", action="store_true", help="Use deterministic eval-time case-intake tools.")
    args = parser.parse_args()

    if args.mode == "context":
        report = asyncio.run(
            run_context_eval(
                output_dir=args.output_dir,
                run_id=args.run_id,
                runtime_profile=coerce_runtime_profile(args.runtime_profile),
                show_progress=args.progress,
            )
        )
    else:
        report = asyncio.run(
            run_context_stress_eval(
                output_dir=args.output_dir,
                run_id=args.run_id,
                manual_dir=args.manual_dir,
                use_real_llm=not args.no_real_llm,
                local_rag=args.local_rag,
                runtime_profile=coerce_runtime_profile(args.runtime_profile),
                mock_case_intake_tool=True if args.mock_case_intake_tool else None,
                show_progress=args.progress,
            )
        )
    print(f"total={report.total} passed={report.passed} pass_rate={report.pass_rate:.4f}")
    print(f"runtime_profile={report.runtime_profile}")
    print(f"context_profile={report.context_profile}")
    print(f"eval_runtime_matches_production={report.eval_runtime_matches_production}")
    print(f"mock_tool_usage_count={report.mock_tool_usage_count}")
    print(f"fact_retention_accuracy={report.fact_retention_accuracy:.4f}")
    print(f"tool_leak_free_rate={report.tool_leak_free_rate:.4f}")
    print(f"evidence_raw_accuracy={report.evidence_raw_accuracy:.4f}")
    print(f"budget_policy_accuracy={report.budget_policy_accuracy:.4f}")


__all__ = [
    "ContextEvalCase",
    "ContextStressCase",
    "ContextEvalExpected",
    "ContextEvalResult",
    "ContextEvalReport",
    "ContextDebugReport",
    "build_context_eval_dataset",
    "build_context_stress_dataset",
    "run_context_eval",
    "run_context_stress_eval",
]


if __name__ == "__main__":
    main()
