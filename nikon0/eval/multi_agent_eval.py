"""Trace-first evaluation harness for nikon0 multi-agent executions."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from nikon0.agent.runtime import build_default_runtime
from nikon0.app.schemas.agent import AgentRequest, AgentResponse
from nikon0.tools.mcp_gateway import McpGatewayTool
from nikon0.tools.runtime import ToolRegistry


class MultiAgentExpected(BaseModel):
    agent_stages: list[str] = Field(default_factory=list)
    required_trace_stages: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    handoff_required: bool | None = None
    approval_required: bool | None = None
    minimum_evidence_count: int = 0
    persistence_required: bool = False
    expected_handoff: dict[str, Any] = Field(default_factory=dict)
    expected_memory_commits: list[str] = Field(default_factory=list)
    expected_persistence: dict[str, Any] = Field(default_factory=dict)
    expected_safety: dict[str, Any] = Field(default_factory=dict)
    expected_evidence: dict[str, Any] = Field(default_factory=dict)
    answer_rubric: list[str] = Field(default_factory=list)


class MultiAgentEvalCase(BaseModel):
    case_id: str
    turns: list[str]
    expected: MultiAgentExpected
    category: str = ""
    source_manuals: list[str] = Field(default_factory=list)
    notes: str = ""


class MultiAgentEvalResult(BaseModel):
    case_id: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    agent_stages: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    handoff_required: bool = False
    approval_required: bool = False
    evidence_count: int = 0
    persistence_observed: bool = False
    handoff: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""


def load_cases(path: str | Path) -> list[MultiAgentEvalCase]:
    return [
        MultiAgentEvalCase.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_multi_agent_response(case: MultiAgentEvalCase, response: AgentResponse) -> MultiAgentEvalResult:
    debug = response.debug or {}
    multi = debug.get("multi_agent") or {}
    trace = debug.get("trace") or {}
    stages = [str(item) for item in multi.get("agent_stages") or []]
    events = [item for item in trace.get("events") or [] if isinstance(item, dict)]
    event_stages = {str(item.get("stage") or "") for item in events}
    actions = [item.model_dump() for item in response.actions]
    tool_names = [str(item.get("name")) for item in actions if item.get("kind") == "tool"]
    evidence_count = sum(int(item.get("evidence_count") or 0) for item in trace.get("knowledge_calls") or [])
    governance = debug.get("memory_governance") or {}
    persistence_observed = bool((governance.get("store_profile") or {}).get("mysql_ok"))
    support_handoff = dict(multi.get("support_handoff") or {})
    failures: list[str] = []
    if case.expected.agent_stages and stages != case.expected.agent_stages:
        failures.append(f"agent stages expected {case.expected.agent_stages}, got {stages}")
    missing_trace = [stage for stage in case.expected.required_trace_stages if stage not in event_stages]
    if missing_trace:
        failures.append(f"missing trace stages: {', '.join(missing_trace)}")
    missing_tools = [tool for tool in case.expected.required_tools if tool not in tool_names]
    if missing_tools:
        failures.append(f"missing tools: {', '.join(missing_tools)}")
    handoff_required = any(item.get("kind") == "handoff" for item in actions)
    approval = any(item.get("kind") == "approval" for item in actions)
    if case.expected.handoff_required is not None and handoff_required != case.expected.handoff_required:
        failures.append(f"handoff expected {case.expected.handoff_required}, got {handoff_required}")
    if case.expected.approval_required is not None and approval != case.expected.approval_required:
        failures.append(f"approval expected {case.expected.approval_required}, got {approval}")
    if evidence_count < case.expected.minimum_evidence_count:
        failures.append(f"evidence expected >= {case.expected.minimum_evidence_count}, got {evidence_count}")
    if case.expected.persistence_required and not persistence_observed:
        failures.append("persistent MySQL memory store was not observed")
    for key, expected_value in case.expected.expected_handoff.items():
        if _nested_value(support_handoff, key) != expected_value:
            failures.append(f"handoff {key} expected {expected_value!r}, got {_nested_value(support_handoff, key)!r}")
    for stage in case.expected.expected_memory_commits:
        if stage not in event_stages:
            failures.append(f"missing memory commit trace: {stage}")
    for key, expected_value in case.expected.expected_safety.items():
        actual = {"handoff_required": handoff_required, "approval_required": approval}.get(key)
        if actual != expected_value:
            failures.append(f"safety {key} expected {expected_value!r}, got {actual!r}")
    expected_manuals = set(str(item) for item in case.expected.expected_evidence.get("manual_names", []) if str(item))
    if expected_manuals:
        actual_manuals = {
            str(item.get("product_resolution", {}).get("manual_names", [""])[0])
            for item in trace.get("knowledge_calls") or []
            if isinstance(item, dict) and isinstance(item.get("product_resolution"), dict)
        }
        if not expected_manuals.intersection(actual_manuals):
            failures.append(f"evidence manuals expected one of {sorted(expected_manuals)}, got {sorted(actual_manuals)}")
    for fact in case.expected.answer_rubric:
        if fact and fact.lower() not in response.answer.lower():
            failures.append(f"answer rubric fact missing: {fact}")
    return MultiAgentEvalResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        agent_stages=stages,
        trace_ids=[response.trace_id],
        tool_names=tool_names,
        handoff_required=handoff_required,
        approval_required=approval,
        evidence_count=evidence_count,
        persistence_observed=persistence_observed,
        handoff=support_handoff,
        answer=response.answer,
    )


def _nested_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


async def run_dataset(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    progress: bool = False,
    mcp_sandbox_endpoint: str = "",
    allow_non_sandbox_mcp: bool = False,
) -> dict[str, Any]:
    cases = load_cases(dataset)
    run_id = uuid4().hex[:12]
    root = Path(output_dir) / run_id
    traces_dir = root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_default_runtime()
    if runtime.multi_agent_coordinator is None:
        raise RuntimeError("multi-agent runtime is not enabled; set NIKON0_MULTI_AGENT_ENABLED=true")
    needs_service = any(case.category in {"service", "composite", "memory", "resilience"} for case in cases)
    if needs_service and not mcp_sandbox_endpoint and not allow_non_sandbox_mcp:
        raise RuntimeError("service eval requires --mcp-sandbox-endpoint; refusing to call a non-sandbox MCP target")
    if mcp_sandbox_endpoint:
        _replace_case_intake_tools_with_sandbox(runtime, mcp_sandbox_endpoint)
    results: list[MultiAgentEvalResult] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        responses: list[AgentResponse] = []
        session_id = f"eval:multiagent:{run_id}:{case.case_id}"
        for turn_index, message in enumerate(case.turns, start=1):
            response = await runtime.run(AgentRequest(session_id=session_id, user_id="eval-user", message=message))
            responses.append(response)
            (traces_dir / f"{case.case_id}-turn-{turn_index}.json").write_text(
                json.dumps(response.debug, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        result = score_multi_agent_response(case, responses[-1])
        result.trace_ids = [item.trace_id for item in responses]
        results.append(result)
        if progress:
            print(f"[{index}/{len(cases)}] {case.case_id}: {'PASS' if result.passed else 'FAIL'}", flush=True)
    report = {
        "run_id": run_id,
        "dataset": str(dataset),
        "total": len(results),
        "passed": sum(item.passed for item in results),
        "pass_rate": sum(item.passed for item in results) / len(results) if results else 0.0,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "mcp_mode": "sandbox" if mcp_sandbox_endpoint else "non_sandbox_explicit",
        "metrics": _metrics(cases, results),
        "results": [item.model_dump() for item in results],
    }
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "results.jsonl").write_text("\n".join(json.dumps(item.model_dump(), ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    (root / "report.md").write_text(
        f"# Multi-Agent Eval\n\n- run_id: `{run_id}`\n- total: {report['total']}\n- passed: {report['passed']}\n- pass_rate: {report['pass_rate']:.2%}\n",
        encoding="utf-8",
    )
    return report


def _metrics(cases: list[MultiAgentEvalCase], results: list[MultiAgentEvalResult]) -> dict[str, float]:
    pairs = list(zip(cases, results))
    total = len(pairs) or 1
    composite = [(case, result) for case, result in pairs if case.category == "composite"]
    business_general = [
        result
        for case, result in pairs
        if case.category != "general" and not result.agent_stages
    ]
    expected_tools = [
        (case, result) for case, result in pairs if case.expected.required_tools
    ]
    persisted = [result for case, result in pairs if case.expected.persistence_required]
    memory_events = [
        result for case, result in pairs if case.category in {"memory", "resilience"}
    ]
    return {
        "delegation_accuracy": sum(result.agent_stages == case.expected.agent_stages for case, result in pairs) / total,
        "unsupported_business_general_answer_rate": len(business_general) / total,
        "composite_order_accuracy": (
            sum(result.agent_stages == ["support", "service"] for _, result in composite) / len(composite)
            if composite else 0.0
        ),
        "tool_chain_completion_rate": (
            sum(all(tool in result.tool_names for tool in case.expected.required_tools) for case, result in expected_tools) / len(expected_tools)
            if expected_tools else 0.0
        ),
        "stage_specific_persistence_rate": (
            sum(result.persistence_observed for result in persisted) / len(persisted) if persisted else 0.0
        ),
        "trace_completeness_rate": sum(
            not any("missing trace stages" in failure for failure in result.failures) for _, result in pairs
        ) / total,
        "memory_case_pass_rate": sum(result.passed for result in memory_events) / len(memory_events) if memory_events else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nikon0/eval/datasets/multi_agent_core_60.jsonl")
    parser.add_argument("--output-dir", default="nikon0/eval/reports/multi-agent")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--mcp-sandbox-endpoint", default="")
    parser.add_argument("--allow-non-sandbox-mcp", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_dataset(
        args.dataset,
        args.output_dir,
        progress=args.progress,
        mcp_sandbox_endpoint=args.mcp_sandbox_endpoint,
        allow_non_sandbox_mcp=args.allow_non_sandbox_mcp,
    )), ensure_ascii=False, indent=2))


def _replace_case_intake_tools_with_sandbox(runtime, endpoint: str) -> None:
    """Keep local extract tool but route all external case operations to an explicit sandbox."""
    from app.services.mcp_gateway.client import McpGatewayClient

    client = McpGatewayClient(endpoint=endpoint)
    current = list(runtime.tool_runtime.registry._tools)  # Runtime-owned registry snapshot.
    retained = [
        tool
        for tool in current
        if tool.spec.service_id != "case-intake" or tool.spec.tool_name == "extract_case_slots"
    ]
    retained.extend(
        [
            McpGatewayTool(service_id="case-intake", tool_name="get_case_intake_status", risk_level="low", client=client),
            McpGatewayTool(service_id="case-intake", tool_name="collect_case_intake", risk_level="medium", client=client),
            McpGatewayTool(service_id="case-intake", tool_name="try_cancel_case_intake", risk_level="medium", client=client),
        ]
    )
    runtime.tool_runtime.registry = ToolRegistry(retained)


if __name__ == "__main__":
    main()
