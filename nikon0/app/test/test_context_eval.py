from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nikon0.agent.runtime import build_default_runtime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.context.runtime import ContextRuntime
from nikon0.eval.context_eval import (
    build_context_eval_dataset,
    build_context_stress_dataset,
    run_context_eval,
    run_context_stress_eval,
)
from nikon0.eval.runtime_profiles import EvalRuntimeProfile
from nikon0.eval.run_agent_eval import build_eval_runtime


def test_agent_runtime_debug_contains_context_debug_payload() -> None:
    runtime = build_eval_runtime(
        manual_dir="/Users/nikonzhang/compeletion/手册",
        use_real_llm=False,
        local_rag=True,
        runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
    )

    response = asyncio.run(runtime.run(AgentRequest(session_id="ctx-debug-s1", message="你好，介绍一下你自己。")))

    context_debug = response.debug["context_debug"]
    assert context_debug["section_count"] >= 3
    assert "current_user" in context_debug["section_names"]
    assert context_debug["budget_report"]["total_budget"] > 0


def test_context_eval_runner_outputs_metrics_and_debug_report(tmp_path: Path) -> None:
    cases = build_context_eval_dataset()[:3]

    report = asyncio.run(
        run_context_eval(
            output_dir=tmp_path,
            run_id="context-eval",
            cases=cases,
            context_runtime=ContextRuntime(),
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        )
    )

    run_dir = tmp_path / "context-eval"
    assert report.total == 3
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.md").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "context_debug.jsonl").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "tool_leak_free_rate" in metrics
    assert "budget_policy_accuracy" in metrics
    assert metrics["runtime_profile"] == "custom_context"
    assert "context_profile" in metrics
    debug_rows = [
        json.loads(line)
        for line in (run_dir / "context_debug.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert debug_rows[0]["debug_report"]["section_names"]


def test_context_eval_workflow_case_blocks_raw_tool_payload_leak(tmp_path: Path) -> None:
    case = next(item for item in build_context_eval_dataset() if item.case_id == "ctx_workflow_tool_observation_no_raw_leak")

    report = asyncio.run(
        run_context_eval(
            output_dir=tmp_path,
            run_id="single-case",
            cases=[case],
            context_runtime=ContextRuntime(),
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        )
    )

    assert report.total == 1
    assert report.tool_leak_free_rate == 1.0
    assert report.results[0].passed is True
    observations = next(section for section in report.results[0].debug_report.sections if section.name == "tool_observations")
    assert "trace://" in observations.content
    assert "RAW_TOOL_SECRET_" not in observations.content


def test_context_stress_runner_outputs_debug_report(tmp_path: Path) -> None:
    runtime = build_eval_runtime(
        manual_dir="/Users/nikonzhang/compeletion/手册",
        use_real_llm=False,
        local_rag=True,
        runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
    )

    report = asyncio.run(
        run_context_stress_eval(
            output_dir=tmp_path,
            run_id="stress-eval",
            cases=build_context_stress_dataset()[:1],
            runtime=runtime,
            use_real_llm=False,
            local_rag=True,
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        )
    )

    run_dir = tmp_path / "stress-eval"
    assert report.total == 1
    assert (run_dir / "context_debug.jsonl").exists()
    result = report.results[0]
    assert result.debug_report.section_names
    assert "current_user" in result.debug_report.section_names
