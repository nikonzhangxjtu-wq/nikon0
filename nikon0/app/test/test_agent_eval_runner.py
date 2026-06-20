from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nikon0.app.schemas.agent import AgentResponse
from nikon0.eval.agent_dataset import AgentEvalCase, ExpectedOutcome, build_high_quality_agent_dataset, write_jsonl_dataset
from nikon0.eval.runtime_profiles import EvalRuntimeProfile
from nikon0.eval.run_agent_eval import _score_case, build_eval_runtime, run_agent_eval
from nikon0.knowledge.runtime import EnterpriseRagBackend, StructuredManualBackend
from nikon0.skills.product_support import ProductSupportSkill


def test_high_quality_dataset_has_150_cases_and_manual_source() -> None:
    manual_dir = Path("/Users/nikonzhang/compeletion/手册")

    cases = build_high_quality_agent_dataset(manual_dir=manual_dir, target_size=150)

    assert len(cases) == 150
    assert len({case.case_id for case in cases}) == 150
    product_cases = [case for case in cases if case.category == "product_support"]
    assert len(product_cases) == 40
    assert all(case.metadata.get("manual_dir") == str(manual_dir) for case in product_cases)
    assert all(case.expected.min_evidence_count >= 1 for case in product_cases)


def test_agent_eval_runner_outputs_answers_metrics_and_failures(tmp_path) -> None:
    manual_dir = Path("/Users/nikonzhang/compeletion/手册")
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "reports"
    cases = build_high_quality_agent_dataset(manual_dir=manual_dir, target_size=8)
    write_jsonl_dataset(cases, dataset_path)

    report = asyncio.run(
        run_agent_eval(
            dataset_path=dataset_path,
            output_dir=output_dir,
            manual_dir=manual_dir,
            run_id="test-run",
            use_real_llm=False,
            local_rag=True,
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        )
    )

    run_dir = output_dir / "test-run"
    assert report.total == 8
    assert (run_dir / "answers.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.md").exists()
    assert (run_dir / "failures.jsonl").exists()
    answers = [
        json.loads(line)
        for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(answers) == 8
    assert all("answer" in item for item in answers)
    assert all("expected" in item for item in answers)
    assert all("golden_answer" in item for item in answers)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["total"] == 8
    assert "answer_constraint_accuracy" in metrics
    assert "fact_coverage_accuracy" in metrics
    assert "evidence_alignment_accuracy" in metrics
    assert "rag_backend_accuracy" in metrics
    assert metrics["runtime_profile"] == "deterministic"
    assert metrics["context_profile"] == "deterministic"
    assert metrics["mock_skill_enabled"] is False
    assert "mock_tool_usage_count" in metrics
    assert "context_governance_enabled" in metrics


def test_agent_eval_runner_records_multi_turn_answers(tmp_path) -> None:
    manual_dir = Path("/Users/nikonzhang/compeletion/手册")
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "reports"
    cases = [
        case
        for case in build_high_quality_agent_dataset(manual_dir=manual_dir, target_size=150)
        if len(case.turns) > 1
    ][:1]
    write_jsonl_dataset(cases, dataset_path)

    asyncio.run(
        run_agent_eval(
            dataset_path=dataset_path,
            output_dir=output_dir,
            manual_dir=manual_dir,
            run_id="multi-turn",
            use_real_llm=False,
            local_rag=True,
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        )
    )

    answer = json.loads((output_dir / "multi-turn" / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(answer["turn_results"]) >= 2
    assert all(item["answer"] for item in answer["turn_results"])


def test_agent_eval_runner_prints_progress(tmp_path, capsys) -> None:
    manual_dir = Path("/Users/nikonzhang/compeletion/手册")
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "reports"
    cases = build_high_quality_agent_dataset(manual_dir=manual_dir, target_size=2)
    write_jsonl_dataset(cases, dataset_path)

    asyncio.run(
        run_agent_eval(
            dataset_path=dataset_path,
            output_dir=output_dir,
            manual_dir=manual_dir,
            run_id="progress",
            use_real_llm=False,
            local_rag=True,
            runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
            show_progress=True,
        )
    )

    captured = capsys.readouterr()
    assert "[eval]" in captured.err
    assert "2/2" in captured.err


def test_eval_runtime_uses_real_llm_by_default(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_answer_generator", lambda *, use_real_llm: sentinel)
    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_selector", lambda *, enabled: None)

    runtime = build_eval_runtime(manual_dir="/missing", runtime_profile=EvalRuntimeProfile.DETERMINISTIC)

    assert runtime.answer_generator is sentinel


def test_eval_runtime_uses_enterprise_rag_by_default(monkeypatch) -> None:
    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_answer_generator", lambda *, use_real_llm: None)
    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_selector", lambda *, enabled: None)

    runtime = build_eval_runtime(manual_dir="/missing", runtime_profile=EvalRuntimeProfile.DETERMINISTIC)
    product_support = runtime.skill_registry.get("product_support")

    assert isinstance(product_support, ProductSupportSkill)
    assert isinstance(product_support.knowledge_runtime.backend, EnterpriseRagBackend)


def test_eval_runtime_keeps_local_rag_as_explicit_baseline(monkeypatch) -> None:
    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_answer_generator", lambda *, use_real_llm: None)
    monkeypatch.setattr("nikon0.eval.runtime_profiles._build_selector", lambda *, enabled: None)

    runtime = build_eval_runtime(manual_dir="/missing", local_rag=True, runtime_profile=EvalRuntimeProfile.DETERMINISTIC)
    product_support = runtime.skill_registry.get("product_support")

    assert isinstance(product_support, ProductSupportSkill)
    assert isinstance(product_support.knowledge_runtime.backend, StructuredManualBackend)


def test_product_support_scoring_uses_fact_coverage_and_evidence_alignment() -> None:
    case = AgentEvalCase(
        case_id="product-semantic-001",
        category="product_support",
        message="空气净化器滤网多久换？",
        expected=ExpectedOutcome(
            acceptable_skills=["product_support"],
            min_evidence_count=1,
            answer_must_contain=["6-12 个月"],
        ),
        golden_answer="手册建议每 6-12 个月更换一次滤网。",
        metadata={
            "source_manual": "/Users/nikonzhang/compeletion/手册/空气净化器手册.txt",
            "source_facts": ["每 6-12 个月更换一次滤网"],
        },
    )
    response = AgentResponse(
        answer="建议每六到十二个月更换一次空气净化器滤网。",
        trace_id="trace-product-semantic",
        debug={
            "skill_selection": {"selected_skill": "product_support", "source": "model"},
            "trace": {
                "knowledge_calls": [
                    {
                        "evidence_count": 1,
                        "backend_trace": [{"backend": "enterprise_rag", "ok": True}],
                    }
                ],
                "memory_updates": [
                    {
                        "key": "product_support",
                        "value": {"manual_names": ["空气净化器手册"]},
                    }
                ],
            },
        },
    )

    assert _score_case(case, response) == []
