"""Memory v4 golden conversation evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.manager import MemoryManagerV4
from app.services.memory.v4.reader import IssueReadPlanner
from app.services.memory.v4.store import InMemorySessionIssueMemoryStore


def _packet(case_id: str, idx: int, turn: dict[str, Any]) -> TurnEvidencePacket:
    return TurnEvidencePacket(
        session_id=case_id,
        user_id=None,
        turn_id=f"{case_id}_turn_{idx}",
        timestamp=float(idx),
        question=str(turn.get("user") or ""),
        answer=str(turn.get("assistant") or ""),
        route_domain_hint=str(turn.get("route_domain_hint") or "customer_service"),
        route_needs_rag=bool(turn.get("route_needs_rag", False)),
        branch_name=str(turn.get("branch_name") or "no_rag"),
        rag_context=str(turn.get("rag_context") or ""),
        branch_result=turn.get("branch_result"),
    )


def _active_values(thread, kind: str) -> list[str]:
    return [f.value for f in thread.facts.values() if f.kind == kind and f.status == "active"]


def _status_values(thread, kind: str, status: str) -> list[str]:
    return [f.value for f in thread.facts.values() if f.kind == kind and f.status == status]


def _find_thread(memory, product_model: str):
    for thread in memory.threads.values():
        if thread.product_model == product_model:
            return thread
    return None


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)
    case_id = str(case["case_id"])
    for idx, turn in enumerate(case.get("turns") or [], start=1):
        manager.observe_and_write(_packet(case_id, idx, turn))
    memory = manager.store.load(case_id)
    failures: list[str] = []
    expected_threads = case.get("expected_threads") or []
    if len(memory.threads) != len(expected_threads):
        failures.append(f"thread_count expected={len(expected_threads)} actual={len(memory.threads)}")
    for expected in expected_threads:
        product = expected.get("product_model")
        thread = _find_thread(memory, product)
        if thread is None:
            failures.append(f"missing thread product_model={product}")
            continue
        for kind, values in (expected.get("active_facts") or {}).items():
            actual = _active_values(thread, kind)
            for value in values:
                if value not in actual:
                    failures.append(f"{product} missing active {kind}={value}; actual={actual}")
        for kind, values in (expected.get("superseded_facts") or {}).items():
            actual = _status_values(thread, kind, "superseded")
            for value in values:
                if value not in actual:
                    failures.append(f"{product} missing superseded {kind}={value}; actual={actual}")
        for kind, values in (expected.get("rejected_facts") or {}).items():
            actual = _status_values(thread, kind, "rejected")
            for value in values:
                if value not in actual:
                    failures.append(f"{product} missing rejected {kind}={value}; actual={actual}")
        for fact in thread.facts.values():
            if fact.status == "active" and fact.evidence_ref_id not in thread.evidence_refs:
                failures.append(f"{product} fact without evidence_ref {fact.kind}={fact.value}")
    rendered = ""
    read_query = case.get("read_query")
    if read_query:
        request = IssueReadPlanner().plan(session_id=case_id, query=str(read_query))
        rendered = manager.read(request).render()
        for text in case.get("expected_render_contains") or []:
            if text not in rendered:
                failures.append(f"render missing {text!r}")
        for text in case.get("expected_render_not_contains") or []:
            if text in rendered:
                failures.append(f"render should not contain {text!r}")
    return {
        "case_id": case_id,
        "passed": not failures,
        "failures": failures,
        "thread_count": len(memory.threads),
        "rendered_context": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset/memory_v4_golden_cases.jsonl")
    parser.add_argument("--output", default="eval/results/memory_v4_eval_report.json")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [evaluate_case(case) for case in cases]
    passed = sum(1 for r in results if r["passed"])
    report = {
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "issue_thread_accuracy": passed / max(1, len(results)),
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
