"""Manual QA Eval Runner - 运行 150 条数据集，收集原始结果和 traces."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.eval.runtime_profiles import build_profiled_eval_runtime, EvalRuntimeProfile


def _trace_payload(response: Any) -> dict[str, Any]:
    debug = getattr(response, "debug", {}) or {}
    return {
        "trace": debug.get("trace") or {},
        "context_debug": debug.get("context_debug") or {},
        "loop": debug.get("loop") or {},
        "plan": debug.get("plan"),
        "skill_selection": debug.get("skill_selection"),
        "skill_manifests": debug.get("skill_manifests") or [],
        "memory_governance": debug.get("memory_governance") or {},
        "trace_persisted": debug.get("trace_persisted"),
        "transcript_entries": debug.get("transcript_entries"),
    }


async def run_one(
    runtime: AgentRuntime,
    item: dict,
    *,
    save_trace: bool = False,
) -> dict[str, Any]:
    """Run an item, including its explicit prior user turns when provided."""
    case_id = item["case_id"]
    message = item.get("message", "")
    session_id = f"eval-{case_id}"

    start = time.perf_counter()
    try:
        prior_turns = item.get("prior_turns", [])
        if not isinstance(prior_turns, list):
            prior_turns = []
        for prior_message in prior_turns:
            if not isinstance(prior_message, str) or not prior_message.strip():
                continue
            await runtime.run(AgentRequest(
                session_id=session_id,
                user_id="eval-user",
                message=prior_message,
            ))
        response = await runtime.run(AgentRequest(
            session_id=session_id,
            user_id="eval-user",
            message=message,
        ))
        elapsed = time.perf_counter() - start
        answer = getattr(response, "answer", "") or ""
        debug = getattr(response, "debug", {}) or {}
        selection = debug.get("skill_selection", {})
        loop_info = debug.get("loop", {})
        result = {
            "case_id": case_id,
            "category": item.get("category", ""),
            "message": message,
            "golden_answer": item.get("golden_answer", ""),
            "answer": answer,
            "status": "ok" if answer else "empty",
            "selected_skill": selection.get("selected_skill", "") or "",
            "selection_source": str(selection.get("source", "")) or "",
            "risk_level": getattr(response, "risk_level", "low"),
            "actions": [
                a.model_dump() if hasattr(a, "model_dump") else a
                for a in getattr(response, "actions", [])
            ],
            "elapsed_sec": round(elapsed, 3),
            "loop_turns": loop_info.get("turn_count", 0),
            "prior_turn_count": len(prior_turns),
            "conversation_mode": "multi_turn" if prior_turns else "single_turn",
            "trace_id": getattr(response, "trace_id", ""),
            "error": None,
        }
        if save_trace:
            result["trace_payload"] = _trace_payload(response)
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "case_id": case_id,
            "category": item.get("category", ""),
            "message": message,
            "golden_answer": item.get("golden_answer", ""),
            "answer": "",
            "status": "error",
            "selected_skill": "",
            "selection_source": "",
            "elapsed_sec": round(elapsed, 3),
            "actions": [],
            "loop_turns": 0,
            "trace_id": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _print_progress(
    current: int,
    total: int,
    *,
    profile: str,
    ok: int,
    error: int,
    case_label: str = "",
) -> None:
    width = 28
    ratio = current / total if total else 1.0
    filled = min(width, max(0, int(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" current={case_label}" if case_label else ""
    sys.stderr.write(
        f"\r[manual-qa][{profile}] [{bar}] {current}/{total} ok={ok} error={error}{suffix}"
    )
    sys.stderr.flush()


async def run_all(
    dataset_path: str | Path,
    output_dir: str | Path,
    manual_dir: str | Path,
    *,
    limit: int | None = None,
    case_id: str | None = None,
    profile: EvalRuntimeProfile = EvalRuntimeProfile.PRODUCTION_LIKE,
    use_real_llm: bool | None = None,
    local_rag: bool | None = None,
    mock_case_intake_tool: bool | None = None,
    show_progress: bool = False,
    save_traces: bool = False,
) -> dict[str, Any]:
    """运行所有 QA items."""
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if case_id:
        items = [item for item in items if item.get("case_id") == case_id]
        if not items:
            raise ValueError(f"case_id not found in dataset: {case_id}")
    elif limit:
        items = items[:limit]

    print(f"Building runtime (profile={profile.value})...", flush=True)
    deterministic_profile = profile in {EvalRuntimeProfile.DETERMINISTIC, EvalRuntimeProfile.LEGACY_EVAL}
    resolved_use_real_llm = (not deterministic_profile) if use_real_llm is None else use_real_llm
    resolved_local_rag = deterministic_profile if local_rag is None else local_rag
    profiled = build_profiled_eval_runtime(
        runtime_profile=profile,
        manual_dir=manual_dir,
        use_real_llm=resolved_use_real_llm,
        local_rag=resolved_local_rag,
        mock_case_intake_tool=mock_case_intake_tool,
    )
    runtime = profiled.runtime
    audit = profiled.audit
    print(
        "Runtime ready:"
        f" profile={audit.runtime_profile}"
        f" context={audit.context_profile}"
        f" use_real_llm={resolved_use_real_llm}"
        f" local_rag={resolved_local_rag}"
        f" production_aligned={audit.eval_runtime_matches_production}"
    )

    print(f"Running {len(items)} items...")
    results = []
    start_time = time.perf_counter()
    if show_progress:
        _print_progress(0, len(items), profile=profile.value, ok=0, error=0, case_label="starting")

    for i, item in enumerate(items):
        cid = item["case_id"]
        cat = item.get("category", "")
        if not show_progress:
            sys.stdout.write(f"\r[{i+1}/{len(items)}] {cid} ({cat})... ")
            sys.stdout.flush()

        result = await run_one(runtime, item, save_trace=save_traces)
        results.append(result)
        if save_traces and result.get("trace_payload") is not None:
            trace_dir = run_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{cid}.json"
            trace_doc = {key: value for key, value in result.items() if key != "trace_payload"}
            trace_doc["debug"] = result["trace_payload"]
            trace_path.write_text(
                json.dumps(trace_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Trace saved to {trace_path}", flush=True)
        if show_progress:
            _print_progress(
                i + 1,
                len(items),
                profile=profile.value,
                ok=sum(1 for row in results if row["status"] == "ok"),
                error=sum(1 for row in results if row["status"] == "error"),
                case_label=cid,
            )

    total_elapsed = time.perf_counter() - start_time
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    print(f"\nDone in {total_elapsed:.1f}s ({total_elapsed/len(items):.2f}s/item)")

    # 统计
    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "ok": 0, "error": 0, "empty": 0})
        by_category[cat]["total"] += 1
        if r["status"] == "error":
            by_category[cat]["error"] += 1
        elif r["status"] == "empty":
            by_category[cat]["empty"] += 1
        else:
            by_category[cat]["ok"] += 1

    ok_count = sum(1 for r in results if r["answer"])
    error_count = sum(1 for r in results if r["status"] == "error")

    report = {
        "dataset": str(dataset_path),
        "runtime_profile": profile.value,
        "runtime_profile_audit": profiled.audit.model_dump(),
        "use_real_llm": resolved_use_real_llm,
        "local_rag": resolved_local_rag,
        "mock_case_intake_tool": profiled.audit.case_intake_tool_mode == "mock",
        "total_items": len(results),
        "total_elapsed_sec": round(total_elapsed, 1),
        "avg_sec_per_item": round(total_elapsed / len(results), 2),
        "ok_count": ok_count,
        "error_count": error_count,
        "by_category": by_category,
        "multi_turn_cases": sum(1 for item in items if item.get("category") == "multi-turn"),
        "true_multi_turn_cases": sum(
            1
            for result in results
            if result.get("category") == "multi-turn" and result.get("prior_turn_count", 0) > 0
        ),
    }

    # 保存结果
    result_rows = [
        {key: value for key, value in row.items() if key != "trace_payload"}
        for row in results
    ]
    (run_dir / "raw_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in result_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport saved to {run_dir}")
    print(f"  OK: {ok_count}, Error: {error_count}, Empty: {len(results) - ok_count - error_count}")
    for cat, stats in sorted(by_category.items()):
        print(f"  {cat}: ok={stats['ok']}, error={stats['error']}, empty={stats['empty']}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl")
    parser.add_argument("--output", default="nikon0/eval/reports/manual-qa-eval-150")
    parser.add_argument("--manual-dir", default="/Users/nikonzhang/compeletion/手册")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", default=None, help="Run a single dataset case by case_id.")
    parser.add_argument("--save-traces", action="store_true", help="Write full debug trace JSON per case under <output>/traces/.")
    parser.add_argument("--profile", default="production_like",
                        choices=["deterministic", "production_like", "production_like_no_llm", "legacy_eval"])
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-real-llm", dest="use_real_llm", action="store_true")
    llm_group.add_argument("--no-real-llm", dest="use_real_llm", action="store_false")
    parser.set_defaults(use_real_llm=None)
    parser.add_argument("--local-rag", action="store_true", default=None, help="Use StructuredManualBackend rather than EnterpriseRagBackend.")
    parser.add_argument("--mock-case-intake-tool", action="store_true", default=None, help="Use eval-only case-intake tool.")
    parser.add_argument("--progress", action="store_true", help="Show an in-terminal progress bar with runtime profile.")
    args = parser.parse_args()

    asyncio.run(run_all(
        dataset_path=args.dataset,
        output_dir=args.output,
        manual_dir=args.manual_dir,
        limit=args.limit,
        case_id=args.case_id,
        profile=EvalRuntimeProfile(args.profile),
        use_real_llm=args.use_real_llm,
        local_rag=args.local_rag,
        mock_case_intake_tool=args.mock_case_intake_tool,
        show_progress=args.progress,
        save_traces=args.save_traces,
    ))
