"""Real Redis/MySQL load test for Memory Governance V1."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

from nikon0.app.schemas.capability import StateUpdate
from nikon0.memory.persistence import RedisMysqlSessionIssueStore, build_memory_store_from_env


async def run_load_test(*, output_dir: str | Path, independent_sessions: int = 40, shared_writers: int = 10) -> dict:
    store = build_memory_store_from_env()
    if not isinstance(store, RedisMysqlSessionIssueStore):
        raise RuntimeError("memory load test requires configured RedisMysqlSessionIssueStore")
    started = time.perf_counter()
    latencies: list[float] = []
    errors: list[str] = []
    run_id = uuid4().hex[:12]

    async def write(session_id: str, update: StateUpdate, turn_id: str) -> None:
        begin = time.perf_counter()
        try:
            await asyncio.to_thread(store.apply_updates, session_id, [update], turn_id=turn_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{session_id}:{type(exc).__name__}:{exc}")
        finally:
            latencies.append((time.perf_counter() - begin) * 1000)

    independent = [
        write(
            f"memory-load-{run_id}-{index}",
            StateUpdate(key="product_support", value={"selected_product_id": "airfryer", "last_query": f"query-{index}"}),
            f"independent-{index}",
        )
        for index in range(independent_sessions)
    ]
    shared_session = f"memory-load-shared-{run_id}"
    shared = [
        write(
            shared_session,
            StateUpdate(key="case_intake", value={"ticket_payload": {"order_id": f"ORD-{1000 + index}"}}),
            f"shared-{index}",
        )
        for index in range(shared_writers)
    ]
    await asyncio.gather(*independent, *shared)
    shared_events = store.sql_persistence.list_state_update_events(shared_session)
    result = {
        "run_id": run_id,
        "store_profile": store.profile(),
        "independent_sessions": independent_sessions,
        "shared_writers": shared_writers,
        "operations": len(latencies),
        "errors": errors,
        "error_rate": len(errors) / len(latencies) if latencies else 0.0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "throughput_ops_sec": len(latencies) / max(time.perf_counter() - started, 0.001),
        "shared_event_count": len(shared_events),
        "shared_consistent": len(shared_events) == shared_writers,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"memory-load-{run_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return round(ordered[index], 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="nikon0/eval/reports/memory-load")
    parser.add_argument("--independent-sessions", type=int, default=40)
    parser.add_argument("--shared-writers", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_load_test(
        output_dir=args.output_dir,
        independent_sessions=args.independent_sessions,
        shared_writers=args.shared_writers,
    )), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
