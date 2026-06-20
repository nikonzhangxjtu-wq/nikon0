"""Inspect Redis/MySQL memory persistence for eval sessions."""

from __future__ import annotations

import json
import sys

import redis
from sqlalchemy import create_engine, text

from app.core.config import settings
from nikon0.memory.persistence import SqlMemoryPersistence, build_memory_store_from_env


def main() -> int:
    prefix = settings.nikon0_memory_redis_prefix
    redis_url = settings.nikon0_memory_redis_url
    mysql_dsn = settings.nikon0_memory_mysql_dsn

    store = build_memory_store_from_env()
    print("=== RUNTIME STORE ===")
    print(f"configured_store={settings.nikon0_memory_store}")
    print(f"runtime_store_type={type(store).__name__}")
    if hasattr(store, "profile"):
        print(f"profile={json.dumps(store.profile(), ensure_ascii=False)}")

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    pattern = f"{prefix}:session:eval-qa-*"
    keys = sorted(r.scan_iter(match=pattern))

    print("\n=== REDIS ===")
    print(f"pattern={pattern}")
    print(f"key_count={len(keys)}")
    for key in keys:
        ttl = r.ttl(key)
        raw = r.get(key)
        try:
            data = json.loads(raw)
            flat_ps = (data.get("flat_state") or {}).get("product_support") or {}
            last_query = flat_ps.get("last_query", "")
            preview = (last_query[:50] + "...") if isinstance(last_query, str) and len(last_query) > 50 else last_query
            threads = {tid: item.get("status") for tid, item in (data.get("threads") or {}).items()}
            print(
                f"- {key} ttl={ttl}s turn_count={data.get('turn_count')} "
                f"memory_version={data.get('memory_version')} active_skill={data.get('active_skill')} "
                f"product={(data.get('active_product') or {}).get('product_id')} threads={threads}"
            )
            print(f"  last_query={preview}")
        except Exception as exc:  # noqa: BLE001
            print(f"- {key} ttl={ttl}s parse_error={exc}")

    sql = SqlMemoryPersistence(mysql_dsn)
    redis_132 = r.get(f"{prefix}:session:eval-qa-132")
    mysql_132 = sql.load_snapshot("eval-qa-132")
    print("\n=== REDIS vs MYSQL (eval-qa-132) ===")
    if not redis_132:
        print("redis: MISSING")
    else:
        ro = json.loads(redis_132)
        print(
            f"redis: turn_count={ro.get('turn_count')} memory_version={ro.get('memory_version')} "
            f"active_thread={ro.get('active_thread_id')}"
        )
    if mysql_132 is None:
        print("mysql: MISSING")
    else:
        md = mysql_132.model_dump()
        print(
            f"mysql: turn_count={md.get('turn_count')} memory_version={md.get('memory_version')} "
            f"active_thread={md.get('active_thread_id')}"
        )
    if redis_132 and mysql_132 is not None:
        identical = json.dumps(json.loads(redis_132), sort_keys=True) == json.dumps(
            mysql_132.model_dump(mode="json"), sort_keys=True
        )
        print(f"redis_mysql_identical={identical}")

    engine = create_engine(mysql_dsn, future=True)
    with engine.begin() as conn:
        sess_count = conn.execute(text("SELECT COUNT(*) FROM nikon0_memory_sessions")).scalar()
        eval_sess = conn.execute(
            text("SELECT COUNT(*) FROM nikon0_memory_sessions WHERE session_id LIKE 'eval-qa-%'")
        ).scalar()
        evt_count = conn.execute(text("SELECT COUNT(*) FROM nikon0_state_update_events")).scalar()
        eval_evt = conn.execute(
            text("SELECT COUNT(*) FROM nikon0_state_update_events WHERE session_id LIKE 'eval-qa-%'")
        ).scalar()

    print("\n=== MYSQL ===")
    print(f"sessions_total={sess_count} eval_sessions={eval_sess}")
    print(f"events_total={evt_count} eval_events={eval_evt}")

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT session_id, turn_count, memory_version, updated_at
                FROM nikon0_memory_sessions
                WHERE session_id LIKE 'eval-qa-%'
                ORDER BY session_id
                """
            )
        ).mappings().all()
    print("eval session rows:")
    for row in rows:
        print(
            f"  {row['session_id']}: turn_count={row['turn_count']} "
            f"memory_version={row['memory_version']} updated_at={row['updated_at']}"
        )

    events_132 = sql.list_state_update_events("eval-qa-132")
    print(f"\neval-qa-132 state_update_events={len(events_132)}")
    for event in events_132:
        update = event.get("update") or {}
        value = update.get("value") if isinstance(update, dict) else {}
        last_query = value.get("last_query") if isinstance(value, dict) else None
        preview = (str(last_query)[:55] + "...") if isinstance(last_query, str) and len(last_query) > 55 else last_query
        turn_id = str(event.get("turn_id") or "")
        print(
            f"  id={event['id']} turn_id={turn_id[:12]} key={event['update_key']} last_query={preview}"
        )

    print("\n=== REQUIREMENT CHECK (eval-qa-126..135) ===")
    expected_ids = [f"eval-qa-{index}" for index in range(126, 136)]
    issues: list[str] = []
    for session_id in expected_ids:
        snapshot = sql.load_snapshot(session_id)
        has_redis = bool(r.exists(f"{prefix}:session:{session_id}"))
        if snapshot is None:
            issues.append(f"{session_id}: missing mysql snapshot")
            continue
        data = snapshot.model_dump()
        if int(data.get("turn_count") or 0) < 2:
            issues.append(f"{session_id}: turn_count={data.get('turn_count')} expected>=2")
        if not has_redis:
            issues.append(f"{session_id}: missing redis key")
        if not data.get("threads"):
            issues.append(f"{session_id}: no threads")
        if not sql.list_state_update_events(session_id):
            issues.append(f"{session_id}: no state_update_events")

    if issues:
        print("ISSUES:")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("PASS: all 10 eval sessions in MySQL+Redis with turn_count>=2 and event logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
