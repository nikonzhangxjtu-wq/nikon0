"""Persistent memory stores for nikon0.

Redis is used as the hot session snapshot. MySQL stores durable snapshots and
append-only StateUpdate events for replay/debug. Tests use SQLite through the
same SQLAlchemy path; production should pass a MySQL SQLAlchemy DSN.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, insert, select, text, update

from nikon0.app.schemas.capability import StateUpdate
from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.memory.session import InMemorySessionIssueStore


class SqlMemoryPersistence:
    """SQL persistence for session snapshots and StateUpdate events."""

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        self.dsn = dsn
        self.engine = create_engine(dsn, echo=echo, future=True)
        self.metadata = MetaData()
        self.sessions = Table(
            "nikon0_memory_sessions",
            self.metadata,
            Column("session_id", String(191), primary_key=True),
            Column("snapshot_json", JSON().with_variant(Text(), "sqlite"), nullable=False),
            Column("turn_count", Integer, nullable=False, default=0),
            Column("memory_version", Integer, nullable=False, default=0),
            Column("updated_at", Float, nullable=False),
        )
        self.events = Table(
            "nikon0_state_update_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(191), nullable=False, index=True),
            Column("turn_id", String(191), nullable=False, default=""),
            Column("update_key", String(191), nullable=False),
            Column("update_json", JSON().with_variant(Text(), "sqlite"), nullable=False),
            Column("reason", Text, nullable=False, default=""),
            Column("evidence_ids_json", JSON().with_variant(Text(), "sqlite"), nullable=False),
            Column("created_at", Float, nullable=False),
        )
        self.write_decisions = Table(
            "nikon0_memory_write_decisions",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(191), nullable=False, index=True),
            Column("thread_id", String(191), nullable=False, default="", index=True),
            Column("turn_id", String(191), nullable=False, default="", index=True),
            Column("candidate_id", String(191), nullable=False, default=""),
            Column("outcome", String(64), nullable=False),
            Column("decision_json", JSON().with_variant(Text(), "sqlite"), nullable=False),
            Column("created_at", Float, nullable=False),
        )
        self.thread_events = Table(
            "nikon0_memory_thread_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(191), nullable=False, index=True),
            Column("thread_id", String(191), nullable=False, index=True),
            Column("turn_id", String(191), nullable=False, default="", index=True),
            Column("action", String(64), nullable=False),
            Column("event_json", JSON().with_variant(Text(), "sqlite"), nullable=False),
            Column("created_at", Float, nullable=False),
        )
        self.metadata.create_all(self.engine)
        self._ensure_compatible_columns()

    def _ensure_compatible_columns(self) -> None:
        """Add the v1 concurrency column for installations created before it."""
        from sqlalchemy import inspect

        columns = {item["name"] for item in inspect(self.engine).get_columns("nikon0_memory_sessions")}
        if "memory_version" not in columns:
            with self.engine.begin() as conn:
                conn.execute(text("ALTER TABLE nikon0_memory_sessions ADD COLUMN memory_version INTEGER NOT NULL DEFAULT 0"))

    def save_snapshot(self, memory: SessionIssueMemory, *, expected_version: int | None = None) -> int:
        payload = memory.model_dump(mode="json")
        now = time.time()
        with self.engine.begin() as conn:
            current = conn.execute(
                select(self.sessions.c.memory_version).where(self.sessions.c.session_id == memory.session_id)
            ).scalar_one_or_none()
            if current is None:
                version = 1
                conn.execute(insert(self.sessions).values(
                    session_id=memory.session_id,
                    snapshot_json=self._json_value(payload),
                    turn_count=memory.turn_count,
                    memory_version=version,
                    updated_at=now,
                ))
            else:
                if expected_version is not None and int(current) != int(expected_version):
                    raise MemoryVersionConflict(memory.session_id, expected_version, int(current))
                version = int(current) + 1
                result = conn.execute(
                    update(self.sessions)
                    .where(self.sessions.c.session_id == memory.session_id)
                    .where(self.sessions.c.memory_version == int(current))
                    .values(
                        snapshot_json=self._json_value(payload),
                        turn_count=memory.turn_count,
                        memory_version=version,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise MemoryVersionConflict(memory.session_id, int(current), -1)
        memory.memory_version = version
        return version

    def load_snapshot(self, session_id: str) -> SessionIssueMemory | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.sessions.c.snapshot_json, self.sessions.c.memory_version).where(self.sessions.c.session_id == session_id)
            ).first()
        if row is None:
            return None
        payload = self._decode_json(row.snapshot_json)
        if not isinstance(payload, dict):
            return None
        # The dedicated SQL column is the concurrency authority. Snapshot JSON
        # is serialized before the successful version increment completes.
        payload["memory_version"] = int(row.memory_version or 0)
        return SessionIssueMemory(**payload)

    def append_write_decisions(self, session_id: str, decisions: list[dict[str, Any]], *, turn_id: str = "") -> None:
        if not decisions:
            return
        now = time.time()
        rows = [
            {
                "session_id": session_id,
                "thread_id": str(item.get("target_thread_id") or ""),
                "turn_id": turn_id,
                "candidate_id": str(item.get("candidate_id") or ""),
                "outcome": str(item.get("outcome") or "unknown"),
                "decision_json": self._json_value(item),
                "created_at": now,
            }
            for item in decisions
        ]
        with self.engine.begin() as conn:
            conn.execute(insert(self.write_decisions), rows)

    def append_thread_event(self, session_id: str, event: dict[str, Any], *, turn_id: str = "") -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(self.thread_events).values(
                session_id=session_id,
                thread_id=str(event.get("thread_id") or ""),
                turn_id=turn_id,
                action=str(event.get("action") or "unknown"),
                event_json=self._json_value(event),
                created_at=time.time(),
            ))

    def list_write_decisions(self, session_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.write_decisions.c.decision_json)
                .where(self.write_decisions.c.session_id == session_id)
                .order_by(self.write_decisions.c.id.asc())
            ).scalars().all()
        return [self._decode_json(row) for row in rows if isinstance(self._decode_json(row), dict)]

    def health_check(self) -> dict[str, Any]:
        with self.engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "dialect": self.engine.dialect.name, "schema_version": 1}

    def append_state_update_events(self, session_id: str, updates: list[StateUpdate], *, turn_id: str = "") -> None:
        if not updates:
            return
        now = time.time()
        rows = [
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "update_key": update.key,
                "update_json": self._json_value(update.model_dump(mode="json")),
                "reason": update.reason,
                "evidence_ids_json": self._json_value(list(update.evidence_ids)),
                "created_at": now,
            }
            for update in updates
        ]
        with self.engine.begin() as conn:
            conn.execute(insert(self.events), rows)

    def list_state_update_events(self, session_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    self.events.c.id,
                    self.events.c.session_id,
                    self.events.c.turn_id,
                    self.events.c.update_key,
                    self.events.c.update_json,
                    self.events.c.reason,
                    self.events.c.evidence_ids_json,
                    self.events.c.created_at,
                )
                .where(self.events.c.session_id == session_id)
                .order_by(self.events.c.id.asc())
            ).mappings().all()
        events: list[dict[str, Any]] = []
        for row in rows:
            update_json = self._decode_json(row["update_json"])
            evidence_ids = self._decode_json(row["evidence_ids_json"])
            events.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "turn_id": row["turn_id"],
                    "update_key": row["update_key"],
                    "update": update_json,
                    "reason": row["reason"],
                    "evidence_ids": evidence_ids if isinstance(evidence_ids, list) else [],
                    "created_at": row["created_at"],
                }
            )
        return events

    def _delete_session(self, conn, session_id: str) -> None:
        conn.execute(delete(self.sessions).where(self.sessions.c.session_id == session_id))

    def _json_value(self, value: Any) -> Any:
        if self.engine.dialect.name == "sqlite":
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class RedisMysqlSessionIssueStore(InMemorySessionIssueStore):
    """Session memory store backed by Redis snapshot + SQL durability."""

    _process_locks: dict[str, threading.Lock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(
        self,
        *,
        redis_client: Any,
        sql_persistence: SqlMemoryPersistence,
        redis_key_prefix: str = "nikon0:memory",
        ttl_seconds: int = 86400,
    ) -> None:
        super().__init__()
        self.redis_client = redis_client
        self.sql_persistence = sql_persistence
        self.redis_key_prefix = redis_key_prefix.strip(":") or "nikon0:memory"
        self.ttl_seconds = max(1, int(ttl_seconds))

    def load(self, session_id: str) -> SessionIssueMemory:
        key = self._redis_key(session_id)
        raw = self.redis_client.get(key)
        if raw:
            memory = self._memory_from_json(raw)
            self._state[session_id] = memory
            return memory.model_copy(deep=True)
        snapshot = self.sql_persistence.load_snapshot(session_id)
        if snapshot is not None:
            self._state[session_id] = snapshot
            self._save_redis(snapshot)
            return snapshot.model_copy(deep=True)
        return super().load(session_id)

    def apply_updates(
        self,
        session_id: str,
        updates: list[StateUpdate],
        *,
        turn_id: str = "",
        target_thread_id: str | None = None,
        create_thread: bool = False,
    ) -> SessionIssueMemory:
        with self._session_write_lock(session_id):
            # SQL is the version authority inside the write critical section.
            self._state[session_id] = self._load_from_sql(session_id)
            expected_version = self._state[session_id].memory_version
            memory = super().apply_updates(
                session_id,
                updates,
                turn_id=turn_id,
                target_thread_id=target_thread_id,
                create_thread=create_thread,
            )
            self.sql_persistence.save_snapshot(memory, expected_version=expected_version)
            self._save_redis(memory)
            self.sql_persistence.append_state_update_events(session_id, updates, turn_id=turn_id)
            return memory.model_copy(deep=True)

    def profile(self) -> dict[str, Any]:
        redis_ok = False
        try:
            redis_ok = bool(self.redis_client.ping())
        except Exception:  # noqa: BLE001
            pass
        sql = self.sql_persistence.health_check()
        return {
            "store_type": type(self).__name__,
            "redis_ok": redis_ok,
            "mysql_ok": bool(sql.get("ok")),
            "sql_dialect": sql.get("dialect"),
            "schema_version": sql.get("schema_version"),
            "degraded": False,
        }

    def _save_redis(self, memory: SessionIssueMemory) -> None:
        payload = memory.model_dump_json()
        self.redis_client.set(self._redis_key(memory.session_id), payload, ex=self.ttl_seconds)

    def _load_from_sql(self, session_id: str) -> SessionIssueMemory:
        snapshot = self.sql_persistence.load_snapshot(session_id)
        if snapshot is None:
            return SessionIssueMemory(session_id=session_id)
        self._save_redis(snapshot)
        return snapshot

    @contextmanager
    def _session_write_lock(self, session_id: str):
        """Use Redis for cross-process exclusion and a local lock as fallback."""
        local_lock = self._local_lock(session_id)
        if not local_lock.acquire(timeout=5):
            raise TimeoutError(f"timed out waiting for local memory lock: {session_id}")
        redis_lock_key = f"{self.redis_key_prefix}:lock:{session_id}"
        token = uuid4().hex
        redis_locked = False
        try:
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    acquired = self.redis_client.set(redis_lock_key, token, nx=True, ex=10)
                    if acquired:
                        redis_locked = True
                        break
                    time.sleep(0.01)
                if not redis_locked:
                    raise TimeoutError(f"timed out waiting for Redis memory lock: {session_id}")
            except TypeError:
                # Test doubles and minimal Redis adapters may not expose NX locks.
                redis_locked = False
            yield
        finally:
            if redis_locked:
                try:
                    self.redis_client.eval(
                        "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0",
                        1,
                        redis_lock_key,
                        token,
                    )
                except Exception:  # noqa: BLE001
                    pass
            local_lock.release()

    @classmethod
    def _local_lock(cls, session_id: str) -> threading.Lock:
        with cls._process_locks_guard:
            lock = cls._process_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                cls._process_locks[session_id] = lock
            return lock

    def _redis_key(self, session_id: str) -> str:
        return f"{self.redis_key_prefix}:session:{session_id}"

    @staticmethod
    def _memory_from_json(raw: Any) -> SessionIssueMemory:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))
        return SessionIssueMemory(**payload)


class MemoryVersionConflict(RuntimeError):
    def __init__(self, session_id: str, expected: int, actual: int) -> None:
        super().__init__(f"memory version conflict for {session_id}: expected={expected}, actual={actual}")


def build_memory_store_from_env(*, settings=None) -> InMemorySessionIssueStore:
    """Build nikon0 memory store from environment.

    Environment:
    - NIKON0_MEMORY_STORE=memory | redis_mysql
    - NIKON0_MEMORY_REDIS_URL=redis://127.0.0.1:6379/2
    - NIKON0_MEMORY_MYSQL_DSN=mysql+pymysql://user:pass@127.0.0.1:3306/nikon0?charset=utf8mb4
    - NIKON0_MEMORY_REDIS_PREFIX=nikon0:memory
    - NIKON0_MEMORY_TTL_SECONDS=86400
    """

    if settings is None:
        try:
            from app.core.config import settings as loaded_settings
        except Exception:  # noqa: BLE001
            loaded_settings = None
        settings = loaded_settings
    mode = str(getattr(settings, "nikon0_memory_store", "memory") or "memory").strip().lower()
    if mode not in {"redis_mysql", "mysql_redis"}:
        return InMemorySessionIssueStore()
    redis_url = str(getattr(settings, "nikon0_memory_redis_url", "") or getattr(settings, "redis_url", "") or "")
    mysql_dsn = str(getattr(settings, "nikon0_memory_mysql_dsn", "") or "")
    if not redis_url or not mysql_dsn:
        print("[WARN] NIKON0_MEMORY_STORE=redis_mysql 但 Redis/MySQL 配置不完整，回退内存。")
        return InMemorySessionIssueStore()
    try:
        import redis as redis_lib  # type: ignore[import-untyped]

        redis_client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        sql = SqlMemoryPersistence(mysql_dsn)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] nikon0 持久化记忆初始化失败，回退内存: {exc}")
        return InMemorySessionIssueStore()
    return RedisMysqlSessionIssueStore(
        redis_client=redis_client,
        sql_persistence=sql,
        redis_key_prefix=str(getattr(settings, "nikon0_memory_redis_prefix", "nikon0:memory") or "nikon0:memory"),
        ttl_seconds=int(getattr(settings, "nikon0_memory_ttl_seconds", 86400) or 86400),
    )
