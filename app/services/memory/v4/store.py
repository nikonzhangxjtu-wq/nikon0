"""v4 session issue memory 存储。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

from app.core.config import settings
from app.services.memory.v4.types import EvidenceRef, IssueFact, IssueThread, SessionIssueMemory


class InMemorySessionIssueMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionIssueMemory] = {}

    def load(self, session_id: str) -> SessionIssueMemory:
        sid = (session_id or "").strip()
        if sid not in self._data:
            self._data[sid] = SessionIssueMemory(session_id=sid, updated_at=time.time())
        return self._data[sid]

    def save(self, memory: SessionIssueMemory) -> None:
        memory.updated_at = time.time()
        self._data[memory.session_id] = memory

    def clear(self, session_id: str) -> None:
        self._data.pop((session_id or "").strip(), None)


class RedisSessionIssueMemoryStore:
    def __init__(
        self,
        client: object,
        *,
        key_prefix: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._r = client
        self._prefix = (key_prefix or settings.redis_conversation_key_prefix or "kf").strip().rstrip(":")
        self._ttl = int(ttl_seconds or settings.conversation_ttl_seconds)

    def _key(self, session_id: str) -> str:
        digest = hashlib.sha256((session_id or "").strip().encode("utf-8")).hexdigest()
        return f"{self._prefix}:memory:v4:session:{digest}"

    def load(self, session_id: str) -> SessionIssueMemory:
        sid = (session_id or "").strip()
        raw = self._r.get(self._key(sid))
        if not raw:
            return SessionIssueMemory(session_id=sid, updated_at=time.time())
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            obj = json.loads(raw)
        except Exception:
            return SessionIssueMemory(session_id=sid, updated_at=time.time())
        return _memory_from_dict(obj, fallback_session_id=sid)

    def save(self, memory: SessionIssueMemory) -> None:
        memory.updated_at = time.time()
        payload = json.dumps(_memory_to_dict(memory), ensure_ascii=False)
        self._r.set(self._key(memory.session_id), payload, ex=max(60, self._ttl))

    def clear(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


def _memory_to_dict(memory: SessionIssueMemory) -> dict[str, Any]:
    return asdict(memory)


def _memory_from_dict(obj: dict[str, Any], *, fallback_session_id: str) -> SessionIssueMemory:
    threads = {}
    for tid, raw_thread in (obj.get("threads") or {}).items():
        facts = {
            fid: IssueFact(**raw_fact)
            for fid, raw_fact in (raw_thread.get("facts") or {}).items()
            if isinstance(raw_fact, dict)
        }
        evidence_refs = {
            eid: EvidenceRef(**raw_ev)
            for eid, raw_ev in (raw_thread.get("evidence_refs") or {}).items()
            if isinstance(raw_ev, dict)
        }
        threads[tid] = IssueThread(
            thread_id=str(raw_thread.get("thread_id") or tid),
            status=str(raw_thread.get("status") or "open"),
            issue_type=str(raw_thread.get("issue_type") or "unknown"),
            product_model=raw_thread.get("product_model"),
            facts=facts,
            evidence_refs=evidence_refs,
            last_turn_ids=[str(x) for x in raw_thread.get("last_turn_ids", [])],
            created_at=float(raw_thread.get("created_at") or 0.0),
            updated_at=float(raw_thread.get("updated_at") or 0.0),
        )
    return SessionIssueMemory(
        session_id=str(obj.get("session_id") or fallback_session_id),
        active_thread_id=obj.get("active_thread_id"),
        threads=threads,
        entity_index={
            str(k): [str(x) for x in v]
            for k, v in (obj.get("entity_index") or {}).items()
            if isinstance(v, list)
        },
        turn_count=int(obj.get("turn_count") or 0),
        updated_at=float(obj.get("updated_at") or 0.0),
    )
