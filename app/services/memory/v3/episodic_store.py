"""Episodic 级 v3 记忆存储。

第一版提供内存实现，保持接口稳定；后续可替换为 Milvus user_memory_v2。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Callable

from app.core.config import settings
from app.services.memory.v3.types import EpisodicEvent, WriteDecision


class InMemoryEpisodicMemoryV3Store:
    def __init__(self) -> None:
        self._events: dict[str, list[EpisodicEvent]] = {}

    def apply_decisions(
        self,
        user_key: str,
        decisions: list[WriteDecision],
        *,
        turn_id: str,
    ) -> list[EpisodicEvent]:
        events = self._events.setdefault(user_key, [])
        for decision in decisions:
            if decision.action != "upsert_episodic" or decision.candidate is None:
                continue
            candidate = decision.candidate
            event = EpisodicEvent(
                event_id=f"event_{uuid.uuid4().hex[:16]}",
                user_key=user_key,
                event_type=_event_type_for(candidate.kind, candidate.value),
                title=f"{candidate.kind}: {candidate.value}",
                summary=f"{candidate.evidence_text or candidate.value}",
                product_model=candidate.product_model,
                created_at=time.time(),
            )
            events.append(event)
        return events

    def search(self, user_key: str, query: str, top_k: int = 3) -> list[EpisodicEvent]:
        query_text = query or ""
        events = self._events.get(user_key, [])
        scored = []
        for event in events:
            score = 0
            if event.product_model and event.product_model in query_text:
                score += 10
            if event.case_id and event.case_id in query_text:
                score += 10
            if any(token and token in event.summary for token in query_text.split()):
                score += 1
            scored.append((score, event))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [event for _, event in scored[:top_k]]


def _event_type_for(kind: str, value: str) -> str:
    if kind == "case_status" and value == "submitted":
        return "case_submitted"
    if kind == "case_status" and value == "resolved":
        return "issue_resolved"
    return "user_reported_history"


class MilvusEpisodicMemoryV3Store:
    """Milvus user_memory_v2 runtime adapter。

    默认 manager 仍使用内存 store，避免本地无 Milvus 时影响主链路；生产环境可以把
    这个类注入 MemoryManagerV3。
    """

    def __init__(self, client: object, embed_fn: Callable[[str], list[float]]) -> None:
        self.client = client
        self.embed_fn = embed_fn
        self.collection_name = settings.memory_v3_episodic_collection

    def apply_decisions(
        self,
        user_key: str,
        decisions: list[WriteDecision],
        *,
        turn_id: str,
    ) -> list[EpisodicEvent]:
        events: list[EpisodicEvent] = []
        rows: list[dict] = []
        user_hash = _hash_user_key(user_key)
        for decision in decisions:
            if decision.action != "upsert_episodic" or decision.candidate is None:
                continue
            candidate = decision.candidate
            now = time.time()
            event = EpisodicEvent(
                event_id=f"event_{uuid.uuid4().hex[:16]}",
                user_key=user_hash,
                event_type=_event_type_for(candidate.kind, candidate.value),
                title=f"{candidate.kind}: {candidate.value}",
                summary=candidate.evidence_text or candidate.value,
                product_model=candidate.product_model,
                issue_thread_id=candidate.issue_thread_id,
                created_at=now,
            )
            events.append(event)
            rows.append(
                {
                    "event_id": event.event_id,
                    "user_key": user_hash,
                    "event_type": event.event_type,
                    "title": event.title,
                    "summary": event.summary,
                    "product_model": event.product_model or "",
                    "case_id": event.case_id or "",
                    "issue_thread_id": event.issue_thread_id or "",
                    "dense_vector": self.embed_fn(event.summary),
                    "created_at": now,
                    "expire_ts": float(event.expires_at or 0.0),
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)
        return events

    def search(self, user_key: str, query: str, top_k: int = 3) -> list[EpisodicEvent]:
        vector = self.embed_fn(query or "")
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="dense_vector",
            limit=top_k,
            filter=f'user_key == "{_hash_user_key(user_key)}"',
            output_fields=[
                "event_id",
                "user_key",
                "event_type",
                "title",
                "summary",
                "product_model",
                "case_id",
                "issue_thread_id",
                "created_at",
                "expire_ts",
            ],
        )
        events: list[EpisodicEvent] = []
        for hit in (result[0] if result else []):
            entity = hit.get("entity") if isinstance(hit, dict) else getattr(hit, "entity", {})
            if not isinstance(entity, dict):
                continue
            events.append(
                EpisodicEvent(
                    event_id=str(entity.get("event_id") or ""),
                    user_key=str(entity.get("user_key") or ""),
                    event_type=str(entity.get("event_type") or ""),
                    title=str(entity.get("title") or ""),
                    summary=str(entity.get("summary") or ""),
                    product_model=str(entity.get("product_model") or "") or None,
                    case_id=str(entity.get("case_id") or "") or None,
                    issue_thread_id=str(entity.get("issue_thread_id") or "") or None,
                    created_at=float(entity.get("created_at") or 0.0),
                    expires_at=float(entity.get("expire_ts") or 0.0) or None,
                )
            )
        return events


def _hash_user_key(user_key: str) -> str:
    return "sha256:" + hashlib.sha256(user_key.encode("utf-8")).hexdigest()
