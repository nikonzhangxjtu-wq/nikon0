"""长期情景向量记忆存储。"""

from __future__ import annotations

import hashlib
import time
from typing import Callable

from app.core.config import settings
from app.services.memory.types import MemoryNote
from app.services.memory.user_profile_store import UserProfileStore

EmbedFn = Callable[[str], list[float]]


class EpisodicMemoryStore:
    """Milvus-backed 情景记忆；依赖不可用时安全降级为空结果。"""

    def __init__(self, client: object | None = None, embed_fn: EmbedFn | None = None) -> None:
        self._client = client
        self._embed_fn = embed_fn
        self._collection = settings.memory_episodic_collection

    def search(self, *, user_key: str, query: str, top_k: int | None = None) -> list[MemoryNote]:
        if not (settings.memory_enabled and settings.memory_episodic_enabled):
            return []
        hashed = UserProfileStore.hash_user_key(user_key)
        if not (self._client and self._embed_fn and hashed and query.strip()):
            return []
        try:
            vector = self._embed_fn(query)
            if not vector:
                return []
            results = self._client.search(
                collection_name=self._collection,
                anns_field="dense_vector",
                data=[vector],
                limit=top_k or settings.memory_episodic_top_k,
                filter=f'user_key == "{hashed}"',
                output_fields=[
                    "memory_id",
                    "user_key",
                    "memory_text",
                    "memory_type",
                    "source_session",
                    "created_at",
                    "expire_ts",
                ],
            )
        except Exception:  # noqa: BLE001
            return []
        hits = results[0] if results and results[0] else []
        notes: list[MemoryNote] = []
        now = time.time()
        for hit in hits:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            score = float(hit.get("score", hit.get("distance", 0.0))) if isinstance(hit, dict) else 0.0
            expire_ts = float(entity.get("expire_ts") or 0.0)
            if expire_ts and expire_ts < now:
                continue
            if score < settings.memory_episodic_score_threshold:
                continue
            notes.append(
                MemoryNote(
                    memory_id=str(entity.get("memory_id") or ""),
                    user_key=str(entity.get("user_key") or hashed),
                    memory_text=str(entity.get("memory_text") or ""),
                    memory_type=str(entity.get("memory_type") or "episodic"),
                    source_session=str(entity.get("source_session") or ""),
                    score=score,
                    created_at=float(entity.get("created_at") or now),
                    expire_ts=expire_ts,
                )
            )
        return notes

    def upsert(self, notes: list[MemoryNote]) -> None:
        if not (self._client and self._embed_fn and notes):
            return
        rows: list[dict] = []
        for note in notes:
            text = note.memory_text.strip()
            if not text:
                continue
            try:
                vector = self._embed_fn(text)
            except Exception:  # noqa: BLE001
                continue
            if not vector:
                continue
            rows.append(
                {
                    "memory_id": note.memory_id or _memory_id(note.user_key, text),
                    "user_key": UserProfileStore.hash_user_key(note.user_key),
                    "memory_text": text,
                    "dense_vector": vector,
                    "memory_type": note.memory_type,
                    "source_session": note.source_session,
                    "created_at": note.created_at,
                    "expire_ts": note.expire_ts,
                }
            )
        if not rows:
            return
        try:
            self._client.upsert(collection_name=self._collection, data=rows)
        except Exception:  # noqa: BLE001
            return

    def forget(self, user_key: str) -> None:
        hashed = UserProfileStore.hash_user_key(user_key)
        if not (self._client and hashed):
            return
        try:
            self._client.delete(
                collection_name=self._collection,
                filter=f'user_key == "{hashed}"',
            )
        except Exception:  # noqa: BLE001
            return


def build_episodic_memory_store() -> EpisodicMemoryStore:
    if not (settings.memory_enabled and settings.memory_episodic_enabled):
        return EpisodicMemoryStore()
    try:
        from pymilvus import MilvusClient
    except Exception:  # noqa: BLE001
        return EpisodicMemoryStore()
    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_db_name,
        )
        if not client.has_collection(collection_name=settings.memory_episodic_collection):
            return EpisodicMemoryStore()
    except Exception:  # noqa: BLE001
        return EpisodicMemoryStore()
    return EpisodicMemoryStore(client, _embed_text)


def _embed_text(text: str) -> list[float]:
    from langchain_ollama import OllamaEmbeddings

    embedder = OllamaEmbeddings(model=settings.embed_model_zh, base_url=settings.ollama_base_url)
    return list(embedder.embed_query(text))


def _memory_id(user_key: str, text: str) -> str:
    digest = hashlib.sha256(f"{user_key}\n{text}".encode("utf-8")).hexdigest()
    return digest
