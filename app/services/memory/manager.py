"""记忆系统门面。"""

from __future__ import annotations

import threading

from app.core.config import settings
from app.services.context.fact_extractor import extract_critical_facts
from app.services.memory.consolidator import MemoryConsolidator, dedupe_memory_notes
from app.services.memory.episodic_store import EpisodicMemoryStore, build_episodic_memory_store
from app.services.memory.session_memory import SessionMemoryStore, build_session_memory_store
from app.services.memory.types import MemoryBundle, SessionMemory, UserProfile
from app.services.memory.user_profile_store import UserProfileStore, build_user_profile_store


class MemoryManager:
    """统一读写短期会话记忆、长期用户画像和情景向量记忆。"""

    def __init__(
        self,
        *,
        session_store: SessionMemoryStore | None = None,
        profile_store: UserProfileStore | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        consolidator: MemoryConsolidator | None = None,
    ) -> None:
        self.session_store = session_store or build_session_memory_store()
        self.profile_store = profile_store or build_user_profile_store()
        self.episodic_store = episodic_store or build_episodic_memory_store()
        self.consolidator = consolidator or MemoryConsolidator()

    def read(
        self,
        *,
        session_id: str | None,
        user_id: str | None = None,
        query: str = "",
    ) -> MemoryBundle:
        if not settings.memory_enabled:
            return MemoryBundle()
        session_memory = self.session_store.get(session_id or "") if session_id else None # 读取session记忆
        user_key = self.resolve_user_key( # 根据user_id或session_memory或query，生成user_key
            user_id=user_id,
            session_memory=session_memory,
            question=query,
        ) # 读取用户画像
        profile = self.profile_store.get(user_key) if user_key and settings.memory_user_profile_enabled else None
        episodic = self.episodic_store.search(user_key=user_key, query=query) if user_key else [] # 读取情景向量
        return MemoryBundle(
            session_memory=session_memory,
            user_profile=profile,
            episodic_notes=episodic,
            user_key=user_key,
        )

    def write_turn(
        self,
        *,
        session_id: str | None,
        question: str,
        answer: str,
        user_id: str | None = None,
        visual_context: str = "",
        context_block: str = "",
    ) -> MemoryBundle:
        if not (settings.memory_enabled and session_id):
            return MemoryBundle()
        facts = extract_critical_facts(
            question=question,
            context_block=context_block,
            visual_context=visual_context,
            conversation_history=answer,
        )
        session = self.session_store.merge_turn_facts(
            session_id,
            facts,
            question=question,
            answer=answer,
        )
        user_key = self.resolve_user_key(user_id=user_id, session_memory=session, question=question)
        profile = None
        if user_key and settings.memory_user_profile_enabled:
            profile = self.profile_store.upsert_from_facts(user_key, session.facts)
        if self._should_consolidate(session):
            self._schedule_consolidation(session_id=session.session_id, user_id=user_id)
        return MemoryBundle(session_memory=session, user_profile=profile, user_key=user_key)

    def consolidate(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> MemoryBundle:
        if not settings.memory_enabled:
            return MemoryBundle()
        session = self.session_store.get(session_id)
        user_key = self.resolve_user_key(user_id=user_id, session_memory=session)
        profile = self.profile_store.get(user_key) if user_key else None
        result = self.consolidator.consolidate(session=session, user_profile=profile)
        if result.session_summary:
            session = SessionMemory(
                session_id=session.session_id,
                facts=session.facts,
                summary=result.session_summary,
                turn_count=session.turn_count,
                updated_at=session.updated_at,
            )
            self.session_store.save(session)
        if user_key and settings.memory_user_profile_enabled:
            profile = self.profile_store.upsert_from_facts(user_key, session.facts)
        if user_key and settings.memory_episodic_enabled:
            existing = self.episodic_store.search(user_key=user_key, query=session.render(), top_k=20)
            candidates = [
                note if note.user_key else note.__class__(
                    memory_id=note.memory_id,
                    user_key=user_key,
                    memory_text=note.memory_text,
                    memory_type=note.memory_type,
                    source_session=note.source_session,
                    score=note.score,
                    created_at=note.created_at,
                    expire_ts=note.expire_ts,
                )
                for note in result.memory_notes
            ]
            self.episodic_store.upsert(dedupe_memory_notes(existing, candidates))
        episodic = self.episodic_store.search(user_key=user_key, query=session.render()) if user_key else []
        return MemoryBundle(session_memory=session, user_profile=profile, episodic_notes=episodic, user_key=user_key)

    def resolve_user_key(
        self,
        *,
        user_id: str | None = None,
        session_memory: SessionMemory | None = None,
        question: str = "",
    ) -> str:
        if user_id and user_id.strip():
            return UserProfileStore.hash_user_key(f"user:{user_id.strip()}")
        if session_memory and session_memory.facts.phones:
            return UserProfileStore.hash_user_key(f"phone:{session_memory.facts.phones[0]}")
        facts = extract_critical_facts(question=question)
        if facts.phones:
            return UserProfileStore.hash_user_key(f"phone:{facts.phones[0]}")
        return ""

    def forget(self, user_key: str) -> None:
        if not user_key:
            return
        self.profile_store.forget(user_key)
        self.episodic_store.forget(user_key)

    def _should_consolidate(self, session: SessionMemory) -> bool:
        every = max(1, settings.memory_consolidation_every_turns)
        return session.turn_count > 0 and session.turn_count % every == 0

    def _schedule_consolidation(self, *, session_id: str, user_id: str | None) -> None:
        if settings.memory_consolidation_async:
            thread = threading.Thread(
                target=lambda: self.consolidate(session_id=session_id, user_id=user_id),
                daemon=True,
            )
            thread.start()
            return
        self.consolidate(session_id=session_id, user_id=user_id)


_manager: MemoryManager | None = None
_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = MemoryManager()
    return _manager


def reset_memory_manager_for_tests() -> None:
    global _manager
    with _lock:
        _manager = None
