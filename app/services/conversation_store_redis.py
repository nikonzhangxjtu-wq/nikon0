"""多轮对话 Redis 存储：STRING + JSON，与会话 TTL 对齐。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.core.config import settings
from app.services.conversation_store import (
    ConversationSession,
    ConversationTurn,
    _strip_pic,
    _truncate,
)


class RedisConversationStore:
    """与 ``ConversationStore`` 同接口，数据落在 Redis。"""

    def __init__(self, client: object) -> None:
        self._r = client
        self._ttl = settings.conversation_ttl_seconds
        self._max_turns = settings.conversation_max_turns
        self._prefix = (settings.redis_conversation_key_prefix or "kf").strip().rstrip(":")

    def _key(self, session_id: str) -> str:
        sid = session_id.strip()
        digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
        return f"{self._prefix}:conversation:{digest}"

    def _serialize_turn(self, t: ConversationTurn) -> dict[str, Any]:
        return {
            "question": t.question,
            "answer": t.answer,
            "user_images": list(t.user_images),
            "answer_images": list(t.answer_images),
            "timestamp": t.timestamp,
        }

    def _deserialize_turn(self, obj: object) -> ConversationTurn | None:
        if not isinstance(obj, dict):
            return None
        return ConversationTurn(
            question=str(obj.get("question", "")),
            answer=str(obj.get("answer", "")),
            user_images=[str(x) for x in obj.get("user_images", []) if str(x)],
            answer_images=[str(x) for x in obj.get("answer_images", []) if str(x)],
            timestamp=float(obj.get("timestamp", time.time())),
        )

    def _load_raw(self, session_id: str) -> ConversationSession | None:
        raw = self._r.get(self._key(session_id))
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        sid = str(obj.get("session_id", session_id.strip()))
        created_at = float(obj.get("created_at", time.time()))
        last_active = float(obj.get("last_active", time.time()))
        turns_raw = obj.get("turns")
        turns: list[ConversationTurn] = []
        if isinstance(turns_raw, list):
            for item in turns_raw:
                t = self._deserialize_turn(item)
                if t is not None:
                    turns.append(t)
        return ConversationSession(
            session_id=sid,
            turns=turns,
            created_at=created_at,
            last_active=last_active,
        )

    def _save(self, session: ConversationSession) -> None:
        payload = json.dumps(
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "turns": [self._serialize_turn(t) for t in session.turns],
            },
            ensure_ascii=False,
        )
        self._r.set(self._key(session.session_id), payload, ex=max(60, int(self._ttl)))

    def get_or_create(self, session_id: str) -> ConversationSession:
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id 不能为空")
        now = time.time()
        session = self._load_raw(sid)
        if session is None:
            session = ConversationSession(session_id=sid)
        else:
            if now - session.last_active > self._ttl:
                session = ConversationSession(session_id=sid)
            else:
                session.last_active = now
        self._save(session)
        return session

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        user_images: list[str] | None = None,
        answer_images: list[str] | None = None,
    ) -> None:
        session = self.get_or_create(session_id)
        session.turns.append(
            ConversationTurn(
                question=question,
                answer=answer,
                user_images=user_images or [],
                answer_images=answer_images or [],
            )
        )
        if len(session.turns) > self._max_turns:
            session.turns = session.turns[-self._max_turns:]
        session.last_active = time.time()
        self._save(session)

    def format_history(self, session_id: str) -> str:
        session = self.get_or_create(session_id)
        turns = session.turns[-self._max_turns:]
        if not turns:
            return ""
        lines = ["对话历史 / Conversation History："]
        for t in turns:
            q_clean = _strip_pic(t.question)
            a_clean = _strip_pic(t.answer)
            lines.append(f"用户: {q_clean}")
            lines.append(f"助手: {a_clean}")
            lines.append("---")
        return "\n".join(lines).rstrip("-\n")

    def format_enrichment(self, session_id: str) -> str:
        session = self.get_or_create(session_id)
        turns = session.turns[-2:]
        if not turns:
            return ""
        parts = ["[对话上文]"]
        for t in turns:
            parts.append(f"用户问「{t.question}」→ 助手答「{_truncate(t.answer, 80)}」")
        parts.append("[对话上文结束]")
        return "\n".join(parts)

    def clear(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))
