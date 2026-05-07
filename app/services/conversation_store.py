"""多轮对话状态存储：内存字典 + TTL 惰性过期。

生产环境可替换为 Redis，接口不变。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings


@dataclass
class ConversationTurn:
    """一轮对话。"""

    question: str
    answer: str
    user_images: list[str] = field(default_factory=list)
    answer_images: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """一个会话。"""

    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    """内存会话存储，惰性过期清理。"""

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.conversation_ttl_seconds
        self._max_turns = settings.conversation_max_turns
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_or_create(self, session_id: str) -> ConversationSession:
        """读取已有会话，不存在或已过期则创建新会话。"""
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id 不能为空")
        with self._lock:
            self._cleanup_expired()
            session = self._sessions.get(sid)
            if session is None:
                session = ConversationSession(session_id=sid)
                self._sessions[sid] = session
            else:
                session.last_active = time.time()
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
        """追加一轮对话，超出上限时移除最早轮次。"""
        session = self.get_or_create(session_id)
        turn = ConversationTurn(
            question=question,
            answer=answer,
            user_images=user_images or [],
            answer_images=answer_images or [],
        )
        session.turns.append(turn)
        if len(session.turns) > self._max_turns:
            session.turns = session.turns[-self._max_turns:]
        session.last_active = time.time()

    def format_history(self, session_id: str) -> str:
        """将最近 N 轮对话格式化为可注入 Prompt 的历史文本。"""
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
        """为检索 query 生成简短上文语境，用于指代消解。"""
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
        """删除指定会话。"""
        with self._lock:
            self._sessions.pop(session_id.strip(), None)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]


# ------------------------------------------------------------------
# 模块级单例
# ------------------------------------------------------------------

_store: Any = None
_lock = threading.Lock()


def get_conversation_store() -> Any:
    """返回会话存储单例：优先 Redis（与 ``REDIS_URL`` 等配置一致），失败回退内存。"""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = _build_conversation_store_backend()
    return _store


def _build_conversation_store_backend() -> Any:
    from app.core.config import settings as _settings

    if not _settings.conversation_redis_enabled:
        return ConversationStore()
    url = (_settings.redis_url or "").strip()
    if not url:
        return ConversationStore()
    try:
        import redis as redis_lib  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        print("[WARN] 未安装 redis 包，ConversationStore 回退内存。pip install redis")
        return ConversationStore()
    try:
        client = redis_lib.Redis.from_url(url, decode_responses=True)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Redis 不可用，ConversationStore 回退内存: {exc}")
        return ConversationStore()
    from app.services.conversation_store_redis import RedisConversationStore

    return RedisConversationStore(client)


def reset_conversation_store_for_tests() -> None:
    """测试用：重置会话存储单例。"""
    global _store
    with _lock:
        _store = None


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------

import re

_PIC_RE = re.compile(r"<PIC>")


def _strip_pic(text: str) -> str:
    """去除 <PIC> 标记，历史中不需要图片占位。"""
    return _PIC_RE.sub("", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
