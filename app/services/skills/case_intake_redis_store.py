"""Case Intake 状态 Redis 持久化。

Redis 侧使用 **STRING** 类型存 JSON（不是 Hash / List），便于整份读写与 TTL。
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Protocol

from app.core.config import settings
from app.services.skills.case_intake_types import CaseState


class CaseIntakeStateStore(Protocol):
    def load(self, session_id: str) -> CaseState | None: ...
    def save(self, session_id: str, state: CaseState) -> None: ...
    def delete(self, session_id: str) -> None: ...


class MemoryCaseIntakeStore:
    """进程内字典（无 Redis 或回退时使用）。"""

    def __init__(self) -> None:
        self._data: dict[str, CaseState] = {}

    def load(self, session_id: str) -> CaseState | None:
        return self._data.get(session_id)

    def save(self, session_id: str, state: CaseState) -> None:
        self._data[session_id] = state

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisCaseIntakeStore:
    """Redis STRING + JSON 存 CaseState。"""

    def __init__(self, client: object, *, key_prefix: str, ttl_seconds: int) -> None:
        self._r = client
        self._prefix = (key_prefix or "kf").strip().rstrip(":")
        self._ttl = max(60, int(ttl_seconds))

    def _key(self, session_id: str) -> str:
        sid = (session_id or "").strip() or "__default__"
        digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
        return f"{self._prefix}:case_intake:{digest}"

    def load(self, session_id: str) -> CaseState | None:
        raw = self._r.get(self._key(session_id))
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        intent = str(obj.get("intent", "repair")).strip() or "repair"
        slots_raw = obj.get("slots")
        slots: dict[str, str] = {}
        if isinstance(slots_raw, dict):
            for k, v in slots_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    slots[k] = v
        return CaseState(intent=intent, slots=slots)

    def save(self, session_id: str, state: CaseState) -> None:
        payload = json.dumps(
            {"intent": state.intent, "slots": state.slots},
            ensure_ascii=False,
        )
        self._r.set(self._key(session_id), payload, ex=self._ttl)

    def delete(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


_store: CaseIntakeStateStore | None = None
_store_lock = threading.Lock()


def get_case_intake_state_store() -> CaseIntakeStateStore:
    """单例：优先 Redis，失败或未启用则内存。"""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        if not settings.case_intake_redis_enabled:
            _store = MemoryCaseIntakeStore()
            return _store
        url = (settings.redis_url or "").strip()
        if not url:
            _store = MemoryCaseIntakeStore()
            return _store
        try:
            import redis as redis_lib  # type: ignore[import-untyped]
        except ModuleNotFoundError:
            print("[WARN] 未安装 redis 包，CaseIntake 状态回退内存存储。pip install redis")
            _store = MemoryCaseIntakeStore()
            return _store
        try:
            client = redis_lib.Redis.from_url(url, decode_responses=True)
            client.ping()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Redis 连接失败，CaseIntake 状态回退内存存储: {exc}")
            _store = MemoryCaseIntakeStore()
            return _store
        _store = RedisCaseIntakeStore(
            client,
            key_prefix=settings.redis_case_intake_key_prefix,
            ttl_seconds=settings.case_intake_redis_ttl_seconds,
        )
        return _store


def reset_case_intake_state_store_for_tests() -> None:
    """测试用：重置单例。"""
    global _store
    with _store_lock:
        _store = None
