"""长期用户画像存储。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import settings
from app.services.memory.types import SessionFacts, UserProfile


class UserProfileStore:
    """Redis JSON 用户画像；无 Redis 时使用进程内回退。"""

    def __init__(self, client: object | None = None) -> None:
        self._r = client
        self._prefix = (settings.redis_conversation_key_prefix or "kf").strip().rstrip(":")
        self._ttl = settings.memory_user_profile_ttl_seconds
        self._memory: dict[str, UserProfile] = {}

    @staticmethod
    def hash_user_key(raw: str) -> str:
        value = (raw or "").strip()
        if not value:
            return ""
        if value.startswith("sha256:"):
            return value
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def key_for(self, user_key: str) -> str:
        hashed = self.hash_user_key(user_key)
        digest = hashed.removeprefix("sha256:")
        return f"{self._prefix}:user_profile:{digest}"

    def get(self, user_key: str) -> UserProfile | None:
        hashed = self.hash_user_key(user_key)
        if not hashed:
            return None
        if self._r is None:
            return self._memory.get(hashed)
        raw = self._r.get(self.key_for(hashed))
        if not raw:
            return None
        try:
            obj: Any = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return UserProfile.from_dict(obj if isinstance(obj, dict) else {}, user_key=hashed)

    def save(self, profile: UserProfile) -> None:
        hashed = self.hash_user_key(profile.user_key)
        if not hashed:
            return
        normalized = UserProfile.from_dict(profile.to_dict(), user_key=hashed)
        payload = json.dumps(normalized.to_dict(), ensure_ascii=False)
        if self._r is None:
            self._memory[hashed] = normalized
            return
        self._r.set(self.key_for(hashed), payload, ex=max(3600, int(self._ttl)))

    def upsert_from_facts(self, user_key: str, facts: SessionFacts) -> UserProfile | None:
        hashed = self.hash_user_key(user_key)
        if not hashed:
            return None
        profile = self.get(hashed) or UserProfile(user_key=hashed)
        updated = profile.merge_facts(facts)
        self.save(updated)
        return updated

    def forget(self, user_key: str) -> None:
        hashed = self.hash_user_key(user_key)
        if not hashed:
            return
        if self._r is None:
            self._memory.pop(hashed, None)
            return
        self._r.delete(self.key_for(hashed))


def build_user_profile_store() -> UserProfileStore:
    if not (settings.memory_enabled and settings.memory_user_profile_enabled):
        return UserProfileStore()
    try:
        import redis as redis_lib  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return UserProfileStore()
    try:
        client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
    except Exception:  # noqa: BLE001
        return UserProfileStore()
    return UserProfileStore(client)
