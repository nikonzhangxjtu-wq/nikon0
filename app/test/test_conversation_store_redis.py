"""RedisConversationStore 单元测试（Mock Redis，不依赖本机 Redis）。"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.conversation_store_redis import RedisConversationStore


def test_redis_conversation_save_roundtrip():
    fake = MagicMock()
    fake.get.return_value = None
    store = RedisConversationStore(fake)

    sid = "sess_1"
    session = store.get_or_create(sid)
    assert session.session_id == sid
    key = fake.set.call_args[0][0]
    assert "conversation" in key

    fake.get.return_value = fake.set.call_args[0][1]
    session2 = store.get_or_create(sid)
    assert session2.session_id == sid


def test_format_history_after_add_turn():
    fake = MagicMock()

    def get_side_effect(k: str) -> str | None:
        _ = k
        return _stored.get("v")

    def set_side_effect(k: str, v: str, **kwargs: object) -> bool:
        _ = kwargs
        _stored["k"], _stored["v"] = k, v
        return True

    _stored: dict[str, str] = {}
    fake.get.side_effect = get_side_effect
    fake.set.side_effect = set_side_effect

    store = RedisConversationStore(fake)
    store.add_turn("s2", question="你好", answer="您好", user_images=[], answer_images=[])
    hist = store.format_history("s2")
    assert "用户:" in hist
    assert "你好" in hist


if __name__ == "__main__":
    test_redis_conversation_save_roundtrip()
    test_format_history_after_add_turn()
    print("[OK] test_conversation_store_redis passed")
