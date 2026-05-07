"""Case intake Redis store 单元测试（不依赖真实 Redis 进程）。"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.skills.case_intake_redis_store import RedisCaseIntakeStore
from app.services.skills.case_intake_types import CaseState


def test_redis_store_roundtrip_json_string():
    fake = MagicMock()
    store = RedisCaseIntakeStore(fake, key_prefix="kf_ut", ttl_seconds=300)

    state = CaseState(intent="repair", slots={"issue": "电机不转", "contact_phone": "13800138000"})
    store.save("sid_a", state)

    key = fake.set.call_args[0][0]
    payload = fake.set.call_args[0][1]
    assert key.startswith("kf_ut:case_intake:")
    assert "电机不转" in payload
    assert fake.set.call_args[1].get("ex") == 300

    fake.get.return_value = payload
    loaded = store.load("sid_a")
    assert loaded is not None
    assert loaded.intent == "repair"
    assert loaded.slots["issue"] == "电机不转"
    assert loaded.slots["contact_phone"] == "13800138000"


if __name__ == "__main__":
    test_redis_store_roundtrip_json_string()
    print("[OK] test_case_intake_redis_store passed")
