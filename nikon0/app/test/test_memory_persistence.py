from __future__ import annotations

from pathlib import Path

from nikon0.app.schemas.capability import StateUpdate
from nikon0.memory.persistence import (
    RedisMysqlSessionIssueStore,
    SqlMemoryPersistence,
    build_memory_store_from_env,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    def ping(self) -> bool:
        return True


def test_redis_mysql_memory_store_persists_snapshot_and_state_update_events(tmp_path: Path) -> None:
    redis_client = FakeRedis()
    sql = SqlMemoryPersistence(f"sqlite:///{tmp_path / 'nikon0_memory.db'}")
    store = RedisMysqlSessionIssueStore(
        redis_client=redis_client,
        sql_persistence=sql,
        redis_key_prefix="test:nikon0",
        ttl_seconds=120,
    )

    updated = store.apply_updates(
        "persist-s1",
        [
            StateUpdate(
                key="product_support",
                value={
                    "selected_product_id": "air_purifier_ac900",
                    "selected_display_name": "AC900 空气净化器",
                    "last_query": "AC900 显示 E2 怎么办？",
                },
                reason="answered with evidence",
                evidence_ids=["ev1"],
            )
        ],
        turn_id="trace-persist-1",
    )

    assert updated.active_product["display_name"] == "AC900 空气净化器"
    assert redis_client.values["test:nikon0:session:persist-s1"]
    assert redis_client.expirations["test:nikon0:session:persist-s1"] == 120

    reloaded = RedisMysqlSessionIssueStore(
        redis_client=redis_client,
        sql_persistence=sql,
        redis_key_prefix="test:nikon0",
        ttl_seconds=120,
    ).load("persist-s1")
    events = sql.list_state_update_events("persist-s1")

    assert reloaded.active_product["display_name"] == "AC900 空气净化器"
    assert len(events) == 1
    assert events[0]["update_key"] == "product_support"
    assert events[0]["turn_id"] == "trace-persist-1"
    assert events[0]["evidence_ids"] == ["ev1"]


def test_redis_mysql_memory_store_restores_from_mysql_when_redis_is_empty(tmp_path: Path) -> None:
    sql = SqlMemoryPersistence(f"sqlite:///{tmp_path / 'nikon0_memory.db'}")
    first_redis = FakeRedis()
    first_store = RedisMysqlSessionIssueStore(
        redis_client=first_redis,
        sql_persistence=sql,
        redis_key_prefix="test:nikon0",
    )
    first_store.apply_updates(
        "restore-s1",
        [StateUpdate(key="case_intake", value={"status": "collecting"}, reason="collecting")],
        turn_id="trace-restore-1",
    )

    second_redis = FakeRedis()
    restored = RedisMysqlSessionIssueStore(
        redis_client=second_redis,
        sql_persistence=sql,
        redis_key_prefix="test:nikon0",
    ).load("restore-s1")

    assert restored.flat_state["case_intake"]["status"] == "collecting"
    assert second_redis.values["test:nikon0:session:restore-s1"]


def test_build_memory_store_from_env_falls_back_to_memory_without_configuration() -> None:
    class EmptySettings:
        nikon0_memory_store = "memory"

    store = build_memory_store_from_env(settings=EmptySettings())

    assert store.__class__.__name__ == "InMemorySessionIssueStore"
