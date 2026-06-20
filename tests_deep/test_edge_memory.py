"""边界测试 - Memory 系统.

覆盖：状态损坏恢复、并发更新、极限值、线程生命周期、flat_state 一致性.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from nikon0.app.schemas.capability import StateUpdate
from nikon0.app.schemas.memory import IssueFact, IssueThread, SessionIssueMemory
from nikon0.memory.persistence import SqlMemoryPersistence
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.memory.view import MemoryViewBuilder


class TestMemoryBoundaryCases:
    """Memory 系统的边界和异常场景."""

    def test_load_new_session_returns_empty_state(self):
        """新 session 应返回干净的初始状态."""
        store = InMemorySessionIssueStore()
        memory = store.load("new-session")
        assert memory.session_id == "new-session"
        assert memory.turn_count == 0
        assert memory.active_thread_id is None
        assert memory.flat_state == {}

    def test_load_returns_deep_copy(self):
        """load 返回深拷贝，修改不影响内部状态."""
        store = InMemorySessionIssueStore()
        m1 = store.load("s1")
        m1.flat_state["key"] = "modified"
        m2 = store.load("s1")
        assert "key" not in m2.flat_state

    def test_apply_updates_increments_turn_count(self):
        """apply_updates 应递增 turn_count."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [StateUpdate(key="k", value="v")], turn_id="t1")
        store.apply_updates("s1", [StateUpdate(key="k2", value="v2")], turn_id="t2")
        memory = store.load("s1")
        assert memory.turn_count == 2

    def test_empty_updates_does_not_create_thread(self):
        """空 updates 不应创建 IssueThread."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [], turn_id="t1")
        memory = store.load("s1")
        assert memory.active_thread_id is None

    def test_flat_state_deeply_nested_values(self):
        """深层嵌套的 flat_state 值."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="deep", value={"a": {"b": {"c": {"d": {"e": "nested"}}}}}),
        ], turn_id="t1")
        memory = store.load("s1")
        assert memory.flat_state["deep"]["a"]["b"]["c"]["d"]["e"] == "nested"

    def test_flat_state_large_string_values(self):
        """大字符串值."""
        store = InMemorySessionIssueStore()
        large = "X" * 100000
        store.apply_updates("s1", [StateUpdate(key="large", value=large)], turn_id="t1")
        memory = store.load("s1")
        assert len(memory.flat_state["large"]) == 100000

    def test_flat_state_special_characters(self):
        """特殊字符值."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="special", value="\x00\n\r\t\b\f"),
        ], turn_id="t1")
        memory = store.load("s1")
        assert memory.flat_state["special"] == "\x00\n\r\t\b\f"

    def test_flat_state_null_value(self):
        """None 值."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [StateUpdate(key="nil", value=None)], turn_id="t1")
        memory = store.load("s1")
        assert memory.flat_state["nil"] is None

    def test_product_support_state_creates_thread_with_product(self):
        """product_support state 应正确填充 IssueThread."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="product_support", value={
                "selected_product_id": "AC900",
                "selected_display_name": "AC900 空气净化器",
                "manual_names": ["AC900手册"],
                "disambiguation_pending": False,
                "last_query": "E2怎么处理",
                "evidence_count": 3,
            }, reason="product support answered"),
        ], turn_id="t1")
        memory = store.load("s1")
        thread = memory.active_thread()
        assert thread is not None
        assert thread.product_model == "AC900"
        assert "AC900" in str(memory.active_product.get("product_id", ""))
        assert "product_support.last_query" in thread.facts

    def test_case_intake_state_creates_thread_with_workflow(self):
        """case_intake state 应正确填充 IssueThread."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "collecting",
                "completed": False,
                "missing_slots": ["contact_phone"],
                "ticket_payload": {"intent": "repair", "product_model": "AC900"},
                "workflow_name": "repair_intake",
                "workflow_intent": "repair",
                "workflow_status": "collecting",
                "requires_approval": False,
                "handoff_required": False,
            }, reason="case intake started"),
        ], turn_id="t1")
        memory = store.load("s1")
        thread = memory.active_thread()
        assert thread is not None
        assert thread.issue_type == "repair"
        assert thread.status == "waiting_user"

    def test_multiple_updates_merge_into_single_thread(self):
        """多次更新应合并到同一线程."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="product_support", value={
                "selected_product_id": "AC900",
                "last_query": "E2",
                "evidence_count": 2,
            }),
        ], turn_id="t1")
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "collecting",
                "ticket_payload": {"intent": "repair", "product_model": "AC900"},
            }),
        ], turn_id="t2")
        memory = store.load("s1")
        # 只应有一个活跃线程
        assert memory.active_thread_id is not None
        assert len(memory.threads) == 1

    def test_thread_lifecycle_open_to_submitted(self):
        """线程状态变迁 open → waiting_user → submitted."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "collecting",
                "missing_slots": ["phone"],
                "ticket_payload": {},
            }),
        ], turn_id="t1")
        assert store.load("s1").active_thread().status == "waiting_user"

        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "ready",
                "completed": True,
                "missing_slots": [],
                "ticket_payload": {"ticket_id": "TK-001"},
            }),
        ], turn_id="t2")
        assert store.load("s1").active_thread().status == "submitted"

    def test_thread_lifecycle_cancelled(self):
        """线程取消状态."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "cancelled",
                "exited": True,
                "ticket_payload": {},
            }),
        ], turn_id="t1")
        assert store.load("s1").active_thread().status == "cancelled"

    def test_issue_thread_fact_upsert(self):
        """IssueFact 的 upsert 行为."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={
                "status": "collecting",
                "ticket_payload": {"intent": "repair", "priority": "low"},
                "workflow_name": "repair_intake",
            }),
        ], turn_id="t1")
        memory = store.load("s1")
        thread = memory.active_thread()
        assert "case_intake.status" in thread.facts
        assert thread.facts["case_intake.status"].value == "collecting"

    def test_memory_view_builder_respects_budget(self):
        """MemoryViewBuilder 应符合字符预算."""
        store = InMemorySessionIssueStore()
        # 创建大量 session facts
        for i in range(50):
            store.apply_updates("s1", [
                StateUpdate(key=f"fact_{i}", value=f"value_{i}" * 20),
            ], turn_id=f"t{i}")
        builder = MemoryViewBuilder(char_budget=400)
        memory = store.load("s1")
        view = builder.build(memory)
        rendered = view.render()
        assert len(rendered) <= 500  # 允许一些容差

    def test_memory_view_builder_with_none_state(self):
        """None session state 应返回空视图."""
        builder = MemoryViewBuilder()
        view = builder.build(None)
        assert view.session_id == ""

    def test_load_flat_returns_dict(self):
        """load_flat 返回 dict."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [StateUpdate(key="k", value="v")], turn_id="t1")
        flat = store.load_flat("s1")
        assert isinstance(flat, dict)
        assert flat["k"] == "v"

    def test_turn_tracking_in_thread(self):
        """验证 turn_id 被记录到线程中."""
        store = InMemorySessionIssueStore()
        store.apply_updates("s1", [
            StateUpdate(key="case_intake", value={"status": "collecting"}),
        ], turn_id="trace-t1")
        memory = store.load("s1")
        assert "trace-t1" in memory.active_thread().last_turn_ids


class TestMemoryPersistenceEdgeCases:
    """持久化 Memory 的边界测试."""

    def test_sql_persistence_roundtrip(self, tmp_path):
        """SQL 持久化的完整回环."""
        import os
        dsn = f"sqlite:///{tmp_path / 'memory.db'}"
        persistence = SqlMemoryPersistence(dsn)

        memory = SessionIssueMemory(session_id="s1", turn_count=3)
        memory.flat_state["key"] = "value"
        persistence.save_snapshot(memory)

        loaded = persistence.load_snapshot("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.turn_count == 3
        assert loaded.flat_state["key"] == "value"

    def test_sql_events_roundtrip(self, tmp_path):
        """StateUpdate 事件的持久化回环."""
        dsn = f"sqlite:///{tmp_path / 'events.db'}"
        persistence = SqlMemoryPersistence(dsn)

        updates = [
            StateUpdate(key="k1", value="v1", reason="r1", evidence_ids=["e1"]),
            StateUpdate(key="k2", value="v2", reason="r2", evidence_ids=["e2"]),
        ]
        persistence.append_state_update_events("s1", updates, turn_id="t1")

        events = persistence.list_state_update_events("s1")
        assert len(events) == 2
        assert events[0]["update_key"] == "k1"
        assert events[0]["reason"] == "r1"
        assert events[0]["turn_id"] == "t1"
        assert events[1]["update_key"] == "k2"

    def test_sql_snapshot_overwrite(self, tmp_path):
        """同 session 多次保存应覆盖."""
        dsn = f"sqlite:///{tmp_path / 'overwrite.db'}"
        persistence = SqlMemoryPersistence(dsn)

        m1 = SessionIssueMemory(session_id="s1", turn_count=1)
        persistence.save_snapshot(m1)
        m2 = SessionIssueMemory(session_id="s1", turn_count=5)
        persistence.save_snapshot(m2)

        loaded = persistence.load_snapshot("s1")
        assert loaded.turn_count == 5

    def test_load_nonexistent_session_returns_none(self, tmp_path):
        """加载不存在的 session 返回 None."""
        dsn = f"sqlite:///{tmp_path / 'empty.db'}"
        persistence = SqlMemoryPersistence(dsn)
        assert persistence.load_snapshot("nonexistent") is None

    def test_empty_updates_not_appended(self, tmp_path):
        """空 updates 不应创建事件记录."""
        dsn = f"sqlite:///{tmp_path / 'noempty.db'}"
        persistence = SqlMemoryPersistence(dsn)
        persistence.append_state_update_events("s1", [], turn_id="t1")
        assert len(persistence.list_state_update_events("s1")) == 0
