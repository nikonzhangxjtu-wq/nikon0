from __future__ import annotations

import sys
import types

from app.services.context.fact_extractor import extract_critical_facts
from app.utils.prompts.context import PromptContext
from app.utils.prompts.registry import compose_generation_prompt


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.deleted: list[str] = []

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.deleted.append(key)
            self.data.pop(key, None)


class FakeMilvus:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted_filter = ""

    def search(self, **kwargs):
        user_filter = kwargs.get("filter", "")
        hits = []
        for row in self.rows:
            if row["user_key"] not in user_filter:
                continue
            hits.append({"score": 0.9, "entity": row})
        return [hits]

    def upsert(self, *, collection_name: str, data: list[dict]) -> None:
        self.rows.extend(data)

    def delete(self, *, collection_name: str, filter: str) -> None:
        self.deleted_filter = filter


class FakeEpisodicStore:
    def __init__(self) -> None:
        self.notes = []
        self.forgotten = ""

    def search(self, *, user_key: str, query: str, top_k: int | None = None):
        return list(self.notes)

    def upsert(self, notes) -> None:
        self.notes.extend(notes)

    def forget(self, user_key: str) -> None:
        self.forgotten = user_key


def test_memory_manager_is_disabled_by_default(monkeypatch):
    from app.services.memory.manager import MemoryManager

    monkeypatch.setattr("app.services.memory.manager.settings.memory_enabled", False)

    manager = MemoryManager()
    bundle = manager.read(session_id="s1", user_id="u1", query="空调 E2 怎么办")

    assert bundle.is_empty()
    assert bundle.render() == ""


def test_session_memory_accumulates_critical_facts_and_renders_context(monkeypatch):
    from app.services.memory.session_memory import SessionMemoryStore

    monkeypatch.setattr("app.services.memory.session_memory.settings.memory_session_summary_trigger_turns", 2)
    store = SessionMemoryStore(FakeRedis())

    facts = extract_critical_facts(
        question="订单 202605300001 的产品型号 AC900 出现 E2，我要报修，手机号 13800138000"
    )
    session = store.merge_turn_facts(
        "session-1",
        facts,
        question="AC900 出现 E2",
        answer="建议先断电重启，若仍报错再报修。",
    )
    session = store.merge_turn_facts(
        "session-1",
        facts,
        question="我已经断电重启过",
        answer="记录为已尝试步骤。",
    )

    rendered = session.render()

    assert "202605300001" in rendered
    assert "13800138000" in rendered
    assert "AC900" in rendered
    assert "E2" in rendered
    assert "会话摘要" in rendered
    assert session.turn_count == 2


def test_memory_context_is_injected_into_prompt():
    prompt = compose_generation_prompt(
        PromptContext(
            question="这个型号之前是什么问题？",
            need_rag=False,
            domain_hint="customer_service",
            memory_context="[记忆]\n产品/型号: AC900\n历史问题: E2 故障",
        )
    )

    assert "[记忆]" in prompt
    assert "AC900" in prompt
    assert "E2 故障" in prompt


def test_user_profile_store_hashes_and_merges_session_facts(monkeypatch):
    from app.services.memory.session_memory import SessionMemoryStore
    from app.services.memory.user_profile_store import UserProfileStore

    monkeypatch.setattr("app.services.memory.user_profile_store.settings.memory_user_profile_ttl_seconds", 3600)
    session_store = SessionMemoryStore(FakeRedis())
    profile_store = UserProfileStore(FakeRedis())

    facts = extract_critical_facts(
        question="手机号 13800138000，型号 AC900，订单 202605300001，E2 故障需要报修"
    )
    session = session_store.merge_turn_facts("sid", facts, question="E2 故障", answer="已记录")
    profile = profile_store.upsert_from_facts("user:alice", session.facts)

    assert profile is not None
    assert profile.user_key.startswith("sha256:")
    assert "AC900" in profile.products
    assert "13800138000" in profile.contact_phones
    assert "202605300001" in profile.active_orders
    assert "E2" in ",".join(profile.historical_issues)


def test_memory_manager_resolves_user_id_before_phone(monkeypatch):
    from app.services.memory.manager import MemoryManager
    from app.services.memory.session_memory import SessionMemoryStore
    from app.services.memory.user_profile_store import UserProfileStore

    monkeypatch.setattr("app.services.memory.manager.settings.memory_enabled", True)
    session_store = SessionMemoryStore(FakeRedis())
    manager = MemoryManager(
        session_store=session_store,
        profile_store=UserProfileStore(FakeRedis()),
    )
    facts = extract_critical_facts(question="手机号 13800138000")
    session = session_store.merge_turn_facts("sid", facts)

    user_key = manager.resolve_user_key(user_id="alice", session_memory=session)
    phone_key = manager.resolve_user_key(user_id=None, session_memory=session)

    assert user_key != phone_key
    assert user_key.startswith("sha256:")
    assert phone_key.startswith("sha256:")


def test_episodic_store_upserts_and_searches_by_user_key(monkeypatch):
    from app.services.memory.episodic_store import EpisodicMemoryStore
    from app.services.memory.types import MemoryNote
    from app.services.memory.user_profile_store import UserProfileStore

    monkeypatch.setattr("app.services.memory.episodic_store.settings.memory_enabled", True)
    monkeypatch.setattr("app.services.memory.episodic_store.settings.memory_episodic_enabled", True)
    monkeypatch.setattr("app.services.memory.episodic_store.settings.memory_episodic_score_threshold", 0.1)
    client = FakeMilvus()
    store = EpisodicMemoryStore(client, embed_fn=lambda text: [0.1, 0.2, 0.3])

    store.upsert([
        MemoryNote(memory_id="m1", user_key="user:alice", memory_text="用户 AC900 曾出现 E2 故障")
    ])
    notes = store.search(user_key="user:alice", query="AC900 故障")

    assert client.rows[0]["user_key"] == UserProfileStore.hash_user_key("user:alice")
    assert notes[0].memory_text == "用户 AC900 曾出现 E2 故障"


def test_build_user_memory_collection_creates_schema(monkeypatch):
    fake_pymilvus = types.SimpleNamespace(
        DataType=types.SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR", DOUBLE="DOUBLE"),
        MilvusClient=None,
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    from app.services import milvus_create

    class FakeSchema:
        def __init__(self) -> None:
            self.fields: list[tuple[str, object, dict]] = []

        def add_field(self, *, field_name, datatype, **kwargs) -> None:
            self.fields.append((field_name, datatype, kwargs))

    class FakeIndexParams:
        def __init__(self) -> None:
            self.indexes: list[dict] = []

        def add_index(self, **kwargs) -> None:
            self.indexes.append(kwargs)

    class FakeClient:
        schema = FakeSchema()
        index_params = FakeIndexParams()

        @staticmethod
        def create_schema(auto_id: bool, enable_dynamic_field: bool):
            return FakeClient.schema

        @staticmethod
        def prepare_index_params():
            return FakeClient.index_params

        def has_collection(self, collection_name: str) -> bool:
            return False

        def create_collection(self, *, collection_name: str, schema) -> None:
            self.created = (collection_name, schema)

        def create_index(self, *, collection_name: str, index_params) -> None:
            self.indexed = (collection_name, index_params)

        def load_collection(self, *, collection_name: str) -> None:
            self.loaded = collection_name

    monkeypatch.setattr("app.services.milvus_create.MilvusClient", FakeClient)
    client = FakeClient()

    milvus_create.build_user_memory_collection(client, vector_dim=3)

    field_names = {field[0] for field in FakeClient.schema.fields}
    assert {"memory_id", "user_key", "memory_text", "dense_vector", "expire_ts"} <= field_names
    assert any(idx.get("field_name") == "dense_vector" for idx in FakeClient.index_params.indexes)


def test_memory_manager_consolidates_session_into_profile_and_notes(monkeypatch):
    from app.services.memory.consolidator import ConsolidationResult
    from app.services.memory.manager import MemoryManager
    from app.services.memory.session_memory import SessionMemoryStore
    from app.services.memory.types import MemoryNote
    from app.services.memory.user_profile_store import UserProfileStore

    class FakeConsolidator:
        def consolidate(self, *, session, user_profile, recent_turns=None):
            return ConsolidationResult(
                session_summary="用户 AC900 出现 E2，已尝试断电重启。",
                memory_notes=[
                    MemoryNote(
                        memory_id="note-1",
                        user_key=user_profile.user_key if user_profile else "",
                        memory_text="用户 AC900 出现 E2，已尝试断电重启。",
                        source_session=session.session_id,
                    )
                ],
            )

    monkeypatch.setattr("app.services.memory.manager.settings.memory_enabled", True)
    monkeypatch.setattr("app.services.memory.manager.settings.memory_user_profile_enabled", True)
    monkeypatch.setattr("app.services.memory.manager.settings.memory_episodic_enabled", True)
    session_store = SessionMemoryStore(FakeRedis())
    profile_store = UserProfileStore(FakeRedis())
    episodic = FakeEpisodicStore()
    manager = MemoryManager(
        session_store=session_store,
        profile_store=profile_store,
        episodic_store=episodic,
        consolidator=FakeConsolidator(),
    )
    manager.write_turn(
        session_id="sid",
        user_id="alice",
        question="型号 AC900 报 E2，手机号 13800138000",
        answer="建议断电重启。",
    )

    bundle = manager.consolidate(session_id="sid", user_id="alice")

    assert "会话摘要" in bundle.render()
    assert "AC900" in bundle.render()
    assert episodic.notes
    assert episodic.notes[0].memory_text == "用户 AC900 出现 E2，已尝试断电重启。"


def test_memory_manager_forget_deletes_profile_and_episodic_notes(monkeypatch):
    from app.services.memory.manager import MemoryManager
    from app.services.memory.session_memory import SessionMemoryStore
    from app.services.memory.user_profile_store import UserProfileStore

    monkeypatch.setattr("app.services.memory.manager.settings.memory_enabled", True)
    profile_store = UserProfileStore(FakeRedis())
    episodic = FakeEpisodicStore()
    manager = MemoryManager(
        session_store=SessionMemoryStore(FakeRedis()),
        profile_store=profile_store,
        episodic_store=episodic,
    )
    bundle = manager.write_turn(
        session_id="sid",
        user_id="alice",
        question="手机号 13800138000，型号 AC900",
        answer="已记录。",
    )

    manager.forget(bundle.user_key)

    assert profile_store.get(bundle.user_key) is None
    assert episodic.forgotten == bundle.user_key


def test_query_rewriter_includes_memory_context_in_prompt():
    from app.services.query_rewriter import QueryRewriter

    prompt = QueryRewriter()._build_prompt(
        "这个型号还报 E2 怎么办？",
        "[对话上文]\n用户问「空调故障」",
        memory_context="[记忆]\n产品/型号: AC900",
    )

    assert "产品/型号: AC900" in prompt
    assert "当前问题：这个型号还报 E2 怎么办？" in prompt
