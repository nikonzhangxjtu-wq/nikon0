from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from app.services.pipeline import ChatPipeline
from app.services.router import RouteDecision
from app.services.memory.v3.adapters import EvidenceAdapterPipeline
from app.services.memory.v3.evidence_packet import TurnEvidencePacketBuilder
from app.services.memory.v3.llm_judge import LlmMemoryJudge
from app.services.memory.v3.manager import MemoryManagerV3, get_memory_manager_v3, reset_memory_manager_v3_for_tests
from app.services.memory.v3.episodic_store import MilvusEpisodicMemoryV3Store
from app.services.memory.v3.read_planner import MemoryReadPlanner
from app.services.memory.v3.renderer import MemoryRenderer
from app.services.memory.v3.session_store import InMemorySessionMemoryV3Store
from app.services.memory.v3.types import (
    LlmMemoryJudgement,
    MemoryReadCandidate,
    MemoryReadRequest,
    ObservationCandidate,
    RawEvidence,
    TurnEvidencePacket,
    WriteDecision,
)
from app.services.memory.v3.write_gate import WriteGate


def test_turn_evidence_packet_builder_keeps_branch_and_tool_context() -> None:
    packet = TurnEvidencePacketBuilder.build(
        session_id="sid",
        user_id="u1",
        question="我的 AC900 报 E2",
        answer="已记录。",
        route_domain_hint="case_intake",
        route_needs_rag=False,
        branch_name="case_intake",
        recent_history="用户: 之前报修",
        rag_context="",
        branch_result={"case_status": "pending"},
        tool_results=[{"case_id": "CASE-1", "status": "submitted"}],
    )

    assert packet.turn_id.startswith("turn_")
    assert packet.branch_name == "case_intake"
    assert packet.branch_result == {"case_status": "pending"}
    assert packet.tool_results[0]["case_id"] == "CASE-1"


def test_user_utterance_adapter_extracts_non_tool_session_facts() -> None:
    packet = TurnEvidencePacket(
        session_id="sid",
        user_id=None,
        turn_id="t1",
        timestamp=1.0,
        question="我的 AC900 显示 E2，我已经断电重启过了，还是不行",
        answer="建议继续排查。",
        route_domain_hint="customer_service",
        route_needs_rag=False,
        branch_name="no_rag",
    )

    raw = EvidenceAdapterPipeline().collect(packet)
    candidates = [c for item in raw for c in item.to_candidates()]
    pairs = {(c.kind, c.value) for c in candidates}

    assert ("product_model", "AC900") in pairs
    assert ("fault_code", "E2") in pairs
    assert ("attempted_action", "断电重启") in pairs
    assert any(c.scope_hint == "session" for c in candidates)


def test_write_gate_blocks_rag_knowledge_but_allows_user_feedback() -> None:
    gate = WriteGate()
    rag_fact = ObservationCandidate(
        kind="manual_step",
        value="先打开后盖再取出滤网",
        source="rag",
        confidence=0.95,
        evidence_text="手册步骤",
        write_intent="observe",
        scope_hint="profile",
    )
    user_feedback = ObservationCandidate(
        kind="attempted_action",
        value="清洗滤网",
        source="user_current",
        confidence=0.88,
        evidence_text="我已经清洗滤网了",
        write_intent="observe",
        scope_hint="session",
    )

    decisions = gate.decide([rag_fact, user_feedback], user_key=None)

    assert decisions[0].action == "discard"
    assert "RAG" in decisions[0].reason
    assert decisions[1].action == "upsert_session"


def test_write_gate_profile_requires_explicit_intent_for_phone() -> None:
    gate = WriteGate()
    plain_phone = ObservationCandidate(
        kind="phone",
        value="13800138000",
        source="user_current",
        confidence=0.95,
        evidence_text="手机号 13800138000",
        write_intent="observe",
        scope_hint="profile",
        pii_level="high",
    )
    remembered_phone = ObservationCandidate(
        kind="phone",
        value="13800138000",
        source="user_current",
        confidence=0.95,
        evidence_text="以后默认用 13800138000 联系",
        write_intent="remember",
        scope_hint="profile",
        pii_level="high",
    )

    decisions = gate.decide([plain_phone, remembered_phone], user_key="user:alice")

    assert decisions[0].action == "upsert_session"
    assert "降级" in decisions[0].reason
    assert decisions[1].action == "upsert_profile"


def test_llm_judge_validates_json_and_does_not_write_directly() -> None:
    judge = LlmMemoryJudge(
        call_model=lambda _prompt: """
        {
          "should_write": true,
          "write_intent": "observe",
          "target_scope": "session",
          "confidence": 0.88,
          "reason": "用户明确反馈已尝试操作",
          "resolved_references": {"这个": "AC900 E2 故障"},
          "observations": [
            {
              "kind": "attempted_action",
              "value": "断电重启",
              "confidence": 0.9,
              "evidence_text": "我已经断电重启过了"
            }
          ]
        }
        """
    )

    result = judge.judge(
        packet=TurnEvidencePacket(
            session_id="sid",
            user_id=None,
            turn_id="t1",
            timestamp=1.0,
            question="我已经断电重启过了",
            answer="已记录。",
            route_domain_hint="customer_service",
            route_needs_rag=False,
            branch_name="no_rag",
        ),
        raw_evidence=[RawEvidence(source="user_current", text="我已经断电重启过了")],
        current_session_memory={},
    )

    assert isinstance(result, LlmMemoryJudgement)
    assert result.observations[0].kind == "attempted_action"
    assert result.observations[0].source == "llm_judge"


def test_session_store_upserts_atoms_and_issue_thread_without_tool() -> None:
    store = InMemorySessionMemoryV3Store()
    manager = MemoryManagerV3(session_store=store, llm_judge=None)
    packet = TurnEvidencePacket(
        session_id="sid",
        user_id=None,
        turn_id="t1",
        timestamp=1.0,
        question="我的 AC900 显示 E2，我已经断电重启过了",
        answer="已记录。",
        route_domain_hint="customer_service",
        route_needs_rag=False,
        branch_name="no_rag",
    )

    trace = manager.observe_and_write(packet)
    session = store.get("sid")

    assert trace.write_session_count >= 3
    assert session.active_issue_thread_id is not None
    active = session.issue_threads[session.active_issue_thread_id]
    assert active.product_model == "AC900"
    assert "E2" in active.fault_codes
    assert "断电重启" in active.attempted_actions


def test_read_planner_renderer_redacts_and_prioritizes_session_memory() -> None:
    planner = MemoryReadPlanner()
    request = planner.plan(
        session_id="sid",
        user_id="alice",
        question="这个型号之前的故障还要怎么处理？",
        recent_history="用户: AC900 显示 E2",
        route_domain_hint="customer_service",
    )

    assert request.include_session is True
    assert request.include_episodic is True

    renderer = MemoryRenderer()
    result = renderer.render(
        MemoryReadRequest(
            session_id="sid",
            user_id="alice",
            query="联系我",
            intents=["profile"],
            entities={},
            include_session=True,
            include_profile=True,
            include_episodic=False,
            budget_tokens=120,
            reason="test",
        ),
        [
            MemoryReadCandidate(
                text="联系电话: 13800138000",
                source_scope="profile",
                source_id="a1",
                score=50,
                reason="profile",
                kind="phone",
            )
        ],
    )

    assert "138****8000" in result.rendered_context
    assert "13800138000" not in result.rendered_context


def test_pipeline_v3_memory_writes_no_rag_session_facts() -> None:
    reset_memory_manager_v3_for_tests()
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="customer_service",
        reason="客服问题",
        confidence=0.8,
        strategy="test",
    )
    generator = MagicMock()
    generator.generate.return_value = "已记录你的情况。"
    vision = MagicMock()
    vision.summarize_images.return_value = ""
    pipeline = ChatPipeline(router=router, generator=generator, vision=vision)

    with (
        patch("app.services.pipeline.settings.memory_enabled", True),
        patch("app.services.pipeline.settings.memory_version", "v3"),
        patch("app.services.memory.v3.manager.settings.memory_enabled", True),
        patch("app.services.memory.v3.manager.settings.memory_v3_llm_judge_enabled", False),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="PROMPT"),
    ):
        pipeline.run("我的 AC900 显示 E2，我已经断电重启过了", images=[], session_id="sid-v3")

    session = get_memory_manager_v3().session_store.get("sid-v3")
    active = session.issue_threads[session.active_issue_thread_id]

    assert active.product_model == "AC900"
    assert "E2" in active.fault_codes
    assert "断电重启" in active.attempted_actions


def test_build_user_memory_v2_collection_schema(monkeypatch) -> None:
    fake_pymilvus = types.SimpleNamespace(
        DataType=types.SimpleNamespace(
            VARCHAR="VARCHAR",
            FLOAT_VECTOR="FLOAT_VECTOR",
            DOUBLE="DOUBLE",
        ),
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
    monkeypatch.setattr("app.services.milvus_create.settings.memory_v3_episodic_collection", "user_memory_v2")
    client = FakeClient()

    milvus_create.build_user_memory_v2_collection(client, vector_dim=3)

    field_names = {field[0] for field in FakeClient.schema.fields}
    assert {
        "event_id",
        "user_key",
        "event_type",
        "title",
        "summary",
        "product_model",
        "case_id",
        "issue_thread_id",
        "dense_vector",
        "created_at",
        "expire_ts",
    } <= field_names
    indexed_fields = {idx.get("field_name") for idx in FakeClient.index_params.indexes}
    assert {"dense_vector", "user_key", "event_type", "product_model", "case_id"} <= indexed_fields


def test_milvus_episodic_store_upserts_user_memory_v2_rows(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.rows = []

        def upsert(self, *, collection_name: str, data: list[dict]) -> None:
            self.collection_name = collection_name
            self.rows.extend(data)

    monkeypatch.setattr("app.services.memory.v3.episodic_store.settings.memory_v3_episodic_collection", "user_memory_v2")
    client = FakeClient()
    store = MilvusEpisodicMemoryV3Store(client, embed_fn=lambda text: [0.1, 0.2, 0.3])
    candidate = ObservationCandidate(
        kind="case_status",
        value="submitted",
        source="tool",
        confidence=0.98,
        evidence_text="工单 CASE-1 已提交",
        scope_hint="episodic",
        write_intent="observe",
    )

    store.apply_decisions(
        "alice",
        [WriteDecision(action="upsert_episodic", reason="submitted", candidate=candidate, confidence=0.98)],
        turn_id="t1",
    )

    assert client.collection_name == "user_memory_v2"
    assert client.rows[0]["event_type"] == "case_submitted"
    assert client.rows[0]["summary"] == "工单 CASE-1 已提交"
    assert client.rows[0]["dense_vector"] == [0.1, 0.2, 0.3]
