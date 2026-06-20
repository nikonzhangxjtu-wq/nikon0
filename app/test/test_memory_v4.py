from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.manager import MemoryManagerV4, get_memory_manager_v4, reset_memory_manager_v4_for_tests
from app.services.memory.v4.reader import IssueReadPlanner
from app.services.memory.v4.store import InMemorySessionIssueMemoryStore, RedisSessionIssueMemoryStore
from app.services.memory.v4.types import IssueFactCandidate, StateChange
from app.services.pipeline import ChatPipeline
from app.services.router import RouteDecision


def packet(
    question: str,
    *,
    answer: str = "已记录。",
    session_id: str = "sid",
    turn_id: str = "turn_1",
    branch_name: str = "no_rag",
    route_needs_rag: bool = False,
    route_domain_hint: str = "customer_service",
    recent_history: str = "",
    rag_context: str = "",
    branch_result: dict | None = None,
) -> TurnEvidencePacket:
    return TurnEvidencePacket(
        session_id=session_id,
        user_id=None,
        turn_id=turn_id,
        timestamp=1.0,
        question=question,
        answer=answer,
        route_domain_hint=route_domain_hint,
        route_needs_rag=route_needs_rag,
        branch_name=branch_name,
        recent_history=recent_history,
        rag_context=rag_context,
        branch_result=branch_result,
    )


def active_thread(manager: MemoryManagerV4, session_id: str = "sid"):
    memory = manager.store.load(session_id)
    assert memory.active_thread_id is not None
    return memory.threads[memory.active_thread_id]


def active_values(thread, kind: str) -> list[str]:
    return [fact.value for fact in thread.facts.values() if fact.kind == kind and fact.status == "active"]


def test_v4_does_not_write_pure_howto_question() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)

    trace = manager.observe_and_write(
        packet(
            "AC900 的滤网怎么清洗？",
            branch_name="rag_manual",
            route_needs_rag=True,
            rag_context="[手册] 清洗滤网步骤",
        )
    )
    memory = manager.store.load("sid")

    assert trace.should_write is False
    assert memory.threads == {}


def test_v4_writes_non_tool_issue_state_and_evidence_refs() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)

    manager.observe_and_write(packet("我的 AC900 显示 E2，我已经断电重启过了，还是不行"))
    thread = active_thread(manager)

    assert thread.product_model == "AC900"
    assert "E2" in active_values(thread, "fault_code")
    assert "断电重启" in active_values(thread, "attempted_action")
    assert "还是不行" in active_values(thread, "symptom")
    assert thread.evidence_refs
    assert all(fact.evidence_ref_id in thread.evidence_refs for fact in thread.facts.values())


def test_v4_separates_multiple_product_issues() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)

    manager.observe_and_write(packet("我的 AC900 显示 E2", turn_id="t1"))
    manager.observe_and_write(packet("另一个 DW200 显示 F1，我清洗滤网了也不行", turn_id="t2"))
    memory = manager.store.load("sid")
    products = sorted(thread.product_model for thread in memory.threads.values())

    assert len(memory.threads) == 2
    assert products == ["AC900", "DW200"]


def test_v4_correction_only_replaces_target_fact() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)

    manager.observe_and_write(packet("我的 AC900 显示 E2", turn_id="t1"))
    manager.observe_and_write(packet("刚才说错了，不是 AC900，是 AC901", turn_id="t2"))
    thread = active_thread(manager)

    assert thread.product_model == "AC901"
    assert "AC901" in active_values(thread, "product_model")
    assert "AC900" not in active_values(thread, "product_model")
    assert any(
        fact.kind == "product_model" and fact.value == "AC900" and fact.status == "superseded"
        for fact in thread.facts.values()
    )


def test_v4_denial_marks_attempted_action_rejected() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)

    manager.observe_and_write(packet("我的 AC900 显示 E2，我已经断电重启过了", turn_id="t1"))
    manager.observe_and_write(packet("我没有断电重启过", turn_id="t2"))
    thread = active_thread(manager)

    assert "断电重启" not in active_values(thread, "attempted_action")
    assert any(
        fact.kind == "attempted_action" and fact.value == "断电重启" and fact.status == "rejected"
        for fact in thread.facts.values()
    )


def test_v4_specific_read_does_not_include_other_product() -> None:
    manager = MemoryManagerV4(store=InMemorySessionIssueMemoryStore(), enabled=True)
    manager.observe_and_write(packet("我的 AC900 显示 E2", turn_id="t1"))
    manager.observe_and_write(packet("另一个 DW200 显示 F1", turn_id="t2"))

    result = manager.read(IssueReadPlanner().plan(session_id="sid", query="AC900 这个故障下一步怎么办？"))

    assert "AC900" in result.rendered_context
    assert "E2" in result.rendered_context
    assert "DW200" not in result.rendered_context
    assert "F1" not in result.rendered_context


def test_v4_redis_store_roundtrip() -> None:
    fake = MagicMock()
    backing: dict[str, str] = {}
    fake.get.side_effect = lambda key: backing.get(key)

    def set_side_effect(key: str, value: str, **kwargs):
        backing[key] = value
        backing[f"{key}:ttl"] = str(kwargs.get("ex"))
        return True

    fake.set.side_effect = set_side_effect
    store = RedisSessionIssueMemoryStore(fake, key_prefix="kf_ut", ttl_seconds=300)
    manager = MemoryManagerV4(store=store, enabled=True)

    manager.observe_and_write(packet("我的 AC900 显示 E2", session_id="persist"))
    reloaded = RedisSessionIssueMemoryStore(fake, key_prefix="kf_ut", ttl_seconds=300).load("persist")

    assert reloaded.active_thread_id is not None
    assert reloaded.threads[reloaded.active_thread_id].product_model == "AC900"
    assert fake.set.call_args.kwargs["ex"] == 300


def test_v4_pipeline_writes_session_issue_memory() -> None:
    reset_memory_manager_v4_for_tests()
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="customer_service",
        reason="客服问题",
        confidence=0.8,
        strategy="test",
    )
    generator = MagicMock()
    generator.generate.return_value = "已记录。"
    vision = MagicMock()
    vision.summarize_images.return_value = ""
    pipeline = ChatPipeline(router=router, generator=generator, vision=vision)

    with (
        patch("app.services.pipeline.settings.memory_enabled", True),
        patch("app.services.pipeline.settings.memory_version", "v4"),
        patch("app.services.memory.v4.manager.settings.memory_enabled", True),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="PROMPT"),
    ):
        pipeline.run("我的 AC900 显示 E2，我已经断电重启过了", images=[], session_id="sid-pipe")

    thread = active_thread(get_memory_manager_v4(), "sid-pipe")
    assert thread.product_model == "AC900"
    assert "E2" in active_values(thread, "fault_code")


def test_v4_allows_llm_candidates_only_when_evidence_backed() -> None:
    class FakeDetector:
        def detect(self, packet, memory):
            return StateChange(
                should_write=True,
                change_type="update",
                candidates=[
                    IssueFactCandidate(
                        kind="attempted_action",
                        value="更换主板",
                        source="llm",
                        confidence=0.9,
                        evidence_text="",
                    )
                ],
                reason="fake",
            )

    manager = MemoryManagerV4(
        store=InMemorySessionIssueMemoryStore(),
        detector=FakeDetector(),
        enabled=True,
    )

    trace = manager.observe_and_write(packet("这个还是不行", recent_history="用户: AC900 显示 E2"))
    assert trace.should_write is False
    assert "LLM 候选缺少可追溯证据" in trace.reason
