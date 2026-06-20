from __future__ import annotations

from nikon0.app.schemas.capability import StateUpdate
from nikon0.app.schemas.capability import FallbackPolicy, SkillManifest, SkillMatch, SkillResult
from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.app.schemas.memory import IssueFact, IssueThread, SessionIssueMemory
from nikon0.memory.governance.lifecycle import IssueThreadLifecycleManager
from nikon0.memory.governance.types import StateUpdateCandidate
from nikon0.memory.governance.write_gate import MemoryWriteGate
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.skills.base import SkillRegistry


def test_write_gate_rejects_invalid_phone() -> None:
    decision = MemoryWriteGate().validate_one(
        SessionIssueMemory(session_id="s1"),
        StateUpdateCandidate(update=StateUpdate(key="case_intake", value={"ticket_payload": {"phone": "123"}})),
    )
    assert decision.outcome == "reject"
    assert "phone" in decision.reason


def test_write_gate_requires_confirmation_for_conflicting_order() -> None:
    thread = IssueThread(
        thread_id="t1",
        facts={"case_intake.order_id": IssueFact(kind="case_intake.order_id", value="ORD-100", source="user", confidence=0.95)},
    )
    memory = SessionIssueMemory(session_id="s1", active_thread_id="t1", threads={"t1": thread})
    decision = MemoryWriteGate().validate_one(
        memory,
        StateUpdateCandidate(
            update=StateUpdate(key="case_intake", value={"ticket_payload": {"order_id": "ORD-200"}}),
            provenance="verified_tool",
            confidence=0.62,
        ),
    )
    assert decision.outcome == "needs_confirmation"
    assert decision.conflicts[0].field == "case_intake.order_id"


def test_write_gate_does_not_compare_new_thread_product_to_old_thread() -> None:
    thread = IssueThread(
        thread_id="t1",
        facts={"product_support.selected_product_id": IssueFact(kind="product_support.selected_product_id", value="air_conditioner", source="user", confidence=0.95)},
    )
    memory = SessionIssueMemory(session_id="s1", active_thread_id="t1", threads={"t1": thread})
    decision = MemoryWriteGate().validate_one(
        memory,
        StateUpdateCandidate(
            update=StateUpdate(key="product_support", value={"selected_product_id": "airfryer"}),
            create_thread=True,
        ),
    )
    assert decision.outcome == "accept"


def test_write_gate_allows_product_support_last_query_to_change() -> None:
    """Conversation state is mutable and must not be treated as product identity."""
    thread = IssueThread(
        thread_id="t1",
        facts={
            "product_support.last_query": IssueFact(
                kind="product_support.last_query",
                value="空气炸锅不加热",
                source="skill",
                confidence=0.82,
            ),
        },
    )
    memory = SessionIssueMemory(session_id="s1", active_thread_id="t1", threads={"t1": thread})

    decision = MemoryWriteGate().validate_one(
        memory,
        StateUpdateCandidate(
            update=StateUpdate(key="product_support", value={"last_query": "我已经拔掉电源重试了"}),
        ),
    )

    assert decision.outcome == "accept"


def test_lifecycle_creates_new_thread_for_different_product_and_switches_back() -> None:
    store = InMemorySessionIssueStore()
    first = store.apply_updates(
        "s1",
        [StateUpdate(key="product_support", value={"selected_product_id": "air_conditioner", "selected_display_name": "空调"})],
        create_thread=True,
    )
    manager = IssueThreadLifecycleManager()
    new_decision = manager.decide(first, "空气炸锅不加热了")
    assert new_decision.action == "create_thread"
    second = store.apply_updates(
        "s1",
        [StateUpdate(key="product_support", value={"selected_product_id": "airfryer", "selected_display_name": "空气炸锅"})],
        create_thread=True,
    )
    switch = manager.decide(second, "刚才的空调问题继续")
    assert switch.action == "switch_open_thread"
    assert switch.thread_id != second.active_thread_id


class _FailingPersistenceStore(InMemorySessionIssueStore):
    def apply_updates(self, *args, **kwargs):
        raise RuntimeError("database unavailable")


class _WriteSkill:
    name = "memory_test_write"
    description = "test memory write"
    risk_level = "low"
    manifest = SkillManifest(
        name=name,
        title="Memory Test Write",
        description=description,
        fallback_policy=FallbackPolicy(),
    )

    def __init__(self, risk_level: str) -> None:
        self.risk_level = risk_level

    async def can_handle(self, context):
        return SkillMatch(matched=True, confidence=1.0, reason="test")

    async def run(self, context):
        return SkillResult(
            status="success",
            answer_draft="done",
            risk_level=self.risk_level,
            state_updates=[StateUpdate(key="product_support", value={"selected_product_id": "airfryer"})],
        )


def test_low_risk_persistence_failure_degrades_but_high_risk_blocks() -> None:
    low = AgentRuntime(
        memory_store=_FailingPersistenceStore(),
        skill_registry=SkillRegistry([_WriteSkill("low")]),
    )
    low_response = __import__("asyncio").run(low.run(AgentRequest(session_id="low", message="test")))
    assert low_response.debug["memory_governance"]["degraded"] is True

    high = AgentRuntime(
        memory_store=_FailingPersistenceStore(),
        skill_registry=SkillRegistry([_WriteSkill("high")]),
    )
    high_response = __import__("asyncio").run(high.run(AgentRequest(session_id="high", message="test")))
    assert high_response.answer.startswith("当前服务状态无法可靠保存")
    assert high_response.debug["memory_governance"]["degraded"] is False
