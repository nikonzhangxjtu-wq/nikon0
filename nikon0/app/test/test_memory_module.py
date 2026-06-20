from __future__ import annotations

import asyncio

from nikon0.agent.context_governance import ContextGovernance
from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import StateUpdate, ToolCallRequest
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.memory.view import MemoryViewBuilder
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.mock_skill import MockSkill
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.tools.case_intake import ExtractCaseSlotsTool
from nikon0.tools.product import ResolveProductTool, SearchProductManualTool, ValidateAnswerGroundingTool
from nikon0.tools.runtime import ToolRegistry, ToolRuntime


class FakeCaseIntakeTool:
    def __init__(self, tool_name: str = "collect_case_intake") -> None:
        from nikon0.app.schemas.capability import ToolSpec

        self.spec = ToolSpec(
            service_id="case-intake",
            tool_name=tool_name,
            description="Fake case intake tool.",
            risk_level="medium",
        )

    async def call(self, request: ToolCallRequest):
        from nikon0.app.schemas.capability import ToolCallResult

        _ = request
        return ToolCallResult(
            ok=True,
            service_id="case-intake",
            tool_name="collect_case_intake",
            data={
                "completed": False,
                "exited": False,
                "reply_text": "为处理退款，请提供订单号和联系电话。",
                "missing_slots": ["order_id", "contact_phone"],
                "ticket_payload": {"intent": "refund", "status": "collecting"},
                "context_block": "[工单收集状态]\nintent: refund\nstatus: collecting",
            },
        )


def test_memory_view_renders_active_product_and_issue_thread() -> None:
    store = InMemorySessionIssueStore()
    store.apply_updates(
        "memory-view-s1",
        [
            StateUpdate(
                key="product_support",
                value={
                    "last_query": "AC900 显示 E2 怎么办？",
                    "selected_product_id": "air_purifier_ac900",
                    "selected_display_name": "AC900 空气净化器",
                    "manual_names": ["AC900手册"],
                    "evidence_count": 2,
                },
                reason="product support answered with knowledge evidence",
            ),
            StateUpdate(
                key="case_intake",
                value={
                    "status": "collecting",
                    "missing_slots": ["contact_phone"],
                    "ticket_payload": {"intent": "repair", "product_model": "AC900"},
                    "workflow_name": "repair_intake",
                    "workflow_intent": "repair",
                    "workflow_status": "collecting",
                    "workflow_missing_slots": ["contact_phone"],
                },
                reason="case intake collecting repair details",
            ),
        ],
        turn_id="turn-memory-view",
    )

    memory = store.load("memory-view-s1")
    view = MemoryViewBuilder().build(memory)
    rendered = view.render()

    assert view.active_product["display_name"] == "AC900 空气净化器"
    assert "AC900 空气净化器" in rendered
    assert "repair_intake" in rendered
    assert "contact_phone" in rendered
    assert "case_intake.status=collecting" in rendered


def test_context_governance_includes_memory_view() -> None:
    context = AgentContext(
        request=AgentRequest(session_id="ctx-memory-s1", message="继续刚才的问题"),
        session_state=None,
        memory_context="[Memory View]\nactive_product: AC900 空气净化器",
        transcript_context="user: AC900 显示 E2 怎么办？",
        trace=ExecutionTrace(trace_id="trace-memory", session_id="ctx-memory-s1", user_message="继续刚才的问题"),
    )

    governed = ContextGovernance().govern(context)

    assert "[Memory View]" in governed.governed_context
    assert "AC900 空气净化器" in governed.governed_context
    assert governed.trace.context_events[-1]["memory_chars"] > 0


def test_runtime_builds_memory_view_before_skill_execution(tmp_path) -> None:
    manual = tmp_path / "AC900手册.txt"
    manual.write_text("AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源，清洁滤网。", encoding="utf-8")
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(tmp_path))
    skill = ProductSupportSkill(knowledge_runtime=knowledge_runtime)
    tool_runtime = ToolRuntime(
        registry=ToolRegistry(
            [
                ResolveProductTool(),
                SearchProductManualTool(knowledge_runtime),
                ValidateAnswerGroundingTool(),
            ]
        )
    )
    memory_store = InMemorySessionIssueStore()
    memory_store.apply_updates(
        "runtime-memory-s1",
        [
            StateUpdate(
                key="product_support",
                value={
                    "selected_product_id": "air_purifier_ac900",
                    "selected_display_name": "AC900 空气净化器",
                    "manual_names": ["AC900手册"],
                    "last_query": "AC900 显示 E2",
                },
                reason="previous product support answer",
            )
        ],
        turn_id="previous-turn",
    )
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([skill, MockSkill()]),
        tool_runtime=tool_runtime,
        memory_store=memory_store,
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="runtime-memory-s1",
                message="E2 继续怎么处理？",
            )
        )
    )

    assert "AC900 空气净化器" in response.debug["trace"]["context_events"][0]["memory_preview"]
    govern_events = [
        event
        for event in response.debug["trace"]["events"]
        if event["stage"] == "context.govern"
    ]
    assert "AC900 空气净化器" in govern_events[0]["payload"]["memory_preview"]


def test_case_intake_workflow_snapshot_is_attached_to_issue_thread() -> None:
    tool_runtime = ToolRuntime(
        registry=ToolRegistry(
            [
                ExtractCaseSlotsTool(),
                FakeCaseIntakeTool(),
                FakeCaseIntakeTool("try_cancel_case_intake"),
            ]
        )
    )
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([CaseIntakeSkill(), MockSkill()]),
        tool_runtime=tool_runtime,
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="workflow-memory-s1",
                message="我要退款，订单号是 O1001",
            )
        )
    )

    memory = runtime.memory_store.load("workflow-memory-s1")
    thread = memory.active_thread()
    assert response.debug["trace"]["selected_skills"] == ["case_intake"]
    assert thread is not None
    assert thread.workflow_snapshot["workflow_name"] == "refund_intake"
    assert thread.workflow_snapshot["requires_approval"] is True
    assert thread.missing_info == ["order_id", "refund_reason", "contact_phone"]


def test_product_support_records_evidence_usage_trace(tmp_path) -> None:
    manual = tmp_path / "AC900手册.txt"
    manual.write_text("AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源，清洁滤网。", encoding="utf-8")
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(tmp_path))
    skill = ProductSupportSkill(knowledge_runtime=knowledge_runtime)
    tool_runtime = ToolRuntime(
        registry=ToolRegistry(
            [
                ResolveProductTool(),
                SearchProductManualTool(knowledge_runtime),
                ValidateAnswerGroundingTool(),
            ]
        )
    )
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([skill, MockSkill()]),
        tool_runtime=tool_runtime,
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="evidence-usage-s1",
                message="AC900 显示 E2 怎么办？",
            )
        )
    )

    events = response.debug["trace"]["events"]
    usage_events = [event for event in events if event["stage"] == "evidence.usage"]
    assert usage_events
    payload = usage_events[-1]["payload"]
    assert payload["retrieved_evidence_ids"]
    assert payload["used_evidence_ids"]
    assert payload["grounding_checked"] is True
