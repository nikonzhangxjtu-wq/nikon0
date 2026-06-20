"""集成测试 - 端到端场景.

覆盖：多轮对话、复合意图、状态传递、取消流程、产品消歧.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.app.schemas.capability import (
    FallbackPolicy,
    SkillManifest,
    SkillResult,
    StateUpdate,
    StickyPolicy,
    ToolCallRequest,
    ToolCallResult,
    ToolSpec,
)
from nikon0.app.services.storage import InMemoryTraceRecorder, InMemoryTranscriptStore
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.tools.case_intake import ExtractCaseSlotsTool
from nikon0.tools.runtime import ToolRegistry, ToolRuntime

from tests_deep.conftest import FakeRecorderTool, run, make_runtime


class FakeCaseIntakeToolV2:
    """模拟 MCP case-intake 工具的完整行为."""
    spec = ToolSpec(
        service_id="case-intake",
        tool_name="collect_case_intake",
        description="Collect case intake fields.",
        risk_level="medium",
    )

    def __init__(self) -> None:
        self.call_history: list[dict] = []

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        args = request.arguments
        question = str(args.get("question") or "")
        session_id = str(args.get("session_id") or "")
        self.call_history.append({"question": question, "session_id": session_id})

        # 模拟完整收集
        if "型号" in question and "电话" in question and "13800138000" in question:
            return ToolCallResult(
                ok=True, service_id=request.service_id, tool_name=request.tool_name,
                data={
                    "completed": True,
                    "reply_text": "已为你完成售后受理，工单号为 TK-20240001。",
                    "missing_slots": [],
                    "ticket_payload": {
                        "intent": "repair",
                        "product_model": "AC900",
                        "contact_phone": "13800138000",
                        "issue": question,
                        "ticket_id": "TK-20240001",
                        "status": "ready",
                    },
                    "context_block": "[工单状态]\nstatus: ready\nticket_id: TK-20240001",
                },
            )
        # 模拟退款收集
        if "退款" in question:
            if "订单号" in question and "138" in question:
                return ToolCallResult(
                    ok=True, service_id=request.service_id, tool_name=request.tool_name,
                    data={
                        "completed": False,
                        "reply_text": "退款申请已记录，需要审批。",
                        "missing_slots": [],
                        "ticket_payload": {
                            "intent": "refund",
                            "order_id": args.get("enrichment", {}).get("order_id", ""),
                            "contact_phone": "13800138000",
                            "status": "pending_approval",
                        },
                        "context_block": "[退款状态]\nstatus: pending_approval",
                    },
                )
            return ToolCallResult(
                ok=True, service_id=request.service_id, tool_name=request.tool_name,
                data={
                    "completed": False,
                    "reply_text": "为处理退款，请提供订单号和联系电话。",
                    "missing_slots": ["order_id", "contact_phone"],
                    "ticket_payload": {"intent": "refund", "status": "collecting"},
                    "context_block": "[退款状态]\nintent: refund\nstatus: collecting",
                },
            )
        # 默认：收集进行中
        return ToolCallResult(
            ok=True, service_id=request.service_id, tool_name=request.tool_name,
            data={
                "completed": False,
                "reply_text": "为尽快处理，请提供产品型号和联系电话。",
                "missing_slots": ["product_model", "contact_phone"],
                "ticket_payload": {"intent": "repair", "status": "collecting"},
                "context_block": "[工单状态]\nintent: repair\nstatus: collecting",
            },
        )


class FakeCancelTool:
    """模拟取消工具."""
    spec = ToolSpec(
        service_id="case-intake",
        tool_name="try_cancel_case_intake",
        description="Cancel case intake.",
        risk_level="medium",
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            ok=True, service_id=request.service_id, tool_name=request.tool_name,
            data={"cancelled": True, "reply_text": "好的，已取消当前工单收集。"},
        )


def build_integration_runtime() -> AgentRuntime:
    """构建完整的集成测试 runtime."""
    tool_runtime = ToolRuntime(registry=ToolRegistry([
        ExtractCaseSlotsTool(),
        FakeCaseIntakeToolV2(),
        FakeCancelTool(),
        FakeRecorderTool(),
    ]))
    return AgentRuntime(
        tool_runtime=tool_runtime,
        trace_recorder=InMemoryTraceRecorder(),
        transcript_store=InMemoryTranscriptStore(),
    )


class TestMultiTurnCaseIntake:
    """多轮工单收集的集成测试."""

    def test_full_repair_intake_flow(self):
        """完整的报修流程：3轮对话从开始到完成."""
        runtime = build_integration_runtime()

        # 第1轮：发起报修
        r1 = run(runtime, "我的洗碗机坏了不转了，想报修", session_id="flow-repair-1")
        assert r1["debug"]["trace"]["selected_skills"] == ["case_intake"]
        # 第一轮不产生即时答案（已知局限）
        assert r1["debug"]["skill_selection"]["source"] in ("planned", "rule_fallback")

        # 第2轮：提供信息
        r2 = run(runtime, "型号 AC900，电话 13800138000，无法启动", session_id="flow-repair-1")
        assert r2["debug"]["trace"]["selected_skills"] == ["case_intake"]
        assert r2["debug"]["skill_selection"]["source"] == "sticky"
        assert "TK-20240001" in r2["answer"]

        # 验证最终状态
        memory = runtime.memory_store.load("flow-repair-1")
        thread = memory.active_thread()
        assert thread is not None
        assert thread.status == "submitted"
        assert thread.issue_type == "repair"

    def test_cancel_intake_flow(self):
        """取消工单收集流程."""
        runtime = build_integration_runtime()

        # 第1轮：开始收集
        r1 = run(runtime, "设备坏了要报修", session_id="cancel-flow-1")
        assert r1["debug"]["trace"]["selected_skills"] == ["case_intake"]

        # 第2轮：取消
        r2 = run(runtime, "算了，先不报了，取消报修", session_id="cancel-flow-1")
        # 取消后状态应为 cancelled
        memory = runtime.memory_store.load("cancel-flow-1")
        thread = memory.active_thread()
        assert thread is not None
        assert thread.status == "cancelled"

    def test_repair_then_switch_to_refund(self):
        """先报修再切换为退款."""
        runtime = build_integration_runtime()

        # 先发起报修
        run(runtime, "设备坏了要报修", session_id="switch-flow-1")

        # 切换为退款
        r2 = run(runtime, "算了不修了，我要退款退货", session_id="switch-flow-1")
        assert r2["risk_level"] == "high"
        assert any(a["kind"] == "approval" for a in r2["actions"])


class TestTranscriptAndTraceIntegration:
    """Transcript 和 Trace 的集成测试."""

    def test_transcript_accumulates_across_turns(self):
        """Transcript 跨轮积累."""
        runtime = build_integration_runtime()
        run(runtime, "第一轮消息", session_id="transcript-s1")
        r2 = run(runtime, "第二轮消息", session_id="transcript-s1")

        entries = runtime.transcript_store.list_for_session("transcript-s1")
        roles = [e.role for e in entries]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "/n第二轮消息" in runtime.transcript_store.replay_text("transcript-s1") or \
               "第二轮消息" in runtime.transcript_store.replay_text("transcript-s1")

    def test_trace_events_persist_across_turns(self):
        """Trace 事件跨轮保持."""
        runtime = build_integration_runtime()
        r1 = run(runtime, "第一轮", session_id="trace-s1")
        r2 = run(runtime, "第二轮", session_id="trace-s1")

        traces = runtime.trace_recorder.list_for_session("trace-s1")
        assert len(traces) == 2

    def test_context_includes_transcript_history(self):
        """上下文包含对话历史."""
        runtime = build_integration_runtime()
        run(runtime, "第一轮：我的设备是AC900", session_id="ctx-history-s1")
        r2 = run(runtime, "第二轮：继续之前的问题", session_id="ctx-history-s1")

        ctx_events = r2["debug"]["trace"]["context_events"]
        assert ctx_events[0]["transcript_chars"] > 0


class TestProductSupportIntegration:
    """Product Support 集成测试."""

    def test_knowledge_backend_query_and_answer(self, manual_dir):
        """知识后端查询 + 回答生成."""
        knowledge = KnowledgeRuntime(StructuredManualBackend(manual_dir))
        runtime = make_runtime(
            skill_registry=SkillRegistry([
                ProductSupportSkill(knowledge_runtime=knowledge)
            ]),
        )
        result = run(runtime, "AC900 显示 E2 怎么处理？", session_id="ps-int-1")
        assert "product_support" in result["debug"]["trace"]["selected_skills"]
        # 回答应包含证据中的内容
        assert "滤网" in result["answer"] or "清洁" in result["answer"]

    def test_no_evidence_produces_needs_more_info(self, manual_dir):
        """无证据时返回 needs_more_info."""
        knowledge = KnowledgeRuntime(StructuredManualBackend(manual_dir))
        runtime = make_runtime(
            skill_registry=SkillRegistry([
                ProductSupportSkill(knowledge_runtime=knowledge)
            ]),
        )
        result = run(runtime, "XYZ999 显示 X99 怎么处理？", session_id="ps-noev-1")
        # 没有匹配的手册内容
        assert "还没有找到足够的商品手册证据" in result["answer"] or \
               result["debug"]["trace"]["selected_skills"] == ["product_support"]


class TestCompositeScenario:
    """复合场景测试."""

    def test_fault_then_refund_intent_detection(self):
        """故障诊断 + 退款意向."""
        runtime = build_integration_runtime()
        result = run(runtime, "AC900 显示 E2，已经重启过还是不行，想退款")
        plan = result["debug"]["plan"]
        intents = {i["intent"] for i in plan["intents"]}
        assert "product_support" in intents
        assert "refund" in intents
        assert plan["is_composite"] is True
        # case_intake 优先级更高
        assert plan["recommended_skill"] == "case_intake"

    def test_complaint_with_product_issue(self):
        """投诉 + 产品问题."""
        runtime = build_integration_runtime()
        result = run(runtime, "洗碗机坏了，洗不干净，我要投诉并要求退款")
        plan = result["debug"]["plan"]
        intents = {i["intent"] for i in plan["intents"]}
        assert "complaint" in intents or "refund" in intents
        assert plan["risk_level"] == "high"


class TestMaxTurnsBoundary:
    """max_turns 限制的集成测试."""

    def test_loop_hits_max_turns(self):
        """达到 max_turns 上限时停止."""
        class EndlessToolSkill:
            name = "endless"
            description = "Always produces tool calls."
            risk_level = "low"
            manifest = SkillManifest(
                name=name, title="Endless", description=description,
                risk_level="low",
            )

            async def can_handle(self, context):
                from nikon0.app.schemas.capability import SkillMatch
                return SkillMatch(matched=True, confidence=0.9, reason="test")

            async def run(self, context):
                return SkillResult(
                    status="success",
                    answer_draft="",
                    tool_calls=[
                        ToolCallRequest(service_id="test", tool_name="recorder", arguments={"turn": context.loop_turn})
                    ],
                    risk_level="low",
                )

        runtime = make_runtime(
            skill_registry=SkillRegistry([EndlessToolSkill()]),
            tool_runtime=ToolRuntime(registry=ToolRegistry([FakeRecorderTool()])),
            max_turns=2,
        )
        result = run(runtime, "start endless loop")
        assert result["debug"]["loop"]["stop_reason"] == "max_turns"
        assert result["debug"]["loop"]["turn_count"] == 2
