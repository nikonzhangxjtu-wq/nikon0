"""边界测试 - Safety 模块.

覆盖：handoff vs approval 优先级、否定语境误判、空上下文、极端组合.
"""
from __future__ import annotations

import asyncio

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import AgentResult
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.safety.gate import SafetyGate


class TestSafetyGateEdgeCases:
    """SafetyGate 的边界和异常场景."""

    def make_ctx(self, message: str, **trace_overrides) -> AgentContext:
        trace = ExecutionTrace(trace_id="t1", session_id="s1", user_message=message)
        for key, value in trace_overrides.items():
            setattr(trace, key, value)
        return AgentContext(
            request=AgentRequest(session_id="s1", message=message),
            trace=trace,
        )

    def test_normal_low_risk_message_is_allowed(self):
        """普通低风险消息应被放行."""
        gate = SafetyGate()
        ctx = self.make_ctx("怎么清洁滤网？")
        result = AgentResult(status="success", answer_draft="请用清水清洁。", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is True
        assert decision.risk_level == "low"

    def test_refund_in_message_triggers_approval(self):
        """消息中含'退款'触发审批."""
        gate = SafetyGate()
        ctx = self.make_ctx("我要退款")
        result = AgentResult(status="success", answer_draft="好的", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.approval_request is not None
        assert decision.approval_request.approval_type == "answer"

    def test_handoff_in_message_triggers_handoff(self):
        """消息中含'转人工'触发转人工."""
        gate = SafetyGate()
        ctx = self.make_ctx("我要投诉升级转人工")
        result = AgentResult(status="success", answer_draft="好的", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.handoff_request is not None

    def test_result_handoff_required_triggers_handoff(self):
        """result status=handoff_required 触发转人工."""
        gate = SafetyGate()
        ctx = self.make_ctx("随便问问")
        result = AgentResult(status="handoff_required", answer_draft="", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.handoff_request is not None

    def test_result_high_risk_triggers_approval(self):
        """result.risk_level=high 触发审批."""
        gate = SafetyGate()
        ctx = self.make_ctx("普通问题")
        result = AgentResult(status="success", answer_draft="答案", risk_level="high")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.approval_request is not None

    def test_handoff_priority_over_approval(self):
        """handoff 优先级高于 approval."""
        gate = SafetyGate()
        ctx = self.make_ctx("我要投诉升级，也要退款")
        result = AgentResult(status="success", answer_draft="好的", risk_level="high")
        decision = asyncio.run(gate.check(ctx, result))
        # handoff 条件先于 approval 条件，应触发 handoff
        assert decision.handoff_request is not None

    def test_negative_refund_not_handled(self):
        """否定语境中的'退款'仍被误判（已知局限）."""
        gate = SafetyGate()
        ctx = self.make_ctx("我不需要退款，只是问问")
        result = AgentResult(status="success", answer_draft="好的", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        # 当前实现：'退款'在消息中，触发审批
        assert decision.allowed is False  # 已知局限：不处理否定语义
        assert decision.approval_request is not None

    def test_approval_words_in_question_not_answer(self):
        """'赔偿'在问题中出现但结果低风险，仍触发审批."""
        gate = SafetyGate()
        ctx = self.make_ctx("货物损坏能赔偿吗？")
        result = AgentResult(status="success", answer_draft="根据政策...", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        # '赔偿' 在 approval_words 中
        assert decision.allowed is False

    def test_workflow_requires_approval_overrides_message(self):
        """workflow decision 的 requires_approval 可独立触发审批."""
        gate = SafetyGate()
        ctx = self.make_ctx("普通问题")
        ctx.trace.add_event(
            "workflow.decision",
            "test workflow",
            requires_approval=True,
            handoff_required=False,
            workflow_name="refund_intake",
        )
        result = AgentResult(status="success", answer_draft="ok", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.approval_request is not None

    def test_workflow_handoff_required_overrides_message(self):
        """workflow decision 的 handoff_required 可独立触发转人工."""
        gate = SafetyGate()
        ctx = self.make_ctx("普通问题")
        ctx.trace.add_event(
            "workflow.decision",
            "test workflow",
            requires_approval=False,
            handoff_required=True,
            workflow_name="complaint_escalation",
        )
        result = AgentResult(status="success", answer_draft="ok", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is False
        assert decision.handoff_request is not None

    def test_safety_decision_recorded_in_trace(self):
        """Safety decision 记录到 trace."""
        gate = SafetyGate()
        ctx = self.make_ctx("你好")
        result = AgentResult(status="success", answer_draft="你好", risk_level="low")
        asyncio.run(gate.check(ctx, result))
        assert len(ctx.trace.safety_decisions) == 1
        assert ctx.trace.safety_decisions[0]["allowed"] is True

    def test_blocked_actions_in_safety_decision(self):
        """高风险请求的 blocked_actions."""
        gate = SafetyGate()
        ctx = self.make_ctx("我要退款")
        result = AgentResult(status="success", answer_draft="好的", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert "send_high_risk_answer" in decision.blocked_actions

    def test_no_workflow_and_normal_message_is_allowed(self):
        """无 workflow decision 且正常消息的处理."""
        gate = SafetyGate()
        ctx = self.make_ctx("今天天气怎么样？")
        result = AgentResult(status="success", answer_draft="不知道", risk_level="low")
        decision = asyncio.run(gate.check(ctx, result))
        assert decision.allowed is True
