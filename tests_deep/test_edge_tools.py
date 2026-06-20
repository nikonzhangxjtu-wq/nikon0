"""边界测试 - ToolRuntime 模块.

覆盖：工具未注册、权限拒绝、参数校验缺失、hook 链、极端参数.
"""
from __future__ import annotations

import asyncio

import pytest

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import (
    PermissionDecision,
    ToolCallRequest,
    ToolCallResult,
    ToolSpec,
)
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.tools.runtime import (
    EchoTool,
    HookRunner,
    ToolPermissionPolicy,
    ToolRegistry,
    ToolRuntime,
)

from tests_deep.conftest import FakeFailingTool, FakeLargeDataTool, FakeRecorderTool


class TestToolRuntimeEdgeCases:
    """ToolRuntime 边界和异常场景."""

    def make_ctx(self, message: str = "test") -> AgentContext:
        trace = ExecutionTrace(trace_id="tt1", session_id="ts1", user_message=message)
        return AgentContext(
            request=AgentRequest(session_id="ts1", message=message),
            trace=trace,
        )

    def test_call_nonexistent_tool_returns_error(self):
        """调用不存在的工具."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([]))
            ctx = self.make_ctx()
            return await runtime.call(ctx, ToolCallRequest(
                service_id="nonexistent", tool_name="ghost",
            ))
        result = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "tool_not_found"

    def test_high_risk_tool_blocked_by_permission_policy(self):
        """高风险工具被权限策略阻止."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([EchoTool()]))
            ctx = self.make_ctx()
            result = await runtime.call(ctx, ToolCallRequest(
                service_id="mock", tool_name="echo",
                risk_level="high",
            ))
            return result, ctx
        result, ctx = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "permission_denied"
        assert any(
            e.stage == "tool.permission_denied" for e in ctx.trace.events
        )

    def test_approval_required_tool_blocked(self):
        """需要审批的工具被阻止."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([EchoTool()]))
            ctx = self.make_ctx()
            return await runtime.call(ctx, ToolCallRequest(
                service_id="mock", tool_name="echo",
                requires_approval=True,
            ))
        result = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "approval_required"
        assert "approval_request" in result.data

    def test_tool_exception_is_caught_and_normalized(self):
        """工具异常被捕获并标准化."""
        class ExplodingTool:
            spec = ToolSpec(service_id="test", tool_name="boom", risk_level="low")
            async def call(self, request):
                raise ValueError("unexpected value error")

        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([ExplodingTool()]))
            ctx = self.make_ctx()
            return await runtime.call(ctx, ToolCallRequest(
                service_id="test", tool_name="boom",
            ))
        result = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "ValueError"
        assert "unexpected value error" in str(result.error_message)

    def test_hook_chain_execution_order(self):
        """Hook 链执行顺序."""
        order = []

        def pre1(ctx, req):
            order.append("pre1")
            return PermissionDecision(allowed=True, reason="ok")

        def pre2(ctx, req):
            order.append("pre2")
            return PermissionDecision(allowed=True, reason="ok")

        def post1(ctx, req, res):
            order.append("post1")
            return "ok"

        async def _run():
            runtime = ToolRuntime(
                registry=ToolRegistry([EchoTool()]),
                hook_runner=HookRunner(pre_tool=(pre1, pre2), post_tool=(post1,)),
            )
            ctx = self.make_ctx()
            await runtime.call(ctx, ToolCallRequest(service_id="mock", tool_name="echo"))
            return order
        result = asyncio.run(_run())
        assert result[:2] == ["pre1", "pre2"]
        assert "post1" in result[2:]

    def test_pre_hook_rejection_stops_chain(self):
        """pre_hook 拒绝后不再执行后续 hooks."""
        order = []

        def pre_reject(ctx, req):
            order.append("pre_reject")
            return PermissionDecision(allowed=False, reason="blocked", blocked_action="echo")

        def pre_should_not_run(ctx, req):
            order.append("pre_should_not_run")
            return PermissionDecision(allowed=True, reason="ok")

        async def _run():
            runtime = ToolRuntime(
                registry=ToolRegistry([EchoTool()]),
                hook_runner=HookRunner(pre_tool=(pre_reject, pre_should_not_run)),
            )
            ctx = self.make_ctx()
            result = await runtime.call(ctx, ToolCallRequest(service_id="mock", tool_name="echo"))
            return result, order
        result, order = asyncio.run(_run())
        assert result.ok is False
        assert order == ["pre_reject"]

    def test_failure_hook_triggered(self):
        """工具失败时触发 failure hook."""
        failures = []

        def on_fail(ctx, req, reason):
            failures.append(reason)
            return f"logged: {reason}"

        async def _run():
            runtime = ToolRuntime(
                registry=ToolRegistry([FakeFailingTool(fail_count=1)]),
                hook_runner=HookRunner(on_failure=(on_fail,)),
            )
            ctx = self.make_ctx()
            await runtime.call(ctx, ToolCallRequest(service_id="test", tool_name="failing"))
            return failures
        failures = asyncio.run(_run())
        assert len(failures) == 1
        assert "Simulated failure" in failures[0]

    def test_tool_not_found_triggers_failure_hook(self):
        """工具未找到时触发 failure hook."""
        failures = []

        def on_fail(ctx, req, reason):
            failures.append(reason)
            return f"logged: {reason}"

        async def _run():
            runtime = ToolRuntime(
                registry=ToolRegistry([]),
                hook_runner=HookRunner(on_failure=(on_fail,)),
            )
            ctx = self.make_ctx()
            await runtime.call(ctx, ToolCallRequest(service_id="nonexistent", tool_name="ghost"))
            return failures
        failures = asyncio.run(_run())
        assert len(failures) == 1
        assert "not registered" in failures[0]

    def test_tool_call_recorded_in_trace(self):
        """工具调用记录到 trace."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([EchoTool()]))
            ctx = self.make_ctx()
            await runtime.call(ctx, ToolCallRequest(
                service_id="mock", tool_name="echo", arguments={"x": "y"},
            ))
            return ctx
        ctx = asyncio.run(_run())
        assert len(ctx.trace.tool_calls) == 1
        assert ctx.trace.tool_calls[0]["service_id"] == "mock"
        assert ctx.trace.tool_calls[0]["tool_name"] == "echo"
        assert ctx.trace.tool_calls[0]["ok"] is True

    def test_call_step_appends_to_context_tool_results(self):
        """call_step 追加到 context.tool_results."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([EchoTool()]))
            ctx = self.make_ctx()
            assert len(ctx.tool_results) == 0
            await runtime.call_step(ctx, ToolCallRequest(service_id="mock", tool_name="echo"))
            return ctx
        ctx = asyncio.run(_run())
        assert len(ctx.tool_results) == 1
        assert ctx.tool_results[0]["ok"] is True

    def test_large_data_tool_result(self):
        """大量数据的工具结果."""
        async def _run():
            runtime = ToolRuntime(registry=ToolRegistry([FakeLargeDataTool()]))
            ctx = self.make_ctx()
            return await runtime.call(ctx, ToolCallRequest(
                service_id="test", tool_name="large_data",
                arguments={"size": 50000},
            ))
        result = asyncio.run(_run())
        assert result.ok is True
        assert len(result.data["text"]) == 50000

    def test_registry_filter_by_service_id(self):
        """按 service_id 过滤工具列表."""
        registry = ToolRegistry([EchoTool(), FakeRecorderTool()])
        specs = registry.list(service_id="mock")
        assert len(specs) == 1
        assert specs[0].tool_name == "echo"

    def test_registry_list_all(self):
        """列出所有工具."""
        registry = ToolRegistry([EchoTool(), FakeRecorderTool()])
        specs = registry.list()
        assert len(specs) >= 2


class TestProductToolsEdgeCases:
    """Product 工具的边界测试."""

    def test_search_product_manual_empty_query(self):
        """空查询."""
        from nikon0.tools.product import SearchProductManualTool
        async def _run():
            tool = SearchProductManualTool()
            return await tool.call(ToolCallRequest(
                service_id="product-support",
                tool_name="search_product_manual",
                arguments={"query": ""},
            ))
        result = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "invalid_arguments"

    def test_validate_answer_grounding_empty_answer(self):
        """空回答的 grounding 检查."""
        from nikon0.tools.product import ValidateAnswerGroundingTool
        async def _run():
            tool = ValidateAnswerGroundingTool()
            return await tool.call(ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": "",
                    "evidence": [{"evidence_id": "e1", "source": "manual", "text": "some evidence"}],
                },
            ))
        result = asyncio.run(_run())
        assert result.data["grounded"] is False

    def test_validate_answer_grounding_missing_required_terms(self):
        """缺少必需术语."""
        from nikon0.tools.product import ValidateAnswerGroundingTool
        async def _run():
            tool = ValidateAnswerGroundingTool()
            return await tool.call(ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": "请清洁滤网",
                    "evidence": [{"evidence_id": "e1", "source": "manual", "text": "清洁滤网并检查风道"}],
                    "required_terms": ["风道"],
                },
            ))
        result = asyncio.run(_run())
        assert result.data["grounded"] is False
        assert "风道" in result.data["missing_terms"]

    def test_validate_answer_grounding_has_overlap(self):
        """回答与证据有重合."""
        from nikon0.tools.product import ValidateAnswerGroundingTool
        async def _run():
            tool = ValidateAnswerGroundingTool()
            return await tool.call(ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": "请关闭电源并清洁滤网",
                    "evidence": [{"evidence_id": "e1", "source": "manual", "text": "关闭电源，清洁滤网"}],
                    "required_terms": [],
                },
            ))
        result = asyncio.run(_run())
        assert result.data["grounded"] is True
        assert result.data["token_overlap"] > 0


class TestCaseIntakeToolsEdgeCases:
    """CaseIntake 工具的边界测试."""

    def test_extract_slots_empty_message(self):
        """空消息的槽位提取."""
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        async def _run():
            tool = ExtractCaseSlotsTool()
            return await tool.call(ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": ""},
            ))
        result = asyncio.run(_run())
        assert result.ok is True
        assert result.data["intent"] == "unknown"
        assert result.data["confidence"] <= 0.3

    def test_extract_slots_repair_intent(self):
        """维修意图的槽位提取."""
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        async def _run():
            tool = ExtractCaseSlotsTool()
            return await tool.call(ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": "AC900 坏了无法启动，需要报修，电话 13800138000"},
            ))
        result = asyncio.run(_run())
        assert result.data["intent"] == "repair"
        assert result.data["slots"]["product_model"] == "AC900"
        assert result.data["slots"]["contact_phone"] == "13800138000"

    def test_extract_slots_refund_intent(self):
        """退款意图检测."""
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        async def _run():
            tool = ExtractCaseSlotsTool()
            return await tool.call(ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": "我要退款退货，订单号 ORD-12345"},
            ))
        result = asyncio.run(_run())
        assert result.data["intent"] == "refund"
        assert result.data["slots"]["order_id"] == "ORD-12345"

    def test_extract_slots_complaint_intent(self):
        """投诉意图检测."""
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        async def _run():
            tool = ExtractCaseSlotsTool()
            return await tool.call(ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": "我要投诉你们的产品质量，找主管"},
            ))
        result = asyncio.run(_run())
        assert result.data["intent"] == "complaint"

    def test_extract_slots_partial_info(self):
        """部分信息缺失."""
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        async def _run():
            tool = ExtractCaseSlotsTool()
            return await tool.call(ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": "设备坏了"},
            ))
        result = asyncio.run(_run())
        assert "product_model" in result.data["missing_slots"]
        assert "contact_phone" in result.data["missing_slots"]


class TestMemoryToolsEdgeCases:
    """Memory 工具的边界测试."""

    def test_write_session_fact_empty_key(self):
        """空 key 写入."""
        from nikon0.tools.memory import WriteSessionFactTool
        async def _run():
            tool = WriteSessionFactTool()
            return await tool.call(ToolCallRequest(
                service_id="memory",
                tool_name="write_session_fact",
                arguments={"key": "", "value": "test"},
            ))
        result = asyncio.run(_run())
        assert result.ok is False
        assert result.error_code == "invalid_arguments"

    def test_read_session_memory_empty_state(self):
        """空 session state 读取."""
        from nikon0.tools.memory import ReadSessionMemoryTool
        async def _run():
            tool = ReadSessionMemoryTool()
            return await tool.call(ToolCallRequest(
                service_id="memory",
                tool_name="read_session_memory",
                arguments={"session_state": {}},
            ))
        result = asyncio.run(_run())
        assert result.ok is True
        assert result.data["session_state"] == {}

    def test_write_session_fact_with_evidence_ids(self):
        """带 evidence_ids 的写入."""
        from nikon0.tools.memory import WriteSessionFactTool
        async def _run():
            tool = WriteSessionFactTool()
            return await tool.call(ToolCallRequest(
                service_id="memory",
                tool_name="write_session_fact",
                arguments={
                    "key": "product_model",
                    "value": "AC900",
                    "reason": "user confirmed",
                    "evidence_ids": ["ev1", "ev2"],
                },
            ))
        result = asyncio.run(_run())
        assert result.ok is True
        assert result.data["state_update"]["evidence_ids"] == ["ev1", "ev2"]
