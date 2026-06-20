"""边界测试 - Context Governance 模块.

覆盖：预算超限、空上下文、超大工具结果、证据去重边界、截断行为.
"""
from __future__ import annotations

import json

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import Evidence
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.context.evidence import EvidenceContextManager
from nikon0.context.budgeter import trim_tail as _trim_tail
from nikon0.context.runtime import ContextRuntime
from nikon0.context.tool_observation import ToolObservationManager


class TestContextRuntimeEdgeCases:
    """ContextRuntime 边界测试."""

    def make_ctx(self, **overrides) -> AgentContext:
        defaults = {
            "request": AgentRequest(session_id="ctx-s1", message="测试消息"),
            "trace": ExecutionTrace(trace_id="ctx-t1", session_id="ctx-s1", user_message="测试消息"),
            "memory_context": "memory content",
            "transcript_context": "previous conversation history",
            "tool_results": [],
            "evidence_context": [],
        }
        defaults.update(overrides)
        return AgentContext(**defaults)

    def test_empty_context_still_builds_pack(self):
        """空上下文仍能构建 context pack."""
        ctx = AgentContext(
            request=AgentRequest(session_id="s1", message=""),
            trace=ExecutionTrace(trace_id="t1", session_id="s1", user_message=""),
        )
        cr = ContextRuntime()
        pack = cr.build_pack(ctx)
        assert len(pack.sections) >= 3
        # system_policy, memory, current_user 至少存在
        names = {s.name for s in pack.sections}
        assert "system_policy" in names
        assert "current_user" in names

    def test_total_budget_exceeded_truncates(self):
        """总预算超出时的截断."""
        cr = ContextRuntime(total_char_budget=50)
        ctx = self.make_ctx(
            transcript_context="A" * 5000,
            memory_context="B" * 5000,
        )
        pack = cr.build_pack(ctx)
        assert pack.budget_report.used_chars <= 50 + 200  # 允许 marker 开销

    def test_section_budget_exceeded_truncates(self):
        """单独 section 预算超出."""
        cr = ContextRuntime(
            section_budgets={"memory": 20},
            total_char_budget=5000,
        )
        ctx = self.make_ctx(memory_context="X" * 500)
        pack = cr.build_pack(ctx)
        memory_section = pack.section("memory")
        assert len(memory_section.content) <= 20 + 100  # 允许 marker

    def test_trim_tail_preserves_recent_content(self):
        """_trim_tail 保留尾部内容."""
        content = "first part. " * 10 + "this is the important end."
        trimmed, was_truncated = _trim_tail(content, 100)
        assert was_truncated
        assert "this is the important end" in trimmed

    def test_trim_tail_exact_budget(self):
        """精确等于预算的内容不被截断."""
        content = "A" * 100
        trimmed, truncated = _trim_tail(content, 100)
        assert not truncated
        assert len(trimmed) == 100
        assert "[truncated" not in trimmed

    def test_trim_tail_zero_budget(self):
        """零预算."""
        content = "some content"
        trimmed, truncated = _trim_tail(content, 0)
        assert trimmed == ""
        assert truncated

    def test_trim_tail_marker_consumes_budget(self):
        """截断 marker 消耗预算."""
        content = "X" * 200
        trimmed, truncated = _trim_tail(content, 30)  # 接近 marker 长度
        assert truncated
        assert len(trimmed) <= 30 or "[truncated" in trimmed

    def test_tool_observations_from_empty_results(self):
        """空工具结果."""
        mgr = ToolObservationManager()
        pack = mgr.build([], trace_id="t1")
        assert len(pack.items) == 0

    def test_tool_observations_from_many_results(self):
        """超过 max_items 的工具结果."""
        mgr = ToolObservationManager(max_items=3)
        results = [
            {
                "service_id": f"svc{i}",
                "tool_name": f"tool{i}",
                "ok": True,
                "data": {"result": f"data_{i}"},
            }
            for i in range(10)
        ]
        pack = mgr.build(results, trace_id="t1")
        # 只保留最后 max_items 个
        assert len(pack.items) == 3

    def test_tool_observation_preserves_error_info(self):
        """工具观察保留错误信息."""
        mgr = ToolObservationManager()
        results = [{
            "service_id": "svc1",
            "tool_name": "tool1",
            "ok": False,
            "error_code": "timeout",
            "error_message": "connection timed out after 30s",
            "data": {},
        }]
        pack = mgr.build(results, trace_id="t1")
        assert pack.items[0].status == "failed"
        assert pack.items[0].error_code == "timeout"
        assert "connection timed out" in pack.items[0].error_message


class TestEvidenceContextEdgeCases:
    """证据上下文管理边界测试."""

    def test_empty_evidence_returns_empty_pack(self):
        """空证据列表."""
        mgr = EvidenceContextManager()
        pack = mgr.build(query="test", evidence=[])
        assert len(pack.items) == 0

    def test_deduplicate_identical_evidence(self):
        """相同证据的去重."""
        mgr = EvidenceContextManager()
        e1 = Evidence(
            evidence_id="e1", source="manual", text="E2 表示滤网堵塞",
            payload={"manual_name": "AC900手册"},
        )
        e2 = Evidence(
            evidence_id="e2", source="manual", text="E2 表示滤网堵塞",
            payload={"manual_name": "AC900手册"},
        )
        pack = mgr.build(query="E2", evidence=[e1, e2])
        assert len(pack.items) == 1  # 去重后只剩一个
        assert e2.evidence_id in pack.usage["deduplicated_evidence_ids"]

    def test_different_manual_same_text_not_deduplicated(self):
        """不同手册相同文本不被去重."""
        mgr = EvidenceContextManager()
        e1 = Evidence(
            evidence_id="e1", source="manual", text="清洁滤网",
            payload={"manual_name": "AC900手册"},
        )
        e2 = Evidence(
            evidence_id="e2", source="manual", text="清洁滤网",
            payload={"manual_name": "BC200手册"},
        )
        pack = mgr.build(query="清洁", evidence=[e1, e2])
        assert len(pack.items) == 2

    def test_excerpt_truncation_at_query_match(self):
        """证据摘要截断在查询匹配附近."""
        mgr = EvidenceContextManager(excerpt_char_budget=50)
        evidence = Evidence(
            evidence_id="e1", source="manual",
            text="前文很长的一段内容。" * 20 + "E2故障代码表示滤网堵塞需要清洁。" + "后文也很长。" * 20,
        )
        pack = mgr.build(query="E2故障代码", evidence=[evidence])
        excerpt = pack.items[0].raw_excerpt
        assert "E2故障代码" in excerpt
        assert len(excerpt) <= 55  # 接近 budget

    def test_excerpt_no_query_match_truncates_head(self):
        """无匹配时的首部截断."""
        mgr = EvidenceContextManager(excerpt_char_budget=30)
        evidence = Evidence(
            evidence_id="e1", source="manual",
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5,
        )
        pack = mgr.build(query="没有匹配", evidence=[evidence])
        excerpt = pack.items[0].raw_excerpt
        assert len(excerpt) <= 30

    def test_excerpt_below_budget_not_truncated(self):
        """短于预算的证据不被截断."""
        mgr = EvidenceContextManager(excerpt_char_budget=500)
        evidence = Evidence(
            evidence_id="e1", source="manual", text="短文本",
        )
        pack = mgr.build(query="test", evidence=[evidence])
        assert pack.items[0].raw_excerpt == "短文本"

    def test_evidence_sorted_by_confidence(self):
        """证据按置信度降序排列."""
        mgr = EvidenceContextManager(max_items=3)
        e1 = Evidence(evidence_id="e1", source="manual", text="low", confidence=0.3)
        e2 = Evidence(evidence_id="e2", source="manual", text="high", confidence=0.9)
        e3 = Evidence(evidence_id="e3", source="manual", text="mid", confidence=0.6)
        pack = mgr.build(query="test", evidence=[e1, e2, e3])
        confidences = [item.confidence for item in pack.items]
        assert confidences == [0.9, 0.6, 0.3]

    def test_max_items_limit_respected(self):
        """max_items 限制生效."""
        mgr = EvidenceContextManager(max_items=2)
        evidence = [
            Evidence(evidence_id=f"e{i}", source="manual", text=f"text{i}", confidence=float(i)/10)
            for i in range(1, 6)
        ]
        pack = mgr.build(query="test", evidence=evidence)
        assert len(pack.items) == 2

    def test_source_metadata_mapping(self):
        """源数据映射正确."""
        mgr = EvidenceContextManager()
        evidence = Evidence(
            evidence_id="e1", source="enterprise_rag", text="some text",
            payload={
                "manual_name": "AC900手册",
                "chunk_id": "chunk_001",
                "knowledge_version": "v3",
                "score": 0.85,
            },
        )
        pack = mgr.build(query="test", evidence=[evidence])
        item = pack.items[0]
        assert item.source["manual_name"] == "AC900手册"
        assert item.source["chunk_id"] == "chunk_001"
        assert item.source_type == "enterprise_rag"


class TestContextPackEdgeCases:
    """ContextPack 的边界测试."""

    def test_section_map_returns_dict(self):
        """section_map 返回名字→内容的映射."""
        from nikon0.context.pack import ContextBudgetReport, ContextPack, ContextSection

        pack = ContextPack(
            sections=[
                ContextSection(name="a", content="content_a", priority=10),
                ContextSection(name="b", content="content_b", priority=20),
            ],
            budget_report=ContextBudgetReport(total_budget=1000),
        )
        smap = pack.section_map()
        assert smap == {"a": "content_a", "b": "content_b"}

    def test_missing_section_raises_keyerror(self):
        """请求不存在的 section 抛出 KeyError."""
        import pytest
        from nikon0.context.pack import ContextBudgetReport, ContextPack, ContextSection

        pack = ContextPack(
            sections=[ContextSection(name="a", content="c", priority=10)],
            budget_report=ContextBudgetReport(total_budget=1000),
        )
        with pytest.raises(KeyError):
            pack.section("nonexistent")

    def test_system_policy_constant_includes_key_rules(self):
        """system_policy 包含关键规则."""
        cr = ContextRuntime()
        policy = cr._system_policy()
        assert "nikon0" in policy
        assert "不能编造" in policy or "base" in policy.lower()
