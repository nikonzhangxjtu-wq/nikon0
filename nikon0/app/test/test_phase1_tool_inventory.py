"""Phase 1 tests for business tool inventory."""

from __future__ import annotations

import asyncio

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import ToolCallRequest
from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.tools.product import SearchProductManualTool
from nikon0.tools.runtime import ToolRegistry, ToolRuntime, default_tools


def _context(message: str = "test") -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="tool-inventory-s1", message=message),
        session_state=SessionIssueMemory(session_id="tool-inventory-s1", flat_state={}),
        trace=ExecutionTrace(
            trace_id="trace-tool-inventory",
            session_id="tool-inventory-s1",
            user_message=message,
        ),
    )


def test_default_tools_include_business_atomic_tools() -> None:
    specs = {(tool.spec.service_id, tool.spec.tool_name) for tool in default_tools()}

    assert ("product-support", "resolve_product") in specs
    assert ("product-support", "search_product_manual") in specs
    assert ("product-support", "validate_answer_grounding") in specs
    assert ("case-intake", "extract_case_slots") in specs
    assert ("case-intake", "collect_case_intake") in specs
    assert ("memory", "read_session_memory") in specs
    assert ("memory", "write_session_fact") in specs


def test_resolve_product_tool_runs_through_tool_runtime() -> None:
    runtime = ToolRuntime()
    context = _context("相机 CF 卡怎么格式化？")

    result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="resolve_product",
                arguments={"message": "相机 CF 卡怎么格式化？"},
            ),
        )
    )

    assert result.ok is True
    resolution = result.data["resolution"]
    assert resolution["status"] == "resolved"
    assert resolution["product_id"] == "canon_dslr"
    assert resolution["manual_names"] == ["DSLR_Camera"]
    assert context.trace.tool_calls[-1]["tool_name"] == "resolve_product"


def test_search_product_manual_tool_returns_evidence_from_injected_backend(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "DSLR_Camera.txt").write_text(
        "格式化 CF 卡时，请打开设置菜单并选择格式化，确认后会清除卡内数据。",
        encoding="utf-8",
    )
    tool = SearchProductManualTool(
        KnowledgeRuntime(StructuredManualBackend(manual_dir)),
    )
    runtime = ToolRuntime(registry=ToolRegistry([tool]))
    context = _context("CF 卡怎么格式化？")

    result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="search_product_manual",
                arguments={
                    "query": "CF 卡怎么格式化？",
                    "allowed_manual_names": ["DSLR_Camera"],
                    "max_evidence": 2,
                },
            ),
        )
    )

    assert result.ok is True
    assert result.data["manual_names"] == ["DSLR_Camera"]
    assert result.data["evidence"][0]["payload"]["manual_name"] == "DSLR_Camera"
    assert "格式化" in result.data["answer_hints"][0]
    assert result.data["backend_trace"][0]["backend"] == "structured_manual"


def test_validate_answer_grounding_tool_reports_missing_terms() -> None:
    runtime = ToolRuntime()
    context = _context("validate")
    evidence = [
        {
            "evidence_id": "ev1",
            "source": "manual",
            "text": "清洁滤网前需要断电，并等待设备停止运行。",
            "payload": {"manual_name": "AirCleaner"},
            "confidence": 1.0,
        }
    ]

    ok_result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": "清洁滤网前需要断电，等设备停止后再操作。",
                    "evidence": evidence,
                    "required_terms": ["断电"],
                },
            ),
        )
    )
    bad_result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": "可以直接清洁滤网。",
                    "evidence": evidence,
                    "required_terms": ["断电"],
                },
            ),
        )
    )

    assert ok_result.data["grounded"] is True
    assert bad_result.data["grounded"] is False
    assert bad_result.data["missing_terms"] == ["断电"]


def test_extract_case_slots_tool_returns_intent_slots_and_missing_slots() -> None:
    runtime = ToolRuntime()
    context = _context("AC900 显示 E2，电话 13800138000，想报修")

    result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="case-intake",
                tool_name="extract_case_slots",
                arguments={"message": "AC900 显示 E2，电话 13800138000，想报修"},
            ),
        )
    )

    assert result.ok is True
    assert result.data["intent"] == "repair"
    assert result.data["slots"]["product_model"] == "AC900"
    assert result.data["slots"]["contact_phone"] == "13800138000"
    assert result.data["missing_slots"] == []


def test_memory_tools_return_explicit_state_patch() -> None:
    runtime = ToolRuntime()
    context = _context("memory")

    read_result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="memory",
                tool_name="read_session_memory",
                arguments={"session_state": {"case_intake": {"status": "collecting"}}},
            ),
        )
    )
    write_result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="memory",
                tool_name="write_session_fact",
                arguments={
                    "key": "case_intake",
                    "value": {"status": "ready"},
                    "reason": "case intake completed",
                    "evidence_ids": ["ev1"],
                },
            ),
        )
    )

    assert read_result.data["session_state"]["case_intake"]["status"] == "collecting"
    assert write_result.ok is True
    assert write_result.data["state_update"] == {
        "key": "case_intake",
        "value": {"status": "ready"},
        "reason": "case intake completed",
        "evidence_ids": ["ev1"],
    }


def test_tool_runtime_call_step_records_result_for_skill_internal_tools() -> None:
    runtime = ToolRuntime()
    context = _context("相机 CF 卡怎么格式化？")

    result = asyncio.run(
        runtime.call_step(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="resolve_product",
                arguments={"message": "相机 CF 卡怎么格式化？"},
            ),
        )
    )

    assert result.ok is True
    assert context.tool_results[-1]["tool_name"] == "resolve_product"
    assert context.trace.tool_calls[-1]["tool_name"] == "resolve_product"
