from __future__ import annotations

import asyncio
import json

from nikon0.agent.context_governance import ContextGovernance
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import Evidence
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.context.evidence import EvidenceContextManager
from nikon0.context.runtime import ContextRuntime
from nikon0.context.tool_observation import ToolObservationManager
from nikon0.context.conversation import ConversationCompactor
from nikon0.context.read_planner import DeterministicContextReadPlanner, LlmContextReadPlanner
from nikon0.context.budgeter import ContextBudgeter
from nikon0.context.llm_compaction import LlmConversationCompactor
from nikon0.context.llm_span_selector import LlmEvidenceSpanSelector
from nikon0.agent.runtime import _build_default_context_governance
from nikon0.llm.prompts import build_general_messages, build_product_support_messages


def _context(
    *,
    message: str = "继续刚才 AC900 的 E2 问题",
    transcript: str = "user: AC900 显示 E2 怎么办？\nassistant: 需要断电并清洁滤网。",
    memory: str = "[Memory View]\nactive_product:\n- display_name: AC900 空气净化器",
) -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="context-pack-s1", user_id="u1", message=message),
        transcript_context=transcript,
        memory_context=memory,
        trace=ExecutionTrace(trace_id="trace-context-pack", session_id="context-pack-s1", user_message=message),
    )


def test_context_runtime_builds_named_sections_and_budget_report() -> None:
    runtime = ContextRuntime(
        total_char_budget=360,
        section_budgets={
            "conversation": 80,
            "memory": 120,
            "tool_observations": 80,
            "current_user": 80,
        },
    )
    context = _context(transcript="历史" * 200)
    context.tool_results.append(
        {
            "service_id": "order",
            "tool_name": "query_order",
            "ok": True,
            "data": {"summary": "订单 O1001 已签收", "raw": "X" * 500},
        }
    )

    pack = runtime.build_pack(context)

    assert pack.section("current_user").content == "继续刚才 AC900 的 E2 问题"
    assert "AC900 空气净化器" in pack.section("memory").content
    assert "order.query_order" in pack.section("tool_observations").content
    assert pack.budget_report.total_budget == 360
    assert pack.budget_report.used_chars <= 360
    assert pack.budget_report.truncated_sections


def test_context_governance_attaches_context_pack_and_trace_event() -> None:
    context = _context()
    governed = ContextGovernance(context_runtime=ContextRuntime(total_char_budget=1000)).govern(context)

    assert governed.context_pack is not None
    assert "[Context Pack]" in governed.governed_context
    assert "current_user" in governed.governed_context
    event = governed.trace.context_events[-1]
    assert event["strategy"] == "context_pack_v1"
    assert event["section_count"] >= 4
    assert event["budget_report"]["total_budget"] == 1000


def test_llm_prompts_use_context_pack_payload() -> None:
    context = ContextGovernance(context_runtime=ContextRuntime(total_char_budget=1000)).govern(_context())
    evidence = [
        Evidence(
            evidence_id="ev1",
            source="manual",
            text="AC900 显示 E2 表示滤网堵塞。处理前请关闭电源。",
            payload={"manual_name": "AC900手册", "chunk_id": "chunk-e2"},
        )
    ]

    product_payload = json.loads(
        build_product_support_messages(context=context, evidence=evidence, answer_hints=[])[1]["content"]
    )
    general_payload = json.loads(build_general_messages(context=context)[1]["content"])

    assert "context_pack" in product_payload
    assert "context_pack" in general_payload
    assert product_payload["context_pack"]["sections"]["memory"]
    assert general_payload["context_pack"]["sections"]["current_user"] == "继续刚才 AC900 的 E2 问题"
    assert "conversation_context" not in product_payload
    assert "memory_context" not in general_payload


def test_evidence_context_manager_keeps_raw_excerpts_and_deduplicates() -> None:
    manager = EvidenceContextManager(max_items=3, excerpt_char_budget=80)
    evidence = [
        Evidence(
            evidence_id="ev1",
            source="manual",
            text="AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源，取出滤网并清洁。不要在通电时拆机。" * 4,
            confidence=0.91,
            payload={"manual_name": "AC900手册", "page": 12, "chunk_id": "chunk-e2"},
        ),
        Evidence(
            evidence_id="ev2",
            source="manual",
            text="AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源，取出滤网并清洁。不要在通电时拆机。" * 4,
            confidence=0.88,
            payload={"manual_name": "AC900手册", "page": 12, "chunk_id": "chunk-e2-dup"},
        ),
        Evidence(
            evidence_id="ev3",
            source="manual",
            text="AC900 睡眠模式会降低风量，并关闭部分指示灯。",
            confidence=0.7,
            payload={"manual_name": "AC900手册", "page": 18, "chunk_id": "chunk-sleep"},
        ),
    ]

    pack = manager.build(query="AC900 E2 怎么处理？", evidence=evidence)

    assert len(pack.items) == 2
    assert pack.items[0].evidence_id == "ev1"
    assert pack.items[0].raw_excerpt
    assert "滤网堵塞" in pack.items[0].raw_excerpt
    assert len(pack.items[0].raw_excerpt) <= 80
    assert pack.items[0].source["manual_name"] == "AC900手册"
    assert pack.usage["retrieved_evidence_ids"] == ["ev1", "ev2", "ev3"]
    assert pack.usage["included_evidence_ids"] == ["ev1", "ev3"]
    assert pack.usage["deduplicated_evidence_ids"] == ["ev2"]


def test_context_runtime_uses_evidence_pack_section() -> None:
    context = _context()
    context.evidence_context = [
        Evidence(
            evidence_id="ev1",
            source="manual",
            text="AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源并清洁滤网。",
            payload={"manual_name": "AC900手册", "page": 12, "chunk_id": "chunk-e2"},
        )
    ]

    pack = ContextRuntime(total_char_budget=1200).build_pack(context)
    evidence_payload = json.loads(pack.section("evidence").content)

    assert evidence_payload["items"][0]["raw_excerpt"].startswith("AC900 显示 E2")
    assert evidence_payload["items"][0]["source"]["chunk_id"] == "chunk-e2"
    assert evidence_payload["usage"]["included_evidence_ids"] == ["ev1"]


def test_tool_observation_manager_drops_raw_payload_and_keeps_refs() -> None:
    manager = ToolObservationManager(max_items=3, summary_char_budget=80)
    observations = manager.build(
        [
            {
                "service_id": "order",
                "tool_name": "query_order",
                "ok": True,
                "data": {
                    "summary": "订单 O1001 已签收，签收时间 2026-06-18。",
                    "order_id": "O1001",
                    "raw_detail": "X" * 1000,
                },
                "raw": {"http_response": "Y" * 1000},
            },
            {
                "service_id": "case-intake",
                "tool_name": "collect_case_intake",
                "ok": False,
                "error_code": "timeout",
                "error_message": "gateway timeout",
                "data": {"raw": "Z" * 1000},
            },
        ],
        trace_id="trace-tool-observation",
    )

    rendered = observations.render_json()

    assert len(observations.items) == 2
    assert observations.items[0].tool == "order.query_order"
    assert observations.items[0].summary.startswith("订单 O1001")
    assert observations.items[0].raw_result_ref == "trace://trace-tool-observation/tool_results/0"
    assert "raw_detail" in observations.items[0].data_keys
    assert observations.items[1].status == "failed"
    assert observations.items[1].error_message == "gateway timeout"
    assert "X" * 100 not in rendered
    assert "Y" * 100 not in rendered
    assert "Z" * 100 not in rendered


def test_context_runtime_uses_tool_observation_pack_section() -> None:
    context = _context()
    context.tool_results.append(
        {
            "service_id": "order",
            "tool_name": "query_order",
            "ok": True,
            "data": {
                "summary": "订单 O1001 已签收。",
                "raw_detail": "X" * 1000,
            },
        }
    )

    pack = ContextRuntime(total_char_budget=1200).build_pack(context)
    observation_payload = json.loads(pack.section("tool_observations").content)

    assert observation_payload["items"][0]["tool"] == "order.query_order"
    assert observation_payload["items"][0]["summary"] == "订单 O1001 已签收。"
    assert observation_payload["items"][0]["raw_result_ref"] == "trace://trace-context-pack/tool_results/0"
    assert "X" * 100 not in pack.section("tool_observations").content


def test_conversation_compactor_keeps_recent_lines_and_summarizes_old_issue_context() -> None:
    transcript = "\n".join(
        [
            "user: AC900 显示 E2 怎么办？",
            "assistant: E2 通常和滤网堵塞有关，请先断电。",
            "user: 我已经清洁滤网了，还是 E2。",
            "assistant: 请检查风道是否有遮挡。",
            "user: 那我想报修。",
            "assistant: 请提供产品型号和联系电话。",
            "user: 型号 AC900，电话 13800138000。",
            "assistant: 已收集到型号和电话，还需要故障现象。",
        ]
    )

    compacted = ConversationCompactor(max_raw_chars=120, recent_line_count=2).compact(
        transcript,
        active_issue_summary="AC900 E2 故障报修",
    )

    assert compacted.compacted is True
    assert "issue_summary: AC900 E2 故障报修" in compacted.render()
    assert "E2 通常和滤网堵塞有关" in compacted.render()
    assert "user: 型号 AC900，电话 13800138000。" in compacted.render()
    assert "assistant: 已收集到型号和电话，还需要故障现象。" in compacted.render()
    assert compacted.raw_recent_lines == [
        "user: 型号 AC900，电话 13800138000。",
        "assistant: 已收集到型号和电话，还需要故障现象。",
    ]


def test_context_runtime_uses_conversation_compactor_when_history_is_long() -> None:
    context = _context(
        transcript="\n".join(
            [
                "user: AC900 显示 E2 怎么办？",
                "assistant: E2 通常和滤网堵塞有关，请先断电。",
                "user: 我已经清洁滤网了，还是 E2。",
                "assistant: 请检查风道是否有遮挡。",
                "user: 型号 AC900，电话 13800138000。",
                "assistant: 已收集到型号和电话，还需要故障现象。",
            ]
        )
    )
    runtime = ContextRuntime(
        total_char_budget=1200,
        conversation_compactor=ConversationCompactor(max_raw_chars=100, recent_line_count=2),
    )

    pack = runtime.build_pack(context)
    conversation = pack.section("conversation").content

    assert "[Conversation Summary]" in conversation
    assert "[Recent Conversation]" in conversation
    assert "user: 型号 AC900，电话 13800138000。" in conversation
    assert any(event.stage == "context.conversation_compact" for event in context.trace.events)


def test_deterministic_context_read_planner_selects_sections_by_intent() -> None:
    planner = DeterministicContextReadPlanner()

    casual = planner.plan(_context(message="你好，今天心情不错"))
    product = planner.plan(_context(message="AC900 显示 E2 怎么处理？"))
    case = planner.plan(_context(message="我要报修，型号 AC900"))

    assert "evidence" not in casual.included_sections
    assert "workflow" not in casual.included_sections
    assert {"current_user", "memory", "conversation", "runtime"} <= set(casual.included_sections)
    assert "evidence" in product.included_sections
    assert "workflow" in case.included_sections
    assert "tool_observations" in case.included_sections


def test_context_runtime_filters_sections_with_read_plan() -> None:
    context = _context(message="你好，随便聊聊")
    context.evidence_context = [
        Evidence(evidence_id="ev1", source="manual", text="不应该进入闲聊上下文")
    ]
    runtime = ContextRuntime(read_planner=DeterministicContextReadPlanner())

    pack = runtime.build_pack(context)
    section_names = {section.name for section in pack.sections}

    assert "evidence" not in section_names
    assert "current_user" in section_names
    assert any(event.stage == "context.read_plan" for event in context.trace.events)


class FakeContextPlannerClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict]] = []

    async def complete(self, messages):
        self.messages.append(messages)
        return self.response


def test_llm_context_read_planner_parses_structured_json_and_uses_prompt() -> None:
    client = FakeContextPlannerClient(
        json.dumps(
            {
                "included_sections": ["system_policy", "current_user", "memory", "evidence", "runtime"],
                "reasons": {"evidence": "商品手册问题需要证据"},
                "confidence": 0.88,
            },
            ensure_ascii=False,
        )
    )
    planner = LlmContextReadPlanner(client)

    plan = asyncio.run(planner.aplan(_context(message="AC900 显示 E2 怎么处理？")))

    assert plan.source == "llm"
    assert "evidence" in plan.included_sections
    assert plan.reasons["evidence"] == "商品手册问题需要证据"
    assert client.messages
    assert "只输出 JSON" in client.messages[0][0]["content"]
    assert "可选 section" in client.messages[0][1]["content"]


def test_llm_context_read_planner_falls_back_to_deterministic_on_bad_output() -> None:
    planner = LlmContextReadPlanner(FakeContextPlannerClient("not-json"))

    plan = asyncio.run(planner.aplan(_context(message="我要退款，订单 O1001")))

    assert plan.source == "deterministic"
    assert "workflow" in plan.included_sections
    assert "llm_failed" in plan.reasons


def test_context_budgeter_reports_priority_degradation_and_protects_core_sections() -> None:
    context = _context(
        message="AC900 显示 E2 怎么办？",
        transcript="历史对话" * 200,
        memory="[Memory View]\nactive_product: AC900\n" + ("重要记忆" * 80),
    )
    context.tool_results.append(
        {"service_id": "order", "tool_name": "query_order", "ok": True, "data": {"summary": "订单摘要" * 100}}
    )
    context.evidence_context = [
        Evidence(evidence_id="ev1", source="manual", text="AC900 显示 E2 表示滤网堵塞。" * 80)
    ]
    budgeter = ContextBudgeter(
        total_char_budget=520,
        section_budgets={"current_user": 80, "system_policy": 120, "evidence": 160},
    )
    runtime = ContextRuntime(
        read_planner=DeterministicContextReadPlanner(),
        budgeter=budgeter,
    )

    pack = runtime.build_pack(context)
    report = pack.budget_report

    assert report.used_chars <= 520
    assert "current_user" in {section.name for section in pack.sections}
    assert "system_policy" in {section.name for section in pack.sections}
    assert report.section_priorities["current_user"] < report.section_priorities["conversation"]
    assert report.degradation_order[0] in {"runtime", "conversation", "tool_observations"}
    assert report.degraded_sections
    assert "current_user" not in report.dropped_sections


class FakeChatClientForContext:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict]] = []

    async def complete(self, messages):
        self.messages.append(messages)
        return self.response


def test_llm_conversation_compactor_uses_structured_prompt_and_fallback() -> None:
    client = FakeChatClientForContext(
        json.dumps(
            {
                "issue_summary": "AC900 E2 故障报修",
                "summary_lines": [
                    "用户反馈 AC900 显示 E2。",
                    "助手建议断电并清洁滤网。",
                ],
            },
            ensure_ascii=False,
        )
    )
    compactor = LlmConversationCompactor(client, deterministic=ConversationCompactor(max_raw_chars=80, recent_line_count=2))
    transcript = "\n".join(
        [
            "user: AC900 显示 E2 怎么办？",
            "assistant: E2 通常和滤网堵塞有关，请先断电。",
            "user: 型号 AC900，电话 13800138000。",
            "assistant: 已收集到型号和电话，还需要故障现象。",
        ]
    )

    compacted = asyncio.run(compactor.acompact(transcript, active_issue_summary="AC900"))

    assert compacted.compacted is True
    assert compacted.issue_summary == "AC900 E2 故障报修"
    assert "断电并清洁滤网" in compacted.render()
    assert "user: 型号 AC900，电话 13800138000。" in compacted.render()
    assert "只输出 JSON" in client.messages[0][0]["content"]


def test_llm_evidence_span_selector_returns_raw_span_not_summary() -> None:
    text = "前文不相关。" * 20 + "AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源并清洁滤网。" + "后文不相关。" * 20
    client = FakeChatClientForContext(
        json.dumps(
            {
                "start": text.index("AC900"),
                "end": text.index("滤网。") + len("滤网。"),
                "reason": "相关原文片段",
            },
            ensure_ascii=False,
        )
    )
    selector = LlmEvidenceSpanSelector(client, max_span_chars=80)

    span = asyncio.run(selector.select_span(query="AC900 E2 怎么处理？", text=text))

    assert span.text == "AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源并清洁滤网。"
    assert span.source == "llm"
    assert span.summary == ""
    assert "只返回原文 span" in client.messages[0][0]["content"]


def test_llm_evidence_span_selector_falls_back_to_raw_excerpt_on_bad_output() -> None:
    selector = LlmEvidenceSpanSelector(FakeChatClientForContext("bad-json"), max_span_chars=30)

    span = asyncio.run(selector.select_span(query="E2", text="AC900 显示 E2 表示滤网堵塞。处理步骤：关闭电源。"))

    assert span.source == "deterministic"
    assert "E2" in span.text


def test_default_context_governance_uses_llm_components_when_enabled(monkeypatch) -> None:
    class Settings:
        nikon0_context_llm_enabled = True
        nikon0_context_llm_model = "deepseek-v4-flash"
        simple_llm_model = "deepseek-v4-flash"
        gen_model = "deepseek-v4-flash"
        nikon0_context_llm_timeout = 9
        nikon0_context_llm_max_tokens = 321
        nikon0_context_total_char_budget = 4321

    governance = _build_default_context_governance(settings=Settings())
    runtime = governance.context_runtime

    assert isinstance(runtime.read_planner, LlmContextReadPlanner)
    assert isinstance(runtime.conversation_compactor, LlmConversationCompactor)
    assert isinstance(runtime.evidence_manager.span_selector, LlmEvidenceSpanSelector)
    assert runtime.budgeter.total_char_budget == 4321


def test_default_context_governance_can_disable_llm(monkeypatch) -> None:
    class Settings:
        nikon0_context_llm_enabled = False
        nikon0_context_total_char_budget = 1000

    governance = _build_default_context_governance(settings=Settings())

    assert isinstance(governance.context_runtime.read_planner, DeterministicContextReadPlanner)
