from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from nikon0.agent.runtime import AgentRuntime, _build_default_answer_generator, _build_default_skill_registry
from nikon0.app.main import app
from nikon0.app.schemas.agent import AgentRequest
from nikon0.app.schemas.capability import SkillManifest, ToolCallRequest
from nikon0.app.schemas.knowledge import KnowledgeRequest
from nikon0.app.schemas.safety import ApprovalRequest
from nikon0.app.services.approvals import JsonlApprovalStore
from nikon0.app.services.storage import InMemoryTraceRecorder, InMemoryTranscriptStore
from nikon0.app.services.storage import JsonlTraceRecorder, JsonlTranscriptStore
from nikon0.eval.agent_dataset import build_golden_agent_dataset
from nikon0.eval.agent_metrics import AgentEvaluationHarness
from nikon0.eval.skill_selection import SkillSelectionCase, SkillSelectionHarness
from nikon0.knowledge.runtime import EnterpriseRagBackend, KnowledgeRuntime, StructuredManualBackend
from nikon0.llm.generation import LlmAnswerGenerator
from nikon0.skills.base import ManifestDrivenSkillSelector, SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.mock_skill import MockSkill
from nikon0.skills.model_selector import BailianOllamaSkillSelectionClient, LlmSkillSelector
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.skills.tool_echo import ToolEchoSkill
from nikon0.tools.mcp_gateway import McpGatewayTool
from nikon0.tools.case_intake import ExtractCaseSlotsTool
from nikon0.tools.product import ResolveProductTool, SearchProductManualTool, ValidateAnswerGroundingTool
from nikon0.tools.runtime import HookRunner, ToolRegistry, ToolRuntime


class FakeCaseIntakeTool:
    def __init__(self, tool_name: str = "collect_case_intake") -> None:
        from nikon0.app.schemas.capability import ToolSpec

        self.spec = ToolSpec(
            service_id="case-intake",
            tool_name=tool_name,
            description="Fake case intake tool for tests.",
            risk_level="medium",
        )

    async def call(self, request):
        from nikon0.app.schemas.capability import ToolCallResult

        if request.tool_name == "try_cancel_case_intake":
            return ToolCallResult(
                ok=True,
                service_id=request.service_id,
                tool_name=request.tool_name,
                data={"cancelled": True},
            )
        question = str(request.arguments.get("question") or "")
        if "型号" in question and "电话" in question:
            payload = {
                "completed": True,
                "exited": False,
                "reply_text": "已为你完成售后受理信息收集。",
                "missing_slots": [],
                "ticket_payload": {
                    "intent": "repair",
                    "product_model": "AC900",
                    "issue": question,
                    "contact_phone": "13800138000",
                    "priority": "medium",
                    "status": "ready",
                },
                "context_block": "[工单收集状态]\nstatus: ready",
            }
        elif "退款" in question:
            payload = {
                "completed": False,
                "exited": False,
                "reply_text": "为处理退款，请提供订单号和联系电话。",
                "missing_slots": ["order_id", "contact_phone"],
                "ticket_payload": {"intent": "refund", "status": "collecting"},
                "context_block": "[工单收集状态]\nintent: refund\nstatus: collecting",
            }
        else:
            payload = {
                "completed": False,
                "exited": False,
                "reply_text": "为尽快处理，请提供产品型号和联系电话。",
                "missing_slots": ["product_model", "contact_phone"],
                "ticket_payload": {"intent": "repair", "status": "collecting"},
                "context_block": "[工单收集状态]\nintent: repair\nstatus: collecting",
            }
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data=payload,
        )


class FakeChatClient:
    def __init__(self, response: str = "LLM 回答") -> None:
        self.response = response
        self.messages: list[list[dict]] = []

    async def complete(self, messages):
        self.messages.append(messages)
        return self.response


class FailingChatClient:
    async def complete(self, messages):
        _ = messages
        raise RuntimeError("llm unavailable")


class FakeFlakyCaseIntakeTool(FakeCaseIntakeTool):
    def __init__(self) -> None:
        super().__init__("collect_case_intake")
        self.calls = 0

    async def call(self, request):
        from nikon0.app.schemas.capability import ToolCallResult

        self.calls += 1
        if self.calls == 1:
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code="temporary_failure",
                error_message="temporary gateway failure",
            )
        return await super().call(request)


class FakeToolDependentSkill:
    name = "case_intake"
    description = "Fake tool dependent skill."
    risk_level = "medium"

    from nikon0.app.schemas.capability import SkillManifest

    manifest = SkillManifest(
        name=name,
        title="Fake Case Intake",
        description=description,
        required_tools=["case-intake.collect_case_intake"],
        risk_level="medium",
    )

    async def can_handle(self, context):
        from nikon0.app.schemas.capability import SkillMatch

        _ = context
        return SkillMatch(matched=True, confidence=0.9, reason="fake matched")

    async def run(self, context):
        from nikon0.app.schemas.capability import SkillResult

        _ = context
        return SkillResult(status="success", answer_draft="fake")


class FakeNonMatchingSkill:
    name = "non_matching"
    description = "Fake skill that guard rejects."
    risk_level = "low"
    manifest = SkillManifest(name=name, title="Non Matching", description=description)

    async def can_handle(self, context):
        from nikon0.app.schemas.capability import SkillMatch

        _ = context
        return SkillMatch(matched=False, confidence=0.0, reason="guard rejected fake skill")

    async def run(self, context):
        from nikon0.app.schemas.capability import SkillResult

        _ = context
        return SkillResult(status="success", answer_draft="should not run")


class FakeFailingSkill:
    name = "failing_skill"
    description = "Fake skill that raises."
    risk_level = "low"
    manifest = SkillManifest(name=name, title="Failing Skill", description=description)

    async def can_handle(self, context):
        from nikon0.app.schemas.capability import SkillMatch

        _ = context
        return SkillMatch(matched=True, confidence=0.95, reason="fake failure match")

    async def run(self, context):
        _ = context
        raise RuntimeError("skill exploded")


class FakeImageEvidence:
    image_id = "img-ac900-e2"
    image_type = "diagram"
    match_reason = "OCR/实体精确匹配：E2"
    prompt_text = "OCR: E2\n操作步骤: 清洁滤网"
    score = 0.91
    parent_chunk_ids = ["chunk-ac900-e2"]


class FakeRetrievedChunk:
    chunk_id = "chunk-ac900-e2"
    text = "AC900 显示 E2 表示滤网堵塞或风道异常。处理步骤：关闭电源，清洁滤网。"
    score = 0.88
    manual_name = "AC900手册"
    image_ids = ["img-ac900-e2"]
    image_evidence = [FakeImageEvidence()]


class FakeEnterpriseRetriever:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[dict] = []

    def retrieve(self, query, top_k=4, manual_name=None, image_inputs=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "manual_name": manual_name,
                "image_inputs": image_inputs or [],
            }
        )
        if self.should_fail:
            raise RuntimeError("milvus unavailable")
        return [FakeRetrievedChunk()]

    def build_trace(self, *, query, top_k, raw_chunks, filtered_chunks, source_queries=None, manual_name_decisions=None):
        return type(
            "Trace",
            (),
            {
                "query": query,
                "top_k": top_k,
                "raw_count": len(raw_chunks),
                "filtered_count": len(filtered_chunks),
                "score_threshold": 0.1,
                "top1_score": raw_chunks[0].score if raw_chunks else None,
                "retrieved_chunk_ids": [c.chunk_id for c in raw_chunks],
                "filtered_chunk_ids": [c.chunk_id for c in filtered_chunks],
                "retrieved_manual_names": [c.manual_name for c in raw_chunks],
                "filtered_manual_names": [c.manual_name for c in filtered_chunks],
                "source_queries": source_queries or [query],
                "manual_name_decisions": manual_name_decisions or [],
                "image_vector_hits": [],
                "ocr_entity_hits": ["img-ac900-e2"],
                "selected_image_ids": ["img-ac900-e2"],
            },
        )()


def no_manual_name_decision(_query: str) -> dict:
    return {
        "manual_name": "",
        "confidence": 0.0,
        "source": "test",
        "reason": "disabled in tests",
        "should_filter": False,
    }


class FakeSelectorClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload


def build_case_intake_runtime() -> AgentRuntime:
    tool_runtime = ToolRuntime(
        registry=ToolRegistry([
            ExtractCaseSlotsTool(),
            FakeCaseIntakeTool("collect_case_intake"),
            FakeCaseIntakeTool("try_cancel_case_intake"),
        ])
    )
    return AgentRuntime(tool_runtime=tool_runtime)


def build_eval_runtime(manual_dir) -> AgentRuntime:
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(manual_dir))
    tool_runtime = ToolRuntime(
        registry=ToolRegistry([
            FakeCaseIntakeTool("collect_case_intake"),
            FakeCaseIntakeTool("try_cancel_case_intake"),
            ExtractCaseSlotsTool(),
            ResolveProductTool(),
            SearchProductManualTool(knowledge_runtime),
            ValidateAnswerGroundingTool(),
            ToolRuntime().registry.get("mock", "echo"),
        ])
    )
    return AgentRuntime(
        skill_registry=SkillRegistry([
            ToolEchoSkill(),
            CaseIntakeSkill(),
            ProductSupportSkill(knowledge_runtime=knowledge_runtime),
            MockSkill(),
        ]),
        tool_runtime=tool_runtime,
    )


def test_agent_runtime_runs_phase1_loop() -> None:
    runtime = AgentRuntime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="s1",
                user_id="u1",
                message="你好，介绍一下 nikon0",
            )
        )
    )

    assert response.answer.startswith("nikon0 已接收到你的请求")
    assert response.risk_level == "low"
    assert response.trace_id
    assert [action.kind for action in response.actions] == ["agent"]
    trace = response.debug["trace"]
    assert trace["selected_agents"] == ["supervisor"]
    assert trace["selected_skills"] == []
    assert trace["memory_updates"] == []
    assert response.debug["loop"]["stop_reason"] == "no_tool_calls"
    assert response.debug["trace_persisted"] == response.trace_id
    assert response.debug["transcript_entries"] == 2
    assert response.debug["plan"]["recommended_skill"] is None
    assert response.debug["plan"]["needs_general_handle"] is True


def test_agent_runtime_persists_trace_and_transcript() -> None:
    trace_recorder = InMemoryTraceRecorder()
    transcript_store = InMemoryTranscriptStore()
    runtime = AgentRuntime(trace_recorder=trace_recorder, transcript_store=transcript_store)

    first = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="persist-s1",
                message="你好",
            )
        )
    )
    second = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="persist-s1",
                message="介绍一下 nikon0",
            )
        )
    )

    assert trace_recorder.get(first.trace_id) is not None
    assert trace_recorder.get(second.trace_id) is not None
    assert len(trace_recorder.list_for_session("persist-s1")) == 2
    transcript = transcript_store.list_for_session("persist-s1")
    assert [entry.role for entry in transcript] == ["user", "assistant", "user", "assistant"]
    assert "user: 你好" in transcript_store.replay_text("persist-s1")


def test_jsonl_transcript_store_appends_and_replays(tmp_path) -> None:
    transcript_path = tmp_path / "transcripts.jsonl"
    transcript_store = JsonlTranscriptStore(transcript_path)
    runtime = AgentRuntime(transcript_store=transcript_store)

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="jsonl-s1",
                message="你好",
            )
        )
    )

    assert transcript_path.exists()
    entries = transcript_store.list_for_session("jsonl-s1")
    assert [entry.role for entry in entries] == ["user", "assistant"]
    assert response.debug["transcript_entries"] == 2
    assert "assistant:" in transcript_store.replay_text("jsonl-s1")


def test_jsonl_trace_recorder_persists_and_replays(tmp_path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_recorder = JsonlTraceRecorder(trace_path)
    runtime = AgentRuntime(trace_recorder=trace_recorder)

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="jsonl-trace-s1",
                message="你好",
            )
        )
    )

    assert trace_path.exists()
    stored = trace_recorder.get(response.trace_id)
    assert stored is not None
    assert stored.trace.trace_id == response.trace_id
    assert trace_recorder.list_for_session("jsonl-trace-s1")[0].trace_id == response.trace_id


def test_jsonl_approval_store_persists_status_updates(tmp_path) -> None:
    store_path = tmp_path / "approvals.jsonl"
    store = JsonlApprovalStore(store_path)
    approval = ApprovalRequest(
        approval_id="approval-test-1",
        trace_id="trace-1",
        session_id="approval-s1",
        approval_type="answer",
        title="Approval",
        reason="test",
        risk_level="high",
        requested_action="send_answer",
    )

    store.create_approval(approval)
    updated = store.update_approval("approval-test-1", "approved")
    reloaded = JsonlApprovalStore(store_path)

    assert updated is not None
    assert updated.status == "approved"
    assert reloaded.get_approval("approval-test-1").status == "approved"
    assert len(reloaded.list_approvals("approval-s1")) >= 1


def test_agent_runtime_creates_approval_for_refund_request() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="s2",
                message="我要退款",
            )
        )
    )

    assert response.risk_level == "high"
    assert "审批请求" in response.answer
    approval_actions = [action for action in response.actions if action.kind == "approval"]
    assert approval_actions
    assert approval_actions[0].status == "pending"
    safety_events = response.debug["trace"]["safety_decisions"]
    assert safety_events[0]["approval_request"]["approval_type"] == "answer"
    workflow_events = [
        event for event in response.debug["trace"]["events"]
        if event["stage"] == "workflow.decision"
    ]
    assert workflow_events[-1]["payload"]["workflow_name"] == "refund_intake"
    assert workflow_events[-1]["payload"]["requires_approval"] is True


def test_agent_runtime_creates_handoff_for_escalation_request() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="handoff-s1",
                message="我要投诉升级并转人工",
            )
        )
    )

    assert response.risk_level == "high"
    assert "人工处理" in response.answer
    handoff_actions = [action for action in response.actions if action.kind == "handoff"]
    assert handoff_actions
    assert response.debug["trace"]["safety_decisions"][0]["handoff_request"]["handoff_id"]
    workflow_events = [
        event for event in response.debug["trace"]["events"]
        if event["stage"] == "workflow.decision"
    ]
    assert workflow_events[-1]["payload"]["workflow_name"] == "complaint_escalation"
    assert workflow_events[-1]["payload"]["handoff_required"] is True


def test_case_intake_skill_is_selected_for_repair_request() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="case-s1",
                message="我的设备坏了，想报修",
            )
        )
    )

    assert "请" in response.answer
    assert response.debug["trace"]["selected_skills"] == ["case_intake"]
    assert response.debug["trace"]["memory_updates"][0]["key"] == "case_intake"
    assert response.debug["plan"]["recommended_skill"] == "case_intake"
    assert response.debug["loop"]["turn_count"] == 2
    assert [item["tool_name"] for item in response.debug["trace"]["tool_calls"][:2]] == [
        "extract_case_slots",
        "collect_case_intake",
    ]
    workflow_events = [
        event for event in response.debug["trace"]["events"]
        if event["stage"] == "workflow.decision"
    ]
    assert workflow_events[-1]["payload"]["workflow_name"] == "repair_intake"


def test_planner_exposes_composite_intent_for_fault_and_refund() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="planner-composite-s1",
                message="我的 AC900 显示 E2，已经重启过，想退款",
            )
        )
    )

    plan = response.debug["plan"]
    intents = {item["intent"] for item in plan["intents"]}
    assert {"product_support", "refund"} <= intents
    assert plan["is_composite"] is True
    assert plan["recommended_skill"] == "case_intake"
    assert response.debug["trace"]["selected_skills"] == ["case_intake"]
    planner_events = [
        event for event in response.debug["trace"]["events"]
        if event["stage"] == "planner.plan"
    ]
    assert planner_events
    assert planner_events[0]["payload"]["is_composite"] is True


def test_product_support_skill_answers_with_knowledge_runtime(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。处理步骤：关闭电源，清洁滤网，检查风道后重新启动。",
        encoding="utf-8",
    )
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(manual_dir))
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([ProductSupportSkill(knowledge_runtime=knowledge_runtime)]),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="product-support-s1",
                message="AC900 显示 E2 怎么处理？",
            )
        )
    )

    assert response.debug["plan"]["recommended_skill"] == "product_support"
    assert response.debug["trace"]["selected_skills"] == ["product_support"]
    tool_names = [item["tool_name"] for item in response.debug["trace"]["tool_calls"]]
    assert tool_names == ["resolve_product", "search_product_manual", "validate_answer_grounding"]
    action_names = [action.name for action in response.actions if action.kind == "tool"]
    assert "product-support.search_product_manual" in action_names
    assert "滤网" in response.answer
    assert response.debug["trace"]["knowledge_calls"][0]["evidence_count"] >= 1
    memory = runtime.memory_store.load("product-support-s1")
    thread = memory.active_thread()
    assert thread is not None
    assert "product_support.last_query" in thread.facts


def test_product_support_skill_uses_llm_answer_generator(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。",
        encoding="utf-8",
    )
    client = FakeChatClient("LLM：请先断电，清洁滤网并检查风道。")
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(manual_dir))
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([
            ProductSupportSkill(
                knowledge_runtime=knowledge_runtime,
                answer_generator=LlmAnswerGenerator(client),
            )
        ]),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="product-llm-s1",
                message="AC900 显示 E2 怎么处理？",
            )
        )
    )

    assert response.answer == "LLM：请先断电，清洁滤网并检查风道。"
    tool_names = [item["tool_name"] for item in response.debug["trace"]["tool_calls"]]
    assert tool_names == ["resolve_product", "search_product_manual", "validate_answer_grounding"]
    assert client.messages
    payload = json.loads(client.messages[0][1]["content"])
    assert "context_pack" in payload
    evidence_section = json.loads(payload["context_pack"]["sections"]["evidence"])
    assert evidence_section["items"][0]["raw_excerpt"] == "AC900 显示 E2 表示滤网堵塞或风道异常。"
    assert evidence_section["items"][0]["source"]["manual_name"] == "AC900手册"
    assert any(event["stage"] == "llm.answer" for event in response.debug["trace"]["events"])


def test_product_support_llm_failure_falls_back_to_evidence_template(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞。",
        encoding="utf-8",
    )
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(manual_dir))
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([
            ProductSupportSkill(
                knowledge_runtime=knowledge_runtime,
                answer_generator=LlmAnswerGenerator(FailingChatClient()),
            )
        ]),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="product-llm-fallback-s1",
                message="AC900 显示 E2 怎么处理？",
            )
        )
    )

    assert "根据当前商品手册证据" in response.answer
    tool_names = [item["tool_name"] for item in response.debug["trace"]["tool_calls"]]
    assert tool_names == ["resolve_product", "search_product_manual", "validate_answer_grounding"]
    assert any(event["stage"] == "llm.answer.error" for event in response.debug["trace"]["events"])


def test_product_support_without_evidence_skips_grounding(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    knowledge_runtime = KnowledgeRuntime(StructuredManualBackend(manual_dir))
    runtime = AgentRuntime(
        skill_registry=SkillRegistry([ProductSupportSkill(knowledge_runtime=knowledge_runtime)]),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="product-no-evidence-s1",
                message="AC900 显示 E2 怎么处理？",
            )
        )
    )

    tool_names = [item["tool_name"] for item in response.debug["trace"]["tool_calls"]]
    assert tool_names == ["resolve_product", "search_product_manual"]
    assert "还没有找到足够的商品手册证据" in response.answer


def test_general_handle_uses_llm_answer_generator() -> None:
    client = FakeChatClient("LLM：你好，我可以帮你处理商品咨询、售后和订单问题。")
    runtime = AgentRuntime(answer_generator=LlmAnswerGenerator(client))

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="general-llm-s1",
                message="你好，你能做什么？",
            )
        )
    )

    assert response.answer == "LLM：你好，我可以帮你处理商品咨询、售后和订单问题。"
    assert client.messages
    assert "你好，你能做什么？" in client.messages[0][1]["content"]
    assert any(event["stage"] == "llm.answer" for event in response.debug["trace"]["events"])


def test_llm_skill_selector_parses_structured_model_output() -> None:
    client = FakeSelectorClient(
        '{"selected_skill":"product_support","confidence":0.82,"reason":"manual QA"}'
    )
    selector = LlmSkillSelector(client)
    runtime = AgentRuntime(
        skill_registry=SkillRegistry(
            [ProductSupportSkill(knowledge_runtime=KnowledgeRuntime(StructuredManualBackend("missing")))],
            selector=selector,
        )
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="llm-selector-s1",
                message="AC900 显示 E2 怎么处理？",
            )
        )
    )

    assert client.prompts
    assert response.debug["skill_selection"]["source"] == "model"
    assert response.debug["skill_selection"]["selected_skill"] == "product_support"


def test_bailian_ollama_skill_selection_client_uses_project_llm_client(monkeypatch) -> None:
    calls = []

    def fake_chat_text(**kwargs):
        calls.append(kwargs)
        return '{"selected_skill":null,"confidence":0.2,"reason":"general"}'

    monkeypatch.setattr("app.services.llm_clients.chat_text", fake_chat_text)
    client = BailianOllamaSkillSelectionClient(model="deepseek-v4-flash", timeout=7)

    raw = asyncio.run(client.complete('{"user_message":"你好"}'))

    assert "selected_skill" in raw
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["timeout"] == 7
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][1]["role"] == "user"


def test_default_skill_registry_enables_llm_selector_from_settings(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "router_llm_enabled", True)
    monkeypatch.setattr(settings, "router_llm_model", "deepseek-v4-flash")

    registry = _build_default_skill_registry()

    assert registry.selector is not None


def test_default_answer_generator_uses_project_llm_settings(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "simple_llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "gen_max_tokens", 512)

    generator = _build_default_answer_generator()

    assert generator is not None


def test_default_skill_registry_injects_enterprise_rag_backend() -> None:
    registry = _build_default_skill_registry()
    product_support = registry.get("product_support")

    assert isinstance(product_support, ProductSupportSkill)
    assert isinstance(product_support.knowledge_runtime.backend, EnterpriseRagBackend)


def test_enterprise_rag_backend_maps_milvus_chunks_images_and_trace() -> None:
    retriever = FakeEnterpriseRetriever()
    backend = EnterpriseRagBackend(
        retriever_factory=lambda: retriever,
        manual_name_decider=no_manual_name_decision,
    )
    runtime = KnowledgeRuntime(backend)

    result = asyncio.run(
        runtime.query(
            KnowledgeRequest(
                query="AC900 显示 E2 怎么处理？",
                images=["base64-image"],
                need_images=True,
                max_evidence=3,
            )
        )
    )

    assert retriever.calls[0]["query"] == "AC900 显示 E2 怎么处理？"
    assert retriever.calls[0]["image_inputs"] == ["base64-image"]
    assert result.evidence[0].source == "enterprise_rag"
    assert result.evidence[0].payload["chunk_id"] == "chunk-ac900-e2"
    assert result.evidence[0].payload["manual_name"] == "AC900手册"
    assert result.evidence[0].payload["image_evidence"][0]["image_id"] == "img-ac900-e2"
    assert result.backend_trace[0]["backend"] == "enterprise_rag"
    assert result.backend_trace[0]["retrieval_trace"]["selected_image_ids"] == ["img-ac900-e2"]


def test_enterprise_rag_backend_filters_manuals_by_permission() -> None:
    backend = EnterpriseRagBackend(
        retriever_factory=lambda: FakeEnterpriseRetriever(),
        manual_name_decider=no_manual_name_decision,
    )
    runtime = KnowledgeRuntime(backend)

    result = asyncio.run(
        runtime.query(
            KnowledgeRequest(
                query="AC900 显示 E2 怎么处理？",
                allowed_manual_names=["其他手册"],
                max_evidence=3,
            )
        )
    )

    assert result.evidence == []
    assert result.backend_trace[0]["permission_filter"]["allowed_manual_names"] == ["其他手册"]
    assert result.backend_trace[0]["filtered_count"] == 0


def test_knowledge_runtime_falls_back_when_enterprise_rag_unavailable(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。处理步骤：关闭电源，清洁滤网。",
        encoding="utf-8",
    )
    enterprise = EnterpriseRagBackend(
        retriever_factory=lambda: FakeEnterpriseRetriever(should_fail=True),
        fallback_backend=StructuredManualBackend(manual_dir),
        manual_name_decider=no_manual_name_decision,
    )
    runtime = KnowledgeRuntime(enterprise)

    result = asyncio.run(
        runtime.query(KnowledgeRequest(query="AC900 显示 E2 怎么处理？", max_evidence=3))
    )

    assert result.evidence
    assert result.evidence[0].source == "manual"
    assert result.backend_trace[0]["backend"] == "enterprise_rag"
    assert result.backend_trace[0]["fallback"] == "structured_manual"


def test_case_intake_pending_state_handles_followup() -> None:
    runtime = build_case_intake_runtime()

    first = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="case-s2",
                message="我的设备坏了，想报修",
            )
        )
    )
    second = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="case-s2",
                message="型号 AC900，无法启动，电话 13800138000",
            )
        )
    )

    assert first.debug["trace"]["selected_skills"] == ["case_intake"]
    assert second.debug["trace"]["selected_skills"] == ["case_intake"]
    assert second.debug["skill_selection"]["source"] == "sticky"
    assert "已为你完成售后受理信息收集" in second.answer


def test_runtime_exposes_skill_selection_and_manifests() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="skill-manifest-s1",
                message="我的设备坏了，想报修",
            )
        )
    )

    selection = response.debug["skill_selection"]
    assert selection["selected_skill"] == "case_intake"
    assert selection["source"] == "planned"
    manifest_by_name = {item["name"]: item for item in response.debug["skill_manifests"]}
    assert manifest_by_name["case_intake"]["sticky_policy"]["enabled"] is True
    assert "case-intake.collect_case_intake" in manifest_by_name["case_intake"]["required_tools"]


def test_manifest_driven_selector_can_choose_skill_without_planner_keyword(tmp_path) -> None:
    class StaticSelector(ManifestDrivenSkillSelector):
        async def select(self, context, manifests):
            _ = manifests
            return self.build_selection(
                selected_skill="product_support",
                reason="model selected manual QA from manifests",
                confidence=0.91,
                manifests=manifests,
            )

    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。",
        encoding="utf-8",
    )
    runtime = AgentRuntime(
        skill_registry=SkillRegistry(
            [ProductSupportSkill(knowledge_runtime=KnowledgeRuntime(StructuredManualBackend(manual_dir)))],
            selector=StaticSelector(),
        ),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="model-selector-s1",
                message="AC900 显示 E2",
            )
        )
    )

    assert response.debug["skill_selection"]["source"] == "model"
    assert response.debug["skill_selection"]["selected_skill"] == "product_support"
    assert response.debug["trace"]["selected_skills"] == ["product_support"]


def test_skill_registry_rejects_model_skill_with_missing_required_tools() -> None:
    class StaticSelector(ManifestDrivenSkillSelector):
        async def select(self, context, manifests):
            _ = context
            return self.build_selection(
                selected_skill="case_intake",
                reason="model selected case intake",
                confidence=0.9,
                manifests=manifests,
            )

    runtime = AgentRuntime(
        skill_registry=SkillRegistry([ProductSupportSkill(), FakeToolDependentSkill()], selector=StaticSelector()),
        tool_runtime=ToolRuntime(registry=ToolRegistry([])),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="missing-tool-s1",
                message="帮我处理售后",
            )
        )
    )

    assert response.debug["skill_selection"]["selected_skill"] is None
    assert response.debug["skill_selection"]["source"] == "none"
    assert "missing required tools" in response.debug["skill_selection"]["reason"]
    assert response.debug["trace"]["selected_skills"] == []


def test_model_selector_trusts_model_choice_without_skill_guard() -> None:
    class StaticSelector(ManifestDrivenSkillSelector):
        async def select(self, context, manifests):
            _ = context
            return self.build_selection(
                selected_skill="non_matching",
                reason="model selected skill directly",
                confidence=0.96,
                manifests=manifests,
            )

    runtime = AgentRuntime(
        skill_registry=SkillRegistry([FakeNonMatchingSkill()], selector=StaticSelector()),
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="model-trust-s1",
                message="你好，随便聊聊",
            )
        )
    )

    assert response.debug["skill_selection"]["selected_skill"] == "non_matching"
    assert response.debug["skill_selection"]["source"] == "model"
    assert response.debug["trace"]["selected_skills"] == ["non_matching"]
    assert "should not run" in response.answer


def test_sticky_policy_respects_max_turns() -> None:
    runtime = build_case_intake_runtime()
    case_skill = runtime.skill_registry.get("case_intake")
    case_skill.manifest = case_skill.manifest.model_copy(deep=True)
    case_manifest = case_skill.manifest
    case_manifest.sticky_policy.max_turns = 1

    asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="sticky-budget-s1",
                message="我的设备坏了，想报修",
            )
        )
    )
    first_followup = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="sticky-budget-s1",
                message="型号还不确定",
            )
        )
    )
    second_followup = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="sticky-budget-s1",
                message="我先不补了，稍后再说",
            )
        )
    )

    assert first_followup.debug["skill_selection"]["source"] == "sticky"
    assert second_followup.debug["skill_selection"]["source"] != "sticky"
    assert any(
        event["stage"] == "skill.sticky_overstay"
        for event in second_followup.debug["trace"]["events"]
    )


def test_supervisor_maps_skill_exception_to_fallback_result() -> None:
    runtime = AgentRuntime(skill_registry=SkillRegistry([FakeFailingSkill()]))

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="skill-error-s1",
                message="trigger failing skill",
            )
        )
    )

    assert response.debug["trace"]["selected_skills"] == ["failing_skill"]
    assert "暂时不可用" in response.answer
    assert any(
        event["stage"] == "skill.exception"
        for event in response.debug["trace"]["events"]
    )


def test_fallback_policy_retries_tool_error_once() -> None:
    flaky_tool = FakeFlakyCaseIntakeTool()
    runtime = AgentRuntime(
        tool_runtime=ToolRuntime(
            registry=ToolRegistry([
                ExtractCaseSlotsTool(),
                flaky_tool,
                FakeCaseIntakeTool("try_cancel_case_intake"),
            ])
        )
    )

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="tool-retry-s1",
                message="我的设备坏了，想报修",
            )
        )
    )

    assert flaky_tool.calls == 2
    assert response.debug["trace"]["selected_skills"] == ["case_intake"]
    collect_calls = [
        item
        for item in response.debug["trace"]["tool_calls"]
        if item["tool_name"] == "collect_case_intake"
    ]
    assert collect_calls[0]["ok"] is False
    assert collect_calls[1]["ok"] is True
    assert any(event["stage"] == "tool.retry" for event in response.debug["trace"]["events"])
    assert "请提供产品型号和联系电话" in response.answer


def test_skill_selection_harness_scores_cases() -> None:
    runtime = build_case_intake_runtime()
    harness = SkillSelectionHarness(runtime)

    report = asyncio.run(
        harness.run_cases(
            [
                SkillSelectionCase(
                    case_id="repair",
                    message="我的设备坏了，想报修",
                    acceptable_skills=["case_intake"],
                ),
                SkillSelectionCase(
                    case_id="general",
                    message="你好",
                    acceptable_skills=[None],
                ),
            ]
        )
    )

    assert report.total == 2
    assert report.passed == 2
    assert report.accuracy == 1.0


def test_golden_agent_dataset_is_precise_and_unique() -> None:
    cases = build_golden_agent_dataset()

    case_ids = [case.case_id for case in cases]
    categories = {case.category for case in cases}
    assert len(case_ids) == len(set(case_ids))
    assert len(cases) >= 8
    assert {"general", "product_support", "case_intake", "refund", "handoff", "composite", "boundary"} <= categories
    assert all(case.expected.acceptable_skills for case in cases)


def test_agent_evaluation_harness_outputs_metrics(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。AC900 滤网清洁步骤：关闭电源，清洁滤网，检查风道后重新启动。",
        encoding="utf-8",
    )
    runtime = build_eval_runtime(manual_dir)
    harness = AgentEvaluationHarness(runtime)

    report = asyncio.run(harness.run_cases(build_golden_agent_dataset()))

    assert report.total >= 8
    assert report.pass_rate >= 0.75
    assert report.skill_accuracy >= 0.75
    assert report.tool_accuracy >= 0.75
    assert report.safety_accuracy >= 0.9
    assert report.approval_rate > 0
    assert report.handoff_rate > 0
    assert report.avg_loop_turns >= 1
    assert "product_support" in report.by_category
    assert all(result.trace_id for result in report.results)


def test_tool_runtime_executes_skill_tool_call() -> None:
    runtime = AgentRuntime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="tool-s1",
                message="please run tool echo for this request",
            )
        )
    )

    tool_actions = [action for action in response.actions if action.kind == "tool"]
    assert response.debug["trace"]["selected_skills"] == ["tool_echo"]
    assert tool_actions[0].name == "mock.echo"
    assert tool_actions[0].status == "success"
    assert response.debug["trace"]["tool_calls"][0]["ok"] is True
    assert response.debug["loop"]["turn_count"] == 2
    assert response.debug["loop"]["stop_reason"] == "no_tool_calls"


def test_context_governance_uses_transcript_history() -> None:
    transcript_store = InMemoryTranscriptStore()
    runtime = AgentRuntime(transcript_store=transcript_store)

    asyncio.run(runtime.run(AgentRequest(session_id="history-s1", message="第一轮问题")))
    response = asyncio.run(runtime.run(AgentRequest(session_id="history-s1", message="第二轮问题")))

    context_events = response.debug["trace"]["context_events"]
    assert context_events[0]["transcript_chars"] > 0
    governed_events = [
        event for event in response.debug["trace"]["events"]
        if event["stage"] == "context.govern"
    ]
    assert governed_events[0]["payload"]["transcript_chars"] > 0


def test_case_intake_writes_formal_issue_memory() -> None:
    runtime = build_case_intake_runtime()

    response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="memory-case-s1",
                message="我的设备坏了，想报修",
            )
        )
    )

    memory = runtime.memory_store.load("memory-case-s1")
    assert memory.active_thread_id
    thread = memory.active_thread()
    assert thread is not None
    assert thread.issue_type == "repair"
    assert thread.status == "waiting_user"
    assert "case_intake.status" in thread.facts
    assert thread.facts["case_intake.status"].value == "collecting"
    assert "case_intake" in memory.flat_state
    assert response.state_summary.startswith("active_thread_id=")


def test_tool_runtime_blocks_high_risk_tool_call() -> None:
    runtime = AgentRuntime()
    context_response = asyncio.run(
        runtime.run(
            AgentRequest(
                session_id="tool-s2",
                message="hello",
            )
        )
    )
    trace = context_response.debug["trace"]
    assert trace["tool_calls"] == []

    direct_runtime = ToolRuntime()
    memory = runtime.memory_store.load("tool-s2")
    assert memory.flat_state == {}

    async def call_blocked_tool() -> bool:
        from nikon0.app.schemas.agent import AgentContext
        from nikon0.app.schemas.trace import ExecutionTrace

        agent_context = AgentContext(
            request=AgentRequest(session_id="tool-s3", message="danger"),
            trace=ExecutionTrace(trace_id="trace-tool-block", session_id="tool-s3", user_message="danger"),
        )
        result = await direct_runtime.call(
            agent_context,
            ToolCallRequest(
                service_id="mock",
                tool_name="echo",
                arguments={"message": "danger"},
                risk_level="high",
            ),
        )
        return result.error_code == "permission_denied" and agent_context.trace.tool_calls[0]["blocked"]

    assert asyncio.run(call_blocked_tool())


def test_tool_runtime_runs_hooks() -> None:
    def pre_hook(context, request):
        _ = context
        assert request.tool_name == "echo"
        from nikon0.app.schemas.capability import PermissionDecision

        return PermissionDecision(allowed=True, reason="custom pre ok")

    def post_hook(context, request, result):
        _ = (context, request)
        return f"custom post ok={result.ok}"

    runtime = ToolRuntime(hook_runner=HookRunner(pre_tool=(pre_hook,), post_tool=(post_hook,)))

    async def call_tool() -> list[str]:
        from nikon0.app.schemas.agent import AgentContext
        from nikon0.app.schemas.trace import ExecutionTrace

        agent_context = AgentContext(
            request=AgentRequest(session_id="hook-s1", message="hook"),
            trace=ExecutionTrace(trace_id="trace-hook", session_id="hook-s1", user_message="hook"),
        )
        await runtime.call(
            agent_context,
            ToolCallRequest(service_id="mock", tool_name="echo", arguments={"x": "1"}),
        )
        return [event.stage for event in agent_context.trace.events]

    stages = asyncio.run(call_tool())
    assert "tool.pre_tool" in stages
    assert "tool.post_tool" in stages


def test_tool_runtime_returns_approval_for_approval_required_call() -> None:
    runtime = AgentRuntime()

    async def call_approval_tool() -> dict:
        from nikon0.app.schemas.agent import AgentContext
        from nikon0.app.schemas.trace import ExecutionTrace

        agent_context = AgentContext(
            request=AgentRequest(session_id="tool-approval", message="approval tool"),
            trace=ExecutionTrace(
                trace_id="trace-tool-approval",
                session_id="tool-approval",
                user_message="approval tool",
            ),
        )
        result = await runtime.tool_runtime.call(
            agent_context,
            ToolCallRequest(
                service_id="mock",
                tool_name="echo",
                arguments={"message": "approval"},
                risk_level="medium",
                requires_approval=True,
            ),
        )
        return {"result": result.model_dump(), "trace": agent_context.trace.model_dump()}

    payload = asyncio.run(call_approval_tool())
    assert payload["result"]["ok"] is False
    assert payload["result"]["error_code"] == "approval_required"
    assert payload["result"]["data"]["approval_request"]["approval_type"] == "tool_call"
    assert payload["trace"]["tool_calls"][0]["approval_id"]


def test_mcp_gateway_tool_maps_success_payload() -> None:
    class FakeGatewayClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict]] = []

        def call_tool(self, *, service_id: str, tool_name: str, arguments: dict) -> dict:
            self.calls.append((service_id, tool_name, arguments))
            return {"has_pending": True}

    client = FakeGatewayClient()
    tool = McpGatewayTool(
        service_id="case-intake",
        tool_name="get_case_intake_status",
        client=client,
    )

    result = asyncio.run(
        tool.call(
            ToolCallRequest(
                service_id="case-intake",
                tool_name="get_case_intake_status",
                arguments={"session_id": "sid1"},
            )
        )
    )

    assert result.ok is True
    assert result.data == {"has_pending": True}
    assert client.calls == [("case-intake", "get_case_intake_status", {"session_id": "sid1"})]


def test_mcp_gateway_tool_maps_client_error() -> None:
    class FailingGatewayClient:
        def call_tool(self, *, service_id: str, tool_name: str, arguments: dict) -> dict:
            _ = (service_id, tool_name, arguments)
            raise RuntimeError("gateway unavailable")

    tool = McpGatewayTool(
        service_id="case-intake",
        tool_name="get_case_intake_status",
        client=FailingGatewayClient(),
    )

    result = asyncio.run(
        tool.call(
            ToolCallRequest(
                service_id="case-intake",
                tool_name="get_case_intake_status",
                arguments={"session_id": "sid1"},
            )
        )
    )

    assert result.ok is False
    assert result.error_code == "RuntimeError"
    assert "gateway unavailable" in (result.error_message or "")


def test_default_tool_runtime_lists_mcp_specs() -> None:
    runtime = ToolRuntime()

    specs = asyncio.run(runtime.list_tools("case-intake"))

    names = {spec.tool_name for spec in specs}
    assert {"get_case_intake_status", "collect_case_intake", "try_cancel_case_intake"} <= names


def test_chat_api() -> None:
    import nikon0.app.api.v1.chat as chat_module

    chat_module.runtime = AgentRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "api-s1", "message": "你好，介绍一下 nikon0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace_id"]
    assert payload["actions"][0]["name"] == "supervisor"
    assert payload["debug"]["trace"]["selected_skills"] == []
    assert payload["debug"]["loop"]["stop_reason"] == "no_tool_calls"


def test_approval_api_lists_and_updates_requests() -> None:
    import nikon0.app.api.v1.chat as chat_module

    chat_module.runtime = AgentRuntime()
    api_runtime = chat_module.runtime

    approval = ApprovalRequest(
        approval_id="approval-api-test",
        trace_id="trace-api-test",
        session_id="api-approval-s1",
        approval_type="answer",
        title="Approval API Test",
        reason="test",
        risk_level="high",
        requested_action="send_answer",
    )
    api_runtime.approval_store.create_approval(approval)
    client = TestClient(app)

    listed = client.get("/api/v1/approvals", params={"session_id": "api-approval-s1"})
    updated = client.post("/api/v1/approvals/approval-api-test/approved")

    assert listed.status_code == 200
    assert listed.json()[0]["approval_id"] == "approval-api-test"
    assert updated.status_code == 200
    assert updated.json()["status"] == "approved"
