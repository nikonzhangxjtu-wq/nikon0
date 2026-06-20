"""Tests for skill routing prompt, signals, and supervisor thresholds."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nikon0.agent.planner import RuleBasedPlanner
from nikon0.agent.runtime import AgentRuntime
from nikon0.agent.supervisor import SupervisorAgent
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.model_selector import LlmSkillSelector
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.skills.routing_signals import looks_like_case_intake, looks_like_product_support
from nikon0.skills.skill_router_prompt import ROUTING_EXCLUDED_SKILLS, SKILL_ROUTER_SYSTEM
from nikon0.tools.runtime import default_tools


DATASET_PATH = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "product_support_40.jsonl"


class FakeSelectorClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload


def _context(message: str) -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="route-test", message=message),
        available_tools=[tool.spec for tool in default_tools()],
        trace=ExecutionTrace(trace_id="route-test-trace", session_id="route-test", user_message=message),
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("空气净化器滤网多久更换一次？", True),
        ("洗碗机喷淋臂多久清洁？", True),
        ("洗碗机可以预约多久？", True),
        ("洗碗机的安全锁怎么开关？", True),
        ("洗碗机为什么不能用排插？", True),
        ("摩托艇启动时能不能握油门？", True),
        ("相机能外接交流电吗？", True),
        ("请问你们家的商品支持7天无理由退换货吗？", False),
        ("我要退款，订单已经付款了", False),
    ],
)
def test_product_support_signal_detection(message: str, expected: bool) -> None:
    matched, _hits = looks_like_product_support(message)
    assert matched is expected


def test_manual_prohibition_is_not_case_intake() -> None:
    assert looks_like_case_intake("洗碗机为什么不能用排插？") is False


def test_multilingual_rule_floor_routes_manual_and_case_intake_requests() -> None:
    assert looks_like_product_support("How do I set the child lock on the microwave?")[0] is True
    assert looks_like_case_intake("I need a warranty claim because my microwave is not heating") is True


def test_skill_router_prompt_contains_classification_rules() -> None:
    assert "product_support" in SKILL_ROUTER_SYSTEM
    assert "case_intake" in SKILL_ROUTER_SYSTEM
    assert "为什么不能用排插" in SKILL_ROUTER_SYSTEM
    assert "selected_skill" in SKILL_ROUTER_SYSTEM


def test_llm_selector_prompt_uses_chinese_manifest_and_excludes_mock() -> None:
    client = FakeSelectorClient('{"selected_skill":"product_support","confidence":0.92,"reason":"manual QA"}')
    selector = LlmSkillSelector(client)
    manifests = (
        ProductSupportSkill().manifest,
        CaseIntakeSkill().manifest,
    )
    prompt = LlmSkillSelector._build_prompt(_context("空气净化器滤网多久更换一次？"), manifests)

    assert "商品说明书知识问答" in prompt
    assert "mock_enterprise_assistant" not in prompt
    assert "空气净化器滤网多久更换一次？" in prompt
    asyncio.run(selector.select(_context("空气净化器滤网多久更换一次？"), manifests))
    assert client.prompts
    assert "商品说明书知识问答" in client.prompts[0]


def test_planner_recommends_product_support_for_short_manual_questions() -> None:
    planner = RuleBasedPlanner()
    for message in (
        "洗碗机喷淋臂多久清洁？",
        "洗碗机可以预约多久？",
        "洗碗机的安全锁怎么开关？",
    ):
        plan = planner.plan(_context(message))
        assert plan.recommended_skill == "product_support"


def test_planner_does_not_send_manual_prohibition_to_case_intake() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan(_context("洗碗机为什么不能用排插？"))
    assert plan.recommended_skill == "product_support"


def test_supervisor_accepts_product_support_rule_fallback_at_082() -> None:
    registry = SkillRegistry([ProductSupportSkill(), CaseIntakeSkill()])
    supervisor = SupervisorAgent(registry, answer_generator=None)
    context = _context("洗碗机喷淋臂多久清洁？")
    result = asyncio.run(supervisor.run(context))
    assert result.selected_skills == ["product_support"]


def test_rule_fallback_routes_all_product_support_40_messages() -> None:
    if not DATASET_PATH.exists():
        pytest.skip("product_support_40 dataset not found")

    runtime = AgentRuntime(
        skill_registry=SkillRegistry(
            [
                CaseIntakeSkill(),
                ProductSupportSkill(
                    knowledge_runtime=KnowledgeRuntime(StructuredManualBackend("missing"))
                ),
            ],
            selector=None,
        ),
        answer_generator=None,
    )
    missed: list[str] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        message = str(row["message"])
        response = asyncio.run(runtime.run(AgentRequest(session_id="eval-route", message=message)))
        selection = response.debug.get("skill_selection") or {}
        if selection.get("selected_skill") != "product_support":
            missed.append(f"{row['case_id']}: {message} -> {selection.get('selected_skill')}")

    assert not missed, "routing misses:\n" + "\n".join(missed)


def test_eval_runtime_excludes_mock_skill() -> None:
    from nikon0.eval.run_agent_eval import build_eval_runtime
    from nikon0.eval.runtime_profiles import EvalRuntimeProfile

    runtime = build_eval_runtime(
        manual_dir="missing",
        use_real_llm=False,
        local_rag=True,
        runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
    )
    names = {skill.name for skill in runtime.skill_registry.list()}
    assert "mock_enterprise_assistant" not in names
    assert "product_support" in names


def test_routing_excluded_skills_constant() -> None:
    assert "mock_enterprise_assistant" in ROUTING_EXCLUDED_SKILLS
