from __future__ import annotations

from nikon0.agent.runtime import _build_default_context_governance, _build_default_skills
from nikon0.context.conversation import ConversationCompactor
from nikon0.context.llm_compaction import LlmConversationCompactor
from nikon0.context.llm_span_selector import LlmEvidenceSpanSelector
from nikon0.context.read_planner import DeterministicContextReadPlanner, LlmContextReadPlanner
from nikon0.eval.runtime_profiles import EvalRuntimeProfile, build_profiled_eval_runtime
from nikon0.knowledge.runtime import EnterpriseRagBackend
from nikon0.skills.product_support import ProductSupportSkill


class _MockSkillOffSettings:
    nikon0_enable_mock_skill = False


class _MockSkillOnSettings:
    nikon0_enable_mock_skill = True


class _LlmContextSettings:
    nikon0_context_llm_enabled = True
    nikon0_context_llm_model = "deepseek-v4-flash"
    simple_llm_model = "deepseek-v4-flash"
    gen_model = "deepseek-v4-flash"
    nikon0_context_llm_timeout = 3
    nikon0_context_llm_max_tokens = 128
    nikon0_context_total_char_budget = 1234
    nikon0_enable_mock_skill = False
    router_llm_enabled = False
    mcp_gateway_endpoint = "http://127.0.0.1:18080/mcp"
    mcp_gateway_bearer_token = ""
    mcp_gateway_timeout_sec = 1


def test_default_skills_exclude_mock_skill_by_default() -> None:
    names = {skill.name for skill in _build_default_skills()}

    assert "mock_enterprise_assistant" not in names


def test_default_skills_can_enable_mock_skill_explicitly(monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.settings", _MockSkillOnSettings())

    names = {skill.name for skill in _build_default_skills()}

    assert "mock_enterprise_assistant" in names


def test_production_like_profile_uses_default_context_governance(monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.settings", _LlmContextSettings())

    expected = _build_default_context_governance(settings=_LlmContextSettings()).context_runtime
    profiled = build_profiled_eval_runtime(
        runtime_profile=EvalRuntimeProfile.PRODUCTION_LIKE,
        manual_dir="missing",
        use_real_llm=False,
        local_rag=True,
    )
    actual = profiled.runtime.context_governance.context_runtime

    assert isinstance(expected.read_planner, LlmContextReadPlanner)
    assert isinstance(actual.read_planner, LlmContextReadPlanner)
    assert isinstance(actual.conversation_compactor, LlmConversationCompactor)
    assert isinstance(actual.evidence_manager.span_selector, LlmEvidenceSpanSelector)
    assert profiled.audit.runtime_profile == "production_like"
    assert profiled.audit.context_profile == "production_like_llm"
    assert profiled.audit.mock_skill_enabled is False


def test_deterministic_profile_uses_deterministic_context() -> None:
    profiled = build_profiled_eval_runtime(
        runtime_profile=EvalRuntimeProfile.DETERMINISTIC,
        manual_dir="missing",
        use_real_llm=False,
        local_rag=True,
    )
    runtime = profiled.runtime.context_governance.context_runtime

    assert isinstance(runtime.read_planner, DeterministicContextReadPlanner)
    assert isinstance(runtime.conversation_compactor, ConversationCompactor)
    assert runtime.evidence_manager.span_selector is None
    assert profiled.audit.context_profile == "deterministic"
    assert profiled.audit.case_intake_tool_mode == "mock"


def test_production_like_profile_disallows_mock_skill() -> None:
    profiled = build_profiled_eval_runtime(
        runtime_profile=EvalRuntimeProfile.PRODUCTION_LIKE,
        manual_dir="missing",
        use_real_llm=False,
        local_rag=True,
    )
    names = {skill.name for skill in profiled.runtime.skill_registry.list()}

    assert "mock_enterprise_assistant" not in names


def test_production_like_profile_records_enterprise_rag_fallback_policy() -> None:
    profiled = build_profiled_eval_runtime(
        runtime_profile=EvalRuntimeProfile.PRODUCTION_LIKE,
        manual_dir="missing",
        use_real_llm=False,
        local_rag=False,
    )
    product_support = profiled.runtime.skill_registry.get("product_support")

    assert isinstance(product_support, ProductSupportSkill)
    assert isinstance(product_support.knowledge_runtime.backend, EnterpriseRagBackend)
    assert product_support.knowledge_runtime.backend.fallback_backend is not None
    assert profiled.audit.rag_backend_policy["structured_manual_fallback"] is True
