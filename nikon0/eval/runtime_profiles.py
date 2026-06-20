"""Runtime profiles for production-aligned evaluation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nikon0.agent.context_governance import ContextGovernance
from nikon0.agent.runtime import (
    AgentRuntime,
    _build_default_context_governance,
    _build_default_memory_read_planner,
    _build_default_skill_registry,
)
from nikon0.context.conversation import ConversationCompactor
from nikon0.context.evidence import EvidenceContextManager
from nikon0.context.llm_compaction import LlmConversationCompactor
from nikon0.context.llm_span_selector import LlmEvidenceSpanSelector
from nikon0.context.read_planner import DeterministicContextReadPlanner, LlmContextReadPlanner
from nikon0.context.runtime import ContextRuntime
from nikon0.knowledge.runtime import EnterpriseRagBackend, KnowledgeRuntime, StructuredManualBackend
from nikon0.llm import BailianOllamaChatClient, LlmAnswerGenerator
from nikon0.memory.persistence import build_memory_store_from_env
from nikon0.skills.base import SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.model_selector import BailianOllamaSkillSelectionClient, LlmSkillSelector
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.skills.tool_echo import ToolEchoSkill
from nikon0.tools.runtime import EchoTool, ToolRegistry, ToolRuntime, default_tools


class EvalRuntimeProfile(str, Enum):
    DETERMINISTIC = "deterministic"
    PRODUCTION_LIKE = "production_like"
    PRODUCTION_LIKE_NO_LLM = "production_like_no_llm"
    LEGACY_EVAL = "legacy_eval"


class RuntimeProfileAudit(BaseModel):
    runtime_profile: str
    context_profile: str
    context_governance_enabled: bool = True
    llm_context_components_enabled: dict[str, bool] = Field(default_factory=dict)
    mock_skill_enabled: bool = False
    mock_tool_names: list[str] = Field(default_factory=list)
    case_intake_tool_mode: str = "production"
    rag_backend_policy: dict[str, Any] = Field(default_factory=dict)
    eval_runtime_matches_production: bool = False
    production_mismatch_reasons: list[str] = Field(default_factory=list)
    memory_store_type: str = "InMemorySessionIssueStore"


class ProfiledRuntime(BaseModel):
    runtime: AgentRuntime
    audit: RuntimeProfileAudit

    model_config = {"arbitrary_types_allowed": True}


def build_profiled_eval_runtime(
    *,
    runtime_profile: str | EvalRuntimeProfile = EvalRuntimeProfile.PRODUCTION_LIKE,
    manual_dir: str | Path,
    use_real_llm: bool = True,
    local_rag: bool = False,
    mock_case_intake_tool: bool | None = None,
) -> ProfiledRuntime:
    profile = coerce_runtime_profile(runtime_profile)
    answer_generator = _build_answer_generator(use_real_llm=use_real_llm and profile != EvalRuntimeProfile.PRODUCTION_LIKE_NO_LLM)
    context_governance = _context_governance_for_profile(profile)
    product_knowledge = _build_product_knowledge_runtime(
        manual_dir=manual_dir,
        local_rag=local_rag,
        structured_manual_fallback=True,
    )
    use_mock_case_intake = _use_mock_case_intake_tool(profile, mock_case_intake_tool)

    if profile == EvalRuntimeProfile.PRODUCTION_LIKE:
        skill_registry = _production_like_skill_registry(
            answer_generator=answer_generator,
            product_knowledge=product_knowledge,
        )
    else:
        selector = _build_selector(
            enabled=use_real_llm and profile == EvalRuntimeProfile.LEGACY_EVAL,
        )
        skill_registry = SkillRegistry(
            _eval_skills(answer_generator=answer_generator, product_knowledge=product_knowledge),
            selector=selector,
        )

    tool_runtime = _tool_runtime_for_profile(
        profile,
        mock_case_intake_tool=use_mock_case_intake,
    )
    memory_store = (
        build_memory_store_from_env()
        if profile in {EvalRuntimeProfile.PRODUCTION_LIKE, EvalRuntimeProfile.PRODUCTION_LIKE_NO_LLM}
        else None
    )
    runtime = AgentRuntime(
        skill_registry=skill_registry,
        context_governance=context_governance,
        tool_runtime=tool_runtime,
        answer_generator=answer_generator,
        memory_store=memory_store,
        memory_read_planner=(
            _build_default_memory_read_planner()
            if profile == EvalRuntimeProfile.PRODUCTION_LIKE
            else None
        ),
    )
    audit = runtime_profile_audit(
        runtime=runtime,
        runtime_profile=profile,
        context_governance=context_governance,
        local_rag=local_rag,
        mock_case_intake_tool=use_mock_case_intake,
        memory_store=memory_store,
    )
    return ProfiledRuntime(runtime=runtime, audit=audit)


def coerce_runtime_profile(value: str | EvalRuntimeProfile) -> EvalRuntimeProfile:
    if isinstance(value, EvalRuntimeProfile):
        return value
    try:
        return EvalRuntimeProfile(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EvalRuntimeProfile)
        raise ValueError(f"unknown runtime profile {value!r}; allowed: {allowed}") from exc


def runtime_profile_audit(
    *,
    runtime: AgentRuntime,
    runtime_profile: EvalRuntimeProfile,
    context_governance: ContextGovernance,
    local_rag: bool,
    mock_case_intake_tool: bool,
    memory_store=None,
) -> RuntimeProfileAudit:
    skill_names = [skill.name for skill in runtime.skill_registry.list()]
    mock_skill_enabled = "mock_enterprise_assistant" in skill_names
    mock_tool_names = [
        f"{spec.service_id}.{spec.tool_name}"
        for spec in runtime.tool_runtime.registry.list()
        if spec.service_id == "mock"
    ]
    if mock_case_intake_tool:
        mock_tool_names.extend(
            f"{spec.service_id}.{spec.tool_name}"
            for spec in runtime.tool_runtime.registry.list("case-intake")
            if spec.tool_name in {"collect_case_intake", "try_cancel_case_intake"}
        )
    context_components = _context_component_flags(context_governance)
    mismatch_reasons: list[str] = []
    if mock_skill_enabled:
        mismatch_reasons.append("mock skill is enabled")
    if mock_case_intake_tool:
        mismatch_reasons.append("case-intake uses eval mock tool")
    if runtime_profile in {EvalRuntimeProfile.DETERMINISTIC, EvalRuntimeProfile.LEGACY_EVAL}:
        mismatch_reasons.append("runtime profile is not production-like")
    if runtime_profile == EvalRuntimeProfile.PRODUCTION_LIKE_NO_LLM:
        mismatch_reasons.append("LLM components are disabled")
    return RuntimeProfileAudit(
        runtime_profile=runtime_profile.value,
        context_profile=_context_profile_name(context_components),
        context_governance_enabled=context_governance is not None,
        llm_context_components_enabled=context_components,
        mock_skill_enabled=mock_skill_enabled,
        mock_tool_names=mock_tool_names,
        case_intake_tool_mode="mock" if mock_case_intake_tool else "production",
        rag_backend_policy={
            "primary": "structured_manual" if local_rag else "enterprise_rag",
            "local_rag": local_rag,
            "structured_manual_fallback": not local_rag,
        },
        eval_runtime_matches_production=not mismatch_reasons,
        production_mismatch_reasons=mismatch_reasons,
        memory_store_type=type(memory_store).__name__ if memory_store is not None else type(runtime.memory_store).__name__,
    )


def _context_governance_for_profile(profile: EvalRuntimeProfile) -> ContextGovernance:
    if profile == EvalRuntimeProfile.PRODUCTION_LIKE:
        return _build_default_context_governance()
    if profile == EvalRuntimeProfile.PRODUCTION_LIKE_NO_LLM:
        return _build_default_context_governance(settings=_NoLlmContextSettings())
    return _deterministic_context_governance()


def _deterministic_context_governance() -> ContextGovernance:
    return ContextGovernance(
        context_runtime=ContextRuntime(
            read_planner=DeterministicContextReadPlanner(),
            conversation_compactor=ConversationCompactor(),
            evidence_manager=EvidenceContextManager(),
        )
    )


def _production_like_skill_registry(
    *,
    answer_generator: LlmAnswerGenerator | None,
    product_knowledge: KnowledgeRuntime,
) -> SkillRegistry:
    registry = _build_default_skill_registry(answer_generator=answer_generator)
    skills = []
    for skill in registry.list():
        if skill.name == "product_support":
            skills.append(ProductSupportSkill(knowledge_runtime=product_knowledge, answer_generator=answer_generator))
        else:
            skills.append(skill)
    return SkillRegistry(skills, selector=registry.selector)


def _eval_skills(
    *,
    answer_generator: LlmAnswerGenerator | None,
    product_knowledge: KnowledgeRuntime,
) -> list:
    return [
        ToolEchoSkill(),
        CaseIntakeSkill(),
        ProductSupportSkill(
            knowledge_runtime=product_knowledge,
            answer_generator=answer_generator,
        ),
    ]


def _tool_runtime_for_profile(
    profile: EvalRuntimeProfile,
    *,
    mock_case_intake_tool: bool,
) -> ToolRuntime:
    if mock_case_intake_tool:
        from nikon0.eval.run_agent_eval import _EvalCaseIntakeTool
        from nikon0.tools.case_intake import ExtractCaseSlotsTool
        from nikon0.tools.memory import ReadSessionMemoryTool, WriteSessionFactTool
        from nikon0.tools.product import ResolveProductTool, SearchProductManualTool, ValidateAnswerGroundingTool

        tools = [
            _EvalCaseIntakeTool("collect_case_intake"),
            _EvalCaseIntakeTool("try_cancel_case_intake"),
            ExtractCaseSlotsTool(),
            ResolveProductTool(),
            SearchProductManualTool(),
            ValidateAnswerGroundingTool(),
            ReadSessionMemoryTool(),
            WriteSessionFactTool(),
            EchoTool(),
        ]
        return ToolRuntime(registry=ToolRegistry(tools))
    if profile == EvalRuntimeProfile.LEGACY_EVAL:
        return ToolRuntime(registry=ToolRegistry([EchoTool()]))
    return ToolRuntime()


def _use_mock_case_intake_tool(profile: EvalRuntimeProfile, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return profile in {EvalRuntimeProfile.DETERMINISTIC, EvalRuntimeProfile.LEGACY_EVAL}


def _build_product_knowledge_runtime(
    *,
    manual_dir: str | Path,
    local_rag: bool = False,
    structured_manual_fallback: bool = True,
) -> KnowledgeRuntime:
    local_backend = StructuredManualBackend(manual_dir)
    if local_rag:
        return KnowledgeRuntime(local_backend)
    if structured_manual_fallback:
        return KnowledgeRuntime(EnterpriseRagBackend(fallback_backend=local_backend))
    return KnowledgeRuntime(EnterpriseRagBackend())


def _build_answer_generator(*, use_real_llm: bool) -> LlmAnswerGenerator | None:
    if not use_real_llm:
        return None
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return None
    model = getattr(settings, "simple_llm_model", "") or getattr(settings, "gen_model", "")
    if not model:
        return None
    return LlmAnswerGenerator(
        BailianOllamaChatClient(
            model=model,
            temperature=float(getattr(settings, "gen_temperature_competition", 0.1) or 0.1),
            max_tokens=int(getattr(settings, "gen_max_tokens", 1024) or 1024),
            timeout=30,
        )
    )


def _build_selector(*, enabled: bool):
    if not enabled:
        return None
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return None
    if not bool(getattr(settings, "router_llm_enabled", False)):
        return None
    model = (
        getattr(settings, "router_llm_model", "")
        or getattr(settings, "simple_llm_model", "")
        or getattr(settings, "gen_model", "")
    )
    if not model:
        return None
    return LlmSkillSelector(
        BailianOllamaSkillSelectionClient(
            model=model,
            temperature=0.0,
            max_tokens=256,
            timeout=15,
        )
    )


def _context_component_flags(context_governance: ContextGovernance) -> dict[str, bool]:
    runtime = context_governance.context_runtime
    return {
        "read_planner": isinstance(runtime.read_planner, LlmContextReadPlanner),
        "conversation_compactor": isinstance(runtime.conversation_compactor, LlmConversationCompactor),
        "evidence_span_selector": isinstance(runtime.evidence_manager.span_selector, LlmEvidenceSpanSelector),
    }


def _context_profile_name(flags: dict[str, bool]) -> str:
    return "production_like_llm" if any(flags.values()) else "deterministic"


class _NoLlmContextSettings:
    nikon0_context_llm_enabled = False
    nikon0_context_total_char_budget = 9000
