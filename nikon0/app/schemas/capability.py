"""Capability schemas shared by agents, skills, tools, and safety gates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]
CapabilityStatus = Literal["success", "needs_more_info", "failed", "handoff_required"]
SkillSelectionSource = Literal["sticky", "model", "planned", "rule_fallback", "none"]


class Evidence(BaseModel):
    """A traceable fact or source used by the assistant."""

    evidence_id: str
    source: str
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StateUpdate(BaseModel):
    """A proposed update to session issue state."""

    key: str
    value: Any
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: str = "skill"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ToolCallRequest(BaseModel):
    """A normalized request for ToolRuntime."""

    service_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = "low"
    requires_approval: bool = False


class ToolSpec(BaseModel):
    """A runtime-visible tool descriptor."""

    service_id: str
    tool_name: str
    description: str = ""
    risk_level: RiskLevel = "low"
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Normalized result returned by ToolRuntime."""

    ok: bool
    service_id: str
    tool_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PermissionDecision(BaseModel):
    """Permission decision for tool execution."""

    allowed: bool
    reason: str
    blocked_action: str | None = None


class SkillMatch(BaseModel):
    matched: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    required_inputs: list[str] = Field(default_factory=list)


class StickyPolicy(BaseModel):
    """Platform-level continuation policy for multi-turn skills."""

    enabled: bool = False
    continue_when: list[str] = Field(default_factory=list)
    exit_when: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=4, ge=1)
    priority: int = 100


class FallbackPolicy(BaseModel):
    """How the platform should degrade when a skill cannot complete."""

    allow_general_fallback: bool = True
    allow_handoff: bool = False
    retry_on_tool_error: bool = False


class SkillManifest(BaseModel):
    """Runtime-visible skill descriptor for discovery and governance."""

    name: str
    title: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    sticky_policy: StickyPolicy = Field(default_factory=StickyPolicy)
    fallback_policy: FallbackPolicy = Field(default_factory=FallbackPolicy)


class SkillCandidate(BaseModel):
    name: str
    matched: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    source: SkillSelectionSource = "rule_fallback"


class RejectedSkill(BaseModel):
    name: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SkillSelection(BaseModel):
    selected_skill: str | None = None
    candidates: list[SkillCandidate] = Field(default_factory=list)
    source: SkillSelectionSource = "none"
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rejected: list[RejectedSkill] = Field(default_factory=list)


class SkillResult(BaseModel):
    status: CapabilityStatus
    answer_draft: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    state_updates: list[StateUpdate] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    handoff_reason: str | None = None


class AgentMatch(BaseModel):
    matched: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str


class AgentResult(BaseModel):
    status: CapabilityStatus
    answer_draft: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    state_updates: list[StateUpdate] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    selected_skills: list[str] = Field(default_factory=list)
    handoff_reason: str | None = None
