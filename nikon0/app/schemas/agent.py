"""Agent request, response, and runtime context schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from nikon0.app.schemas.capability import Evidence, RiskLevel, SkillSelection, ToolSpec
from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.app.schemas.planner import PlannerResult
from nikon0.app.schemas.trace import ExecutionTrace


class AgentRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    message: str
    images: list[str] = Field(default_factory=list)
    channel: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "message")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value


class AgentActionRecord(BaseModel):
    kind: str
    name: str
    status: str
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    answer: str
    images: list[str] = Field(default_factory=list)
    state_summary: str = ""
    risk_level: RiskLevel = "low"
    trace_id: str
    actions: list[AgentActionRecord] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    request: AgentRequest
    session_state: SessionIssueMemory | None = None
    memory_context: str = ""
    transcript_context: str = ""
    governed_context: str = ""
    context_pack: Any | None = None
    context_governance: Any | None = None
    evidence_context: list[Evidence] = Field(default_factory=list)
    selected_agent: str | None = None
    selected_skill: str | None = None
    skill_selection: SkillSelection | None = None
    plan: PlannerResult | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_runtime: Any | None = None
    allowed_tool_names: set[str] = Field(default_factory=set)
    agent_handoff: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[ToolSpec] = Field(default_factory=list)
    retry_tool_errors: bool = False
    loop_turn: int = 0
    max_turns: int = 4
    trace: ExecutionTrace
