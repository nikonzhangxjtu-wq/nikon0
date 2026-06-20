"""Typed contracts for governed memory reads and writes."""

from __future__ import annotations

import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import RiskLevel, StateUpdate

MemoryProvenance = Literal["user", "verified_tool", "workflow", "skill", "model", "runtime"]
MemoryScope = Literal["thread", "session"]
WriteOutcome = Literal["accept", "reject", "needs_confirmation", "no_op"]
ThreadAction = Literal["continue_active", "switch_open_thread", "create_thread", "close_thread", "needs_clarification"]


class StateUpdateCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"memcand_{uuid4().hex}")
    update: StateUpdate
    target_thread_id: str | None = None
    create_thread: bool = False
    scope: MemoryScope = "thread"
    provenance: MemoryProvenance = "skill"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    risk_level: RiskLevel = "low"
    idempotency_key: str = ""
    source_agent: str = ""
    execution_stage: str = ""
    created_at: float = Field(default_factory=time.time)


class MemoryConflict(BaseModel):
    field: str
    existing_value: Any
    incoming_value: Any
    existing_provenance: MemoryProvenance = "runtime"
    incoming_provenance: MemoryProvenance = "skill"
    reason: str


class MemoryWriteDecision(BaseModel):
    candidate_id: str
    outcome: WriteOutcome
    target_thread_id: str | None = None
    reason: str
    conflicts: list[MemoryConflict] = Field(default_factory=list)
    update: StateUpdate | None = None
    degraded: bool = False


class ThreadDecision(BaseModel):
    action: ThreadAction
    thread_id: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["llm", "deterministic"] = "deterministic"
    reason: str = ""


class MemoryReadPlan(BaseModel):
    thread_ids: list[str] = Field(default_factory=list)
    include_session_facts: bool = False
    include_workflow: bool = False
    include_ticket_history: bool = False
    source: Literal["llm", "deterministic"] = "deterministic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = ""
    fallback_reason: str = ""
