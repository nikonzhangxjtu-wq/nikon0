"""Execution trace schemas."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class ExecutionTrace(BaseModel):
    trace_id: str
    session_id: str
    user_message: str
    selected_agents: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    context_events: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    safety_decisions: list[dict[str, Any]] = Field(default_factory=list)
    memory_updates: list[dict[str, Any]] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    final_risk_level: str = "low"

    def add_event(self, stage: str, message: str, **payload: Any) -> None:
        self.events.append(TraceEvent(stage=stage, message=message, payload=payload))
