"""Safety and approval schemas."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import RiskLevel


ApprovalStatus = Literal["pending", "approved", "rejected", "expired", "cancelled"]
ApprovalType = Literal["tool_call", "handoff", "answer"]


class ApprovalRequest(BaseModel):
    approval_id: str
    trace_id: str
    session_id: str
    approval_type: ApprovalType
    title: str
    reason: str
    risk_level: RiskLevel
    requested_action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: ApprovalStatus = "pending"
    created_at: float = Field(default_factory=time.time)


class HandoffRequest(BaseModel):
    handoff_id: str
    trace_id: str
    session_id: str
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class SafetyDecision(BaseModel):
    allowed: bool
    risk_level: RiskLevel
    requires_human: bool = False
    reason: str
    blocked_actions: list[str] = Field(default_factory=list)
    approval_request: ApprovalRequest | None = None
    handoff_request: HandoffRequest | None = None
