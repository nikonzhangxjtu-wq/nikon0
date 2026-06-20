"""Session issue memory schemas."""

from __future__ import annotations

import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


IssueStatus = Literal["open", "diagnosing", "waiting_user", "submitted", "resolved", "cancelled"]
IssueType = Literal["howto", "fault", "repair", "refund", "complaint", "unknown"]


class EvidenceRef(BaseModel):
    evidence_ref_id: str = Field(default_factory=lambda: f"evref_{uuid4().hex}")
    turn_id: str
    source: str
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class IssueFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: f"fact_{uuid4().hex}")
    kind: str
    value: Any
    status: str = "active"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "runtime"
    evidence_ref_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class IssueThread(BaseModel):
    thread_id: str = Field(default_factory=lambda: f"thread_{uuid4().hex}")
    status: IssueStatus = "open"
    issue_type: IssueType = "unknown"
    product_model: str | None = None
    product_ref: dict[str, Any] = Field(default_factory=dict)
    user_goal: str = ""
    summary: str = ""
    missing_info: list[str] = Field(default_factory=list)
    workflow_snapshot: dict[str, Any] = Field(default_factory=dict)
    linked_ticket_id: str | None = None
    facts: dict[str, IssueFact] = Field(default_factory=dict)
    evidence_refs: dict[str, EvidenceRef] = Field(default_factory=dict)
    last_turn_ids: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class SessionIssueMemory(BaseModel):
    session_id: str
    tenant_id: str | None = None
    user_id: str | None = None
    locale: str = "zh-CN"
    active_thread_id: str | None = None
    active_product: dict[str, Any] = Field(default_factory=dict)
    active_skill: str | None = None
    open_ticket_draft_id: str | None = None
    session_facts: dict[str, IssueFact] = Field(default_factory=dict)
    threads: dict[str, IssueThread] = Field(default_factory=dict)
    entity_index: dict[str, list[str]] = Field(default_factory=dict)
    turn_count: int = 0
    memory_version: int = 0
    flat_state: dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)

    def active_thread(self) -> IssueThread | None:
        if not self.active_thread_id:
            return None
        return self.threads.get(self.active_thread_id)
