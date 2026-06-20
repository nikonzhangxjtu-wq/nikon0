"""v4 记忆系统类型。

v4 只服务同一聊天框内的产品问题追踪：核心是 IssueThread，而不是用户画像。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceRef:
    evidence_ref_id: str
    turn_id: str
    source: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class IssueFact:
    fact_id: str
    kind: str
    value: str
    status: str
    confidence: float
    source: str
    source_priority: int
    evidence_ref_id: str
    created_at: float
    updated_at: float


@dataclass
class IssueFactCandidate:
    kind: str
    value: str
    source: str
    confidence: float
    evidence_text: str
    source_priority: int = 80
    status: str = "active"


@dataclass
class StateChange:
    should_write: bool
    change_type: str
    candidates: list[IssueFactCandidate] = field(default_factory=list)
    reason: str = ""


@dataclass
class IssueThread:
    thread_id: str
    status: str
    issue_type: str
    product_model: str | None = None
    facts: dict[str, IssueFact] = field(default_factory=dict)
    evidence_refs: dict[str, EvidenceRef] = field(default_factory=dict)
    last_turn_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SessionIssueMemory:
    session_id: str
    active_thread_id: str | None = None
    threads: dict[str, IssueThread] = field(default_factory=dict)
    entity_index: dict[str, list[str]] = field(default_factory=dict)
    turn_count: int = 0
    updated_at: float = 0.0


@dataclass
class IssueReadRequest:
    session_id: str
    query: str
    read_mode: str
    query_entities: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""


@dataclass
class IssueSummary:
    rendered_context: str
    thread_ids: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return self.rendered_context


@dataclass
class IssueMemoryTrace:
    should_write: bool = False
    change_type: str = "no_change"
    reason: str = ""
    target_thread_id: str | None = None
    written_fact_count: int = 0
    rejected_fact_count: int = 0
    read_mode: str = "none"
