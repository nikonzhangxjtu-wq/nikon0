"""v3 记忆系统的数据契约。

这些 dataclass 是模块之间的边界：adapter 只产证据，LLM 只产候选，
WriteGate 才能把候选变成真正的写入动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnEvidencePacket:
    session_id: str
    user_id: str | None
    turn_id: str
    timestamp: float
    question: str
    answer: str
    route_domain_hint: str
    route_needs_rag: bool
    branch_name: str
    recent_history: str = ""
    memory_context_used: str = ""
    visual_context: str = ""
    rag_context: str = ""
    branch_result: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ObservationCandidate:
    kind: str
    value: str
    source: str
    confidence: float
    evidence_text: str
    scope_hint: str | None = None
    write_intent: str = "observe"
    product_model: str | None = None
    issue_thread_id: str | None = None
    pii_level: str = "none"
    source_priority: int = 50


@dataclass
class RawEvidence:
    source: str
    evidence_type: str = "explicit_fact"
    text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source_priority: int = 50

    def to_candidates(self) -> list[ObservationCandidate]:
        candidates: list[ObservationCandidate] = []
        for key, value in self.structured.items():
            if value is None or value == "":
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                item_text = str(item).strip()
                if not item_text:
                    continue
                candidates.append(
                    ObservationCandidate(
                        kind=str(key),
                        value=item_text,
                        source=self.source,
                        confidence=self.confidence,
                        evidence_text=self.text,
                        scope_hint=_scope_hint_for(key, self.evidence_type),
                        write_intent=_intent_for(self.evidence_type),
                        pii_level=_pii_level_for(key),
                        source_priority=self.source_priority,
                    )
                )
        return candidates


@dataclass
class LlmMemoryJudgement:
    should_write: bool
    write_intent: str
    target_scope: str
    confidence: float
    reason: str
    observations: list[ObservationCandidate] = field(default_factory=list)
    resolved_references: dict[str, str] = field(default_factory=dict)


@dataclass
class WriteDecision:
    action: str
    reason: str
    candidate: ObservationCandidate | None
    confidence: float
    target_scope: str | None = None
    target_issue_thread_id: str | None = None
    supersede_atom_id: str | None = None


@dataclass
class MemoryAtom:
    atom_id: str
    scope: str
    kind: str
    value: str
    confidence: float
    source: str
    source_turn_id: str | None
    source_priority: int
    product_model: str | None = None
    issue_thread_id: str | None = None
    pii_level: str = "none"
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float | None = None
    evidence_text: str = ""


@dataclass
class IssueThread:
    thread_id: str
    status: str
    category: str
    product_model: str | None = None
    order_id: str | None = None
    case_id: str | None = None
    symptoms: list[str] = field(default_factory=list)
    fault_codes: list[str] = field(default_factory=list)
    user_goals: list[str] = field(default_factory=list)
    attempted_actions: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    assistant_commitments: list[str] = field(default_factory=list)
    source_atom_ids: list[str] = field(default_factory=list)
    last_turn_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SessionMemoryV3:
    session_id: str
    atoms: dict[str, MemoryAtom] = field(default_factory=dict)
    issue_threads: dict[str, IssueThread] = field(default_factory=dict)
    entity_index: dict[str, list[str]] = field(default_factory=dict)
    active_issue_thread_id: str | None = None
    recent_turn_ids: list[str] = field(default_factory=list)
    turn_count: int = 0
    updated_at: float = 0.0


@dataclass
class UserProfileV3:
    user_key: str
    stable_atoms: dict[str, MemoryAtom] = field(default_factory=dict)
    preferred_contact_phone_atom_id: str | None = None
    default_product_model_atom_id: str | None = None
    updated_at: float = 0.0


@dataclass
class EpisodicEvent:
    event_id: str
    user_key: str
    event_type: str
    title: str
    summary: str
    product_model: str | None = None
    case_id: str | None = None
    issue_thread_id: str | None = None
    atom_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    expires_at: float | None = None


@dataclass
class MemoryReadRequest:
    session_id: str | None
    user_id: str | None
    query: str
    intents: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    include_session: bool = True
    include_profile: bool = True
    include_episodic: bool = True
    budget_tokens: int = 600
    reason: str = ""


@dataclass
class MemoryReadCandidate:
    text: str
    source_scope: str
    source_id: str
    score: float
    reason: str
    kind: str
    product_model: str | None = None
    issue_thread_id: str | None = None


@dataclass
class MemoryReadResult:
    rendered_context: str
    candidates: list[MemoryReadCandidate] = field(default_factory=list)
    selected: list[MemoryReadCandidate] = field(default_factory=list)
    token_estimate: int = 0
    trace: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return self.rendered_context


@dataclass
class MemoryTrace:
    read_enabled: bool = False
    read_intents: list[str] = field(default_factory=list)
    read_candidate_count: int = 0
    read_selected_count: int = 0
    read_tokens: int = 0
    write_raw_evidence_count: int = 0
    write_candidate_count: int = 0
    llm_judge_used: bool = False
    llm_judge_confidence: float = 0.0
    write_decisions: list[WriteDecision] = field(default_factory=list)
    write_discard_reasons: list[str] = field(default_factory=list)
    write_session_count: int = 0
    write_profile_count: int = 0
    write_episodic_count: int = 0
    pii_redaction_count: int = 0
    conflict_count: int = 0


def _scope_hint_for(kind: str, evidence_type: str) -> str:
    if evidence_type in {"preference", "remember"}:
        return "profile"
    if evidence_type in {"event", "tool_fact"} and kind in {"case_id", "case_status"}:
        return "episodic"
    return "session"


def _intent_for(evidence_type: str) -> str:
    if evidence_type in {"remember", "preference"}:
        return "remember"
    if evidence_type == "forget":
        return "forget"
    if evidence_type == "correction":
        return "correct"
    return "observe"


def _pii_level_for(kind: str) -> str:
    if kind in {"phone", "address"}:
        return "high"
    if kind in {"order_id", "case_id"}:
        return "low"
    return "none"
