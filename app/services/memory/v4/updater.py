"""IssueThread 状态更新器。"""

from __future__ import annotations

import time
import uuid

from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.types import (
    EvidenceRef,
    IssueFact,
    IssueFactCandidate,
    IssueMemoryTrace,
    IssueThread,
    SessionIssueMemory,
    StateChange,
)


class IssueThreadUpdater:
    def apply(
        self,
        memory: SessionIssueMemory,
        change: StateChange,
        *,
        packet: TurnEvidencePacket,
        target_thread_id: str | None,
        create_new: bool,
    ) -> IssueMemoryTrace:
        if not change.should_write:
            return IssueMemoryTrace(False, change.change_type, change.reason)
        if not change.candidates:
            return IssueMemoryTrace(False, change.change_type, "无可写候选")
        if _has_unbacked_llm_candidate(change.candidates, packet):
            return IssueMemoryTrace(False, change.change_type, "LLM 候选缺少可追溯证据")

        thread = self._get_or_create_thread(
            memory,
            packet=packet,
            target_thread_id=target_thread_id,
            create_new=create_new,
            candidates=change.candidates,
        )
        evidence = EvidenceRef(
            evidence_ref_id=f"ev_{uuid.uuid4().hex[:12]}",
            turn_id=packet.turn_id,
            source=change.candidates[0].source if change.candidates else "user",
            text=_evidence_text(packet, change),
            payload={
                "branch_name": packet.branch_name,
                "route_domain_hint": packet.route_domain_hint,
                "branch_result": packet.branch_result or {},
            },
            created_at=time.time(),
        )
        thread.evidence_refs[evidence.evidence_ref_id] = evidence
        written = 0
        rejected = 0
        for candidate in change.candidates:
            if candidate.status == "rejected":
                rejected += self._reject_matching(thread, candidate, evidence)
                continue
            if change.change_type == "correction":
                self._supersede_kind(thread, candidate.kind)
            fact = self._upsert_fact(thread, candidate, evidence)
            self._merge_fact_to_thread(thread, fact)
            written += 1
        if change.change_type == "resolution":
            thread.status = "resolved"
        elif any(c.kind == "case_status" and c.value == "submitted" for c in change.candidates):
            thread.status = "submitted"
        elif thread.status == "open":
            thread.status = "diagnosing"
        if packet.turn_id not in thread.last_turn_ids:
            thread.last_turn_ids.append(packet.turn_id)
            thread.last_turn_ids = thread.last_turn_ids[-8:]
        thread.updated_at = time.time()
        memory.active_thread_id = thread.thread_id if thread.status not in {"resolved", "cancelled"} else memory.active_thread_id
        memory.turn_count += 1
        memory.updated_at = time.time()
        self._rebuild_entity_index(memory)
        return IssueMemoryTrace(
            should_write=True,
            change_type=change.change_type,
            reason=change.reason,
            target_thread_id=thread.thread_id,
            written_fact_count=written,
            rejected_fact_count=rejected,
        )

    def _get_or_create_thread(
        self,
        memory: SessionIssueMemory,
        *,
        packet: TurnEvidencePacket,
        target_thread_id: str | None,
        create_new: bool,
        candidates: list[IssueFactCandidate],
    ) -> IssueThread:
        if target_thread_id and target_thread_id in memory.threads and not create_new:
            return memory.threads[target_thread_id]
        thread_id = f"issue_{uuid.uuid4().hex[:12]}"
        now = time.time()
        thread = IssueThread(
            thread_id=thread_id,
            status="open",
            issue_type=_issue_type(candidates, packet),
            product_model=_first_value(candidates, "product_model"),
            created_at=now,
            updated_at=now,
        )
        memory.threads[thread_id] = thread
        memory.active_thread_id = thread_id
        return thread

    @staticmethod
    def _upsert_fact(
        thread: IssueThread,
        candidate: IssueFactCandidate,
        evidence: EvidenceRef,
    ) -> IssueFact:
        existing = _find_active_fact(thread, candidate.kind, candidate.value)
        now = time.time()
        if existing:
            existing.updated_at = now
            return existing
        fact = IssueFact(
            fact_id=f"fact_{uuid.uuid4().hex[:12]}",
            kind=candidate.kind,
            value=candidate.value,
            status="active",
            confidence=candidate.confidence,
            source=candidate.source,
            source_priority=candidate.source_priority,
            evidence_ref_id=evidence.evidence_ref_id,
            created_at=now,
            updated_at=now,
        )
        thread.facts[fact.fact_id] = fact
        return fact

    @staticmethod
    def _merge_fact_to_thread(thread: IssueThread, fact: IssueFact) -> None:
        if fact.kind == "product_model":
            thread.product_model = fact.value

    @staticmethod
    def _supersede_kind(thread: IssueThread, kind: str) -> None:
        for fact in thread.facts.values():
            if fact.kind == kind and fact.status == "active":
                fact.status = "superseded"
                fact.updated_at = time.time()

    @staticmethod
    def _reject_matching(thread: IssueThread, candidate: IssueFactCandidate, evidence: EvidenceRef) -> int:
        count = 0
        for fact in thread.facts.values():
            if fact.kind == candidate.kind and fact.value == candidate.value and fact.status == "active":
                fact.status = "rejected"
                fact.updated_at = time.time()
                count += 1
        if count == 0:
            fact = IssueFact(
                fact_id=f"fact_{uuid.uuid4().hex[:12]}",
                kind=candidate.kind,
                value=candidate.value,
                status="rejected",
                confidence=candidate.confidence,
                source=candidate.source,
                source_priority=candidate.source_priority,
                evidence_ref_id=evidence.evidence_ref_id,
                created_at=time.time(),
                updated_at=time.time(),
            )
            thread.facts[fact.fact_id] = fact
            count = 1
        return count

    @staticmethod
    def _rebuild_entity_index(memory: SessionIssueMemory) -> None:
        index: dict[str, list[str]] = {}
        for thread in memory.threads.values():
            for fact in thread.facts.values():
                if fact.status != "active":
                    continue
                key = f"{fact.kind}:{fact.value}"
                index.setdefault(key, [])
                if thread.thread_id not in index[key]:
                    index[key].append(thread.thread_id)
        memory.entity_index = index


def _has_unbacked_llm_candidate(candidates: list[IssueFactCandidate], packet: TurnEvidencePacket) -> bool:
    source_text = "\n".join([packet.question, packet.answer, packet.recent_history, packet.visual_context])
    for candidate in candidates:
        if candidate.source == "llm" and (not candidate.evidence_text or candidate.value not in source_text):
            return True
    return False


def _evidence_text(packet: TurnEvidencePacket, change: StateChange) -> str:
    for candidate in change.candidates:
        if candidate.evidence_text:
            return candidate.evidence_text
    return packet.question or packet.answer


def _find_active_fact(thread: IssueThread, kind: str, value: str) -> IssueFact | None:
    for fact in thread.facts.values():
        if fact.kind == kind and fact.value == value and fact.status == "active":
            return fact
    return None


def _first_value(candidates: list[IssueFactCandidate], kind: str) -> str | None:
    for candidate in candidates:
        if candidate.kind == kind and candidate.status == "active":
            return candidate.value
    return None


def _issue_type(candidates: list[IssueFactCandidate], packet: TurnEvidencePacket) -> str:
    if any(c.kind in {"fault_code", "symptom", "attempted_action"} for c in candidates):
        return "fault"
    if any(c.kind == "user_goal" and c.value == "退款" for c in candidates):
        return "refund"
    if any(c.kind == "user_goal" and c.value == "投诉" for c in candidates):
        return "complaint"
    if packet.branch_name == "rag_manual":
        return "howto"
    return "unknown"
