"""Session 级 v3 记忆存储。"""

from __future__ import annotations

import time
import uuid

from app.services.memory.v3.types import (
    IssueThread,
    MemoryAtom,
    SessionMemoryV3,
    WriteDecision,
)


class InMemorySessionMemoryV3Store:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemoryV3] = {}

    def get(self, session_id: str) -> SessionMemoryV3:
        sid = (session_id or "").strip()
        if sid not in self._sessions:
            self._sessions[sid] = SessionMemoryV3(session_id=sid, updated_at=time.time())
        return self._sessions[sid]

    def apply_decisions(
        self,
        session_id: str,
        decisions: list[WriteDecision],
        *,
        turn_id: str,
    ) -> SessionMemoryV3:
        session = self.get(session_id)
        for decision in decisions:
            if decision.action == "upsert_session" and decision.candidate is not None:
                atom = _atom_from_decision(decision, scope="session", turn_id=turn_id)
                self._upsert_atom(session, atom, turn_id=turn_id)
            elif decision.action in {"delete", "supersede"} and decision.candidate is not None:
                self._mark_matching_atoms(session, decision.candidate.kind, status="superseded")
                if decision.action == "supersede":
                    atom = _atom_from_decision(decision, scope="session", turn_id=turn_id)
                    self._upsert_atom(session, atom, turn_id=turn_id)
        session.turn_count += 1
        if turn_id not in session.recent_turn_ids:
            session.recent_turn_ids.append(turn_id)
            session.recent_turn_ids = session.recent_turn_ids[-6:]
        session.updated_at = time.time()
        return session

    def _upsert_atom(self, session: SessionMemoryV3, atom: MemoryAtom, *, turn_id: str) -> None:
        session.atoms[atom.atom_id] = atom
        key = f"{atom.kind}:{atom.value}"
        session.entity_index.setdefault(key, [])
        if atom.atom_id not in session.entity_index[key]:
            session.entity_index[key].append(atom.atom_id)
        thread = self._ensure_issue_thread(session, atom, turn_id=turn_id)
        atom.issue_thread_id = thread.thread_id
        if atom.atom_id not in thread.source_atom_ids:
            thread.source_atom_ids.append(atom.atom_id)
        _merge_atom_into_thread(thread, atom)

    @staticmethod
    def _ensure_issue_thread(
        session: SessionMemoryV3,
        atom: MemoryAtom,
        *,
        turn_id: str,
    ) -> IssueThread:
        thread_id = session.active_issue_thread_id
        if not thread_id:
            thread_id = f"issue_{uuid.uuid4().hex[:12]}"
            session.issue_threads[thread_id] = IssueThread(
                thread_id=thread_id,
                status="open",
                category=_category_for_atom(atom),
                created_at=time.time(),
                updated_at=time.time(),
            )
            session.active_issue_thread_id = thread_id
        thread = session.issue_threads[thread_id]
        if turn_id not in thread.last_turn_ids:
            thread.last_turn_ids.append(turn_id)
            thread.last_turn_ids = thread.last_turn_ids[-6:]
        thread.updated_at = time.time()
        return thread

    @staticmethod
    def _mark_matching_atoms(session: SessionMemoryV3, kind: str, *, status: str) -> None:
        for atom in session.atoms.values():
            if atom.kind == kind and atom.status == "active":
                atom.status = status
                atom.updated_at = time.time()


def _atom_from_decision(decision: WriteDecision, *, scope: str, turn_id: str) -> MemoryAtom:
    candidate = decision.candidate
    assert candidate is not None
    now = time.time()
    return MemoryAtom(
        atom_id=f"atom_{uuid.uuid4().hex[:16]}",
        scope=scope,
        kind=candidate.kind,
        value=candidate.value,
        confidence=candidate.confidence,
        source=candidate.source,
        source_turn_id=turn_id,
        source_priority=candidate.source_priority,
        product_model=candidate.product_model,
        pii_level=candidate.pii_level,
        created_at=now,
        updated_at=now,
        evidence_text=candidate.evidence_text,
    )


def _merge_atom_into_thread(thread: IssueThread, atom: MemoryAtom) -> None:
    if atom.kind == "product_model":
        thread.product_model = atom.value
    elif atom.kind == "order_id":
        thread.order_id = atom.value
    elif atom.kind == "case_id":
        thread.case_id = atom.value
    elif atom.kind == "fault_code":
        _append_unique(thread.fault_codes, atom.value)
    elif atom.kind == "symptom":
        _append_unique(thread.symptoms, atom.value)
    elif atom.kind == "user_goal":
        _append_unique(thread.user_goals, atom.value)
        if atom.value in {"报修", "退款", "投诉"}:
            thread.category = {"报修": "repair", "退款": "refund", "投诉": "complaint"}[atom.value]
    elif atom.kind == "attempted_action":
        _append_unique(thread.attempted_actions, atom.value)
    elif atom.kind == "missing_slot":
        _append_unique(thread.missing_slots, atom.value)
    elif atom.kind == "assistant_commitment":
        _append_unique(thread.assistant_commitments, atom.value)
    elif atom.kind == "case_status":
        if atom.value in {"submitted", "resolved", "cancelled", "pending"}:
            thread.status = atom.value


def _category_for_atom(atom: MemoryAtom) -> str:
    if atom.kind in {"fault_code", "symptom", "attempted_action"}:
        return "repair"
    if atom.kind == "user_goal" and atom.value == "退款":
        return "refund"
    if atom.kind == "user_goal" and atom.value == "投诉":
        return "complaint"
    return "unknown"


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
