"""IssueThread 归属判定。"""

from __future__ import annotations

import re

from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.types import IssueFactCandidate, SessionIssueMemory, StateChange


class IssueThreadResolver:
    def resolve(
        self,
        memory: SessionIssueMemory,
        change: StateChange,
        packet: TurnEvidencePacket,
    ) -> tuple[str | None, bool]:
        if not change.should_write:
            return None, False
        product = _first_value(change.candidates, "product_model")
        case_id = _first_value(change.candidates, "case_id")
        if case_id:
            for thread in memory.threads.values():
                if _thread_has_fact(thread, "case_id", case_id):
                    return thread.thread_id, False
        if product:
            if re.search(r"另一个|另外一台|另一台|其他", packet.question or ""):
                return None, True
            for thread in memory.threads.values():
                if thread.product_model == product and thread.status not in {"resolved", "cancelled"}:
                    return thread.thread_id, False
            if memory.active_thread_id:
                active = memory.threads.get(memory.active_thread_id)
                if active and active.product_model is None:
                    return active.thread_id, False
            return None, True
        if re.search(r"这个|那个|刚才|还是|继续|下一步|我试过|还不行", packet.question or ""):
            return memory.active_thread_id, False
        return memory.active_thread_id, memory.active_thread_id is None


def _first_value(candidates: list[IssueFactCandidate], kind: str) -> str | None:
    for candidate in candidates:
        if candidate.kind == kind:
            return candidate.value
    return None


def _thread_has_fact(thread, kind: str, value: str) -> bool:
    return any(fact.kind == kind and fact.value == value and fact.status == "active" for fact in thread.facts.values())
