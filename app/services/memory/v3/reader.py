"""从 v3 stores 读取候选记忆。"""

from __future__ import annotations

from app.services.memory.v3.episodic_store import InMemoryEpisodicMemoryV3Store
from app.services.memory.v3.profile_store import InMemoryUserProfileV3Store
from app.services.memory.v3.session_store import InMemorySessionMemoryV3Store
from app.services.memory.v3.types import MemoryReadCandidate, MemoryReadRequest


class MemoryReader:
    def __init__(
        self,
        *,
        session_store: InMemorySessionMemoryV3Store,
        profile_store: InMemoryUserProfileV3Store,
        episodic_store: InMemoryEpisodicMemoryV3Store,
    ) -> None:
        self.session_store = session_store
        self.profile_store = profile_store
        self.episodic_store = episodic_store

    def collect_candidates(self, request: MemoryReadRequest) -> list[MemoryReadCandidate]:
        candidates: list[MemoryReadCandidate] = []
        if request.include_session and request.session_id:
            session = self.session_store.get(request.session_id)
            for atom in session.atoms.values():
                if atom.status not in {"active", "conflict"}:
                    continue
                candidates.append(
                    MemoryReadCandidate(
                        text=f"{_label(atom.kind)}: {atom.value}",
                        source_scope="session",
                        source_id=atom.atom_id,
                        score=atom.confidence * 10,
                        reason="session_atom",
                        kind=atom.kind,
                        product_model=atom.product_model,
                        issue_thread_id=atom.issue_thread_id,
                    )
                )
            for thread in session.issue_threads.values():
                if thread.status in {"cancelled", "resolved"}:
                    continue
                summary = _thread_summary(thread)
                if summary:
                    candidates.append(
                        MemoryReadCandidate(
                            text=summary,
                            source_scope="session",
                            source_id=thread.thread_id,
                            score=20,
                            reason="active_issue_thread",
                            kind="issue_thread",
                            product_model=thread.product_model,
                            issue_thread_id=thread.thread_id,
                        )
                    )
        if request.include_profile and request.user_id:
            profile = self.profile_store.get(request.user_id)
            if profile:
                for atom in profile.stable_atoms.values():
                    if atom.status == "active":
                        candidates.append(
                            MemoryReadCandidate(
                                text=f"{_label(atom.kind)}: {atom.value}",
                                source_scope="profile",
                                source_id=atom.atom_id,
                                score=atom.confidence * 10,
                                reason="profile_atom",
                                kind=atom.kind,
                            )
                        )
        if request.include_episodic and request.user_id:
            for event in self.episodic_store.search(request.user_id, request.query):
                candidates.append(
                    MemoryReadCandidate(
                        text=event.summary,
                        source_scope="episodic",
                        source_id=event.event_id,
                        score=8,
                        reason="episodic_event",
                        kind=event.event_type,
                        product_model=event.product_model,
                        issue_thread_id=event.issue_thread_id,
                    )
                )
        return candidates


def _label(kind: str) -> str:
    return {
        "product_model": "产品",
        "fault_code": "故障码",
        "attempted_action": "已尝试",
        "symptom": "现象",
        "phone": "联系电话",
        "user_goal": "用户诉求",
        "assistant_commitment": "助手建议/承诺",
        "missing_slot": "缺失信息",
        "case_id": "工单号",
        "case_status": "工单状态",
    }.get(kind, kind)


def _thread_summary(thread) -> str:
    parts = []
    if thread.product_model:
        parts.append(f"产品 {thread.product_model}")
    if thread.fault_codes:
        parts.append(f"故障码 {','.join(thread.fault_codes)}")
    if thread.symptoms:
        parts.append(f"现象 {','.join(thread.symptoms)}")
    if thread.attempted_actions:
        parts.append(f"已尝试 {','.join(thread.attempted_actions)}")
    if thread.missing_slots:
        parts.append(f"仍缺 {','.join(thread.missing_slots)}")
    return "；".join(parts)
