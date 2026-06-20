"""v3 记忆系统总入口。"""

from __future__ import annotations

from app.core.config import settings
from app.services.memory.v3.adapters import EvidenceAdapterPipeline
from app.services.memory.v3.episodic_store import InMemoryEpisodicMemoryV3Store
from app.services.memory.v3.llm_judge import LlmMemoryJudge
from app.services.memory.v3.profile_store import InMemoryUserProfileV3Store
from app.services.memory.v3.ranker import MemoryRanker
from app.services.memory.v3.reader import MemoryReader
from app.services.memory.v3.renderer import MemoryRenderer
from app.services.memory.v3.session_store import InMemorySessionMemoryV3Store
from app.services.memory.v3.types import (
    MemoryReadRequest,
    MemoryReadResult,
    MemoryTrace,
    ObservationCandidate,
    TurnEvidencePacket,
)
from app.services.memory.v3.write_gate import WriteGate

_manager_v3: "MemoryManagerV3 | None" = None


class MemoryManagerV3:
    def __init__(
        self,
        *,
        session_store: InMemorySessionMemoryV3Store | None = None,
        profile_store: InMemoryUserProfileV3Store | None = None,
        episodic_store: InMemoryEpisodicMemoryV3Store | None = None,
        adapters: EvidenceAdapterPipeline | None = None,
        llm_judge: LlmMemoryJudge | None = None,
        write_gate: WriteGate | None = None,
        reader: MemoryReader | None = None,
        ranker: MemoryRanker | None = None,
        renderer: MemoryRenderer | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.session_store = session_store or InMemorySessionMemoryV3Store()
        self.profile_store = profile_store or InMemoryUserProfileV3Store()
        self.episodic_store = episodic_store or InMemoryEpisodicMemoryV3Store()
        self.adapters = adapters or EvidenceAdapterPipeline()
        self.llm_judge = llm_judge
        self.write_gate = write_gate or WriteGate()
        self.reader = reader or MemoryReader(
            session_store=self.session_store,
            profile_store=self.profile_store,
            episodic_store=self.episodic_store,
        )
        self.ranker = ranker or MemoryRanker()
        self.renderer = renderer or MemoryRenderer()

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        if not self.enabled:
            return MemoryReadResult(rendered_context="", trace={"disabled": True})
        candidates = self.reader.collect_candidates(request)
        ranked = self.ranker.rank(request, candidates)
        return self.renderer.render(request, ranked)

    def observe_and_write(self, packet: TurnEvidencePacket) -> MemoryTrace:
        trace = MemoryTrace()
        if not (self.enabled and packet.session_id):
            return trace
        raw_evidence = self.adapters.collect(packet)
        candidates: list[ObservationCandidate] = []
        for item in raw_evidence:
            candidates.extend(item.to_candidates())
        trace.write_raw_evidence_count = len(raw_evidence)

        if self._should_use_llm(packet, candidates):
            judge = self.llm_judge or LlmMemoryJudge()
            judgement = judge.judge(
                packet=packet,
                raw_evidence=raw_evidence,
                current_session_memory=self._session_snapshot(packet.session_id),
            )
            trace.llm_judge_used = True
            if judgement is not None and judgement.should_write:
                trace.llm_judge_confidence = judgement.confidence
                candidates.extend(self._verified_llm_candidates(judgement.observations, packet))

        trace.write_candidate_count = len(candidates)
        user_key = packet.user_id or None
        decisions = self.write_gate.decide(candidates, user_key=user_key)
        trace.write_decisions = decisions
        trace.write_discard_reasons = [
            decision.reason for decision in decisions if decision.action == "discard"
        ]
        session_decisions = [d for d in decisions if d.action in {"upsert_session", "delete", "supersede"}]
        profile_decisions = [d for d in decisions if d.action == "upsert_profile"]
        episodic_decisions = [d for d in decisions if d.action == "upsert_episodic"]
        self.session_store.apply_decisions(packet.session_id, session_decisions, turn_id=packet.turn_id)
        if user_key and profile_decisions:
            self.profile_store.apply_decisions(user_key, profile_decisions, turn_id=packet.turn_id)
        if user_key and episodic_decisions:
            self.episodic_store.apply_decisions(user_key, episodic_decisions, turn_id=packet.turn_id)
        trace.write_session_count = len([d for d in session_decisions if d.action == "upsert_session"])
        trace.write_profile_count = len(profile_decisions)
        trace.write_episodic_count = len(episodic_decisions)
        return trace

    @staticmethod
    def _should_use_llm(packet: TurnEvidencePacket, candidates: list[ObservationCandidate]) -> bool:
        if not getattr(settings, "memory_v3_llm_judge_enabled", True):
            return False
        if any(token in packet.question for token in ["这个", "那个", "刚才", "之前", "上次", "还是"]):
            return True
        if any(c.write_intent in {"remember", "forget", "correct"} for c in candidates):
            return True
        if not candidates and len(packet.question) >= 8:
            return True
        return False

    @staticmethod
    def _verified_llm_candidates(
        candidates: list[ObservationCandidate],
        packet: TurnEvidencePacket,
    ) -> list[ObservationCandidate]:
        verified: list[ObservationCandidate] = []
        source_text = "\n".join(
            [packet.question, packet.answer, packet.recent_history, packet.visual_context]
        )
        for candidate in candidates:
            if candidate.kind in {"manual_step", "manual_knowledge"}:
                continue
            # LLM 产出的事实若完全不在输入中出现，只允许作为低风险指代结果进入 session。
            if candidate.value not in source_text and candidate.kind not in {
                "attempted_action",
                "symptom",
                "assistant_commitment",
            }:
                continue
            if candidate.scope_hint == "profile" and candidate.pii_level == "high" and candidate.write_intent != "remember":
                candidate.scope_hint = "session"
            verified.append(candidate)
        return verified

    def _session_snapshot(self, session_id: str) -> dict:
        session = self.session_store.get(session_id)
        active = None
        if session.active_issue_thread_id:
            thread = session.issue_threads.get(session.active_issue_thread_id)
            if thread:
                active = {
                    "product_model": thread.product_model,
                    "fault_codes": thread.fault_codes,
                    "symptoms": thread.symptoms,
                    "attempted_actions": thread.attempted_actions,
                    "missing_slots": thread.missing_slots,
                }
        return {"active_issue": active}


def get_memory_manager_v3() -> MemoryManagerV3:
    global _manager_v3
    if _manager_v3 is None:
        _manager_v3 = MemoryManagerV3(enabled=settings.memory_enabled)
    return _manager_v3


def reset_memory_manager_v3_for_tests() -> None:
    global _manager_v3
    _manager_v3 = None
