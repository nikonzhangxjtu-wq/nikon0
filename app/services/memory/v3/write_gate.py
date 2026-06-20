"""记忆写入门控。

这里是 v3 的安全边界：LLM 和 adapter 只能给候选，最终写不写、写哪里由 WriteGate 决定。
"""

from __future__ import annotations

from app.services.memory.v3.types import ObservationCandidate, WriteDecision


class WriteGate:
    def __init__(self, *, min_confidence: float = 0.45) -> None:
        self.min_confidence = min_confidence

    def decide(
        self,
        candidates: list[ObservationCandidate],
        *,
        user_key: str | None,
    ) -> list[WriteDecision]:
        decisions: list[WriteDecision] = []
        for candidate in candidates:
            decisions.append(self._decide_one(candidate, user_key=user_key))
        return decisions

    def _decide_one(
        self,
        candidate: ObservationCandidate,
        *,
        user_key: str | None,
    ) -> WriteDecision:
        if candidate.confidence < self.min_confidence:
            return WriteDecision(
                action="discard",
                reason="候选置信度低，拒绝写入",
                candidate=candidate,
                confidence=candidate.confidence,
            )

        if candidate.write_intent == "forget":
            return WriteDecision(
                action="delete",
                reason="用户明确要求删除或不要保存",
                candidate=candidate,
                confidence=candidate.confidence,
                target_scope=candidate.scope_hint or "session",
            )

        if candidate.write_intent == "correct":
            return WriteDecision(
                action="supersede",
                reason="用户明确纠错，旧事实应被替换",
                candidate=candidate,
                confidence=candidate.confidence,
                target_scope="session",
            )

        if candidate.source == "rag" or candidate.kind in {"manual_step", "manual_knowledge"}:
            # RAG 是产品知识源，不是用户记忆源；只允许用户反馈动作另行写入。
            if candidate.kind != "attempted_action":
                return WriteDecision(
                    action="discard",
                    reason="RAG 手册知识禁止写入用户记忆",
                    candidate=candidate,
                    confidence=candidate.confidence,
                )

        if candidate.scope_hint == "profile":
            if not user_key:
                return WriteDecision(
                    action="upsert_session",
                    reason="缺少 user_key，profile 候选降级为 session",
                    candidate=candidate,
                    confidence=candidate.confidence,
                    target_scope="session",
                )
            if candidate.pii_level == "high" and candidate.write_intent != "remember":
                return WriteDecision(
                    action="upsert_session",
                    reason="高敏 PII 未出现显式记忆意图，降级写入 session",
                    candidate=candidate,
                    confidence=candidate.confidence,
                    target_scope="session",
                )
            if candidate.write_intent == "remember" or candidate.kind in {"user_preference", "preference"}:
                return WriteDecision(
                    action="upsert_profile",
                    reason="用户显式长期偏好或记忆意图，允许写入 profile",
                    candidate=candidate,
                    confidence=candidate.confidence,
                    target_scope="profile",
                )
            return WriteDecision(
                action="upsert_session",
                reason="非稳定长期事实，先写入 session",
                candidate=candidate,
                confidence=candidate.confidence,
                target_scope="session",
            )

        if candidate.scope_hint == "episodic":
            if candidate.kind in {"case_id", "case_status"} or candidate.write_intent == "remember":
                return WriteDecision(
                    action="upsert_episodic",
                    reason="业务事件完成或明确历史事件，写入 episodic",
                    candidate=candidate,
                    confidence=candidate.confidence,
                    target_scope="episodic",
                )
            return WriteDecision(
                action="upsert_session",
                reason="episodic 条件不足，先写入 session",
                candidate=candidate,
                confidence=candidate.confidence,
                target_scope="session",
            )

        return WriteDecision(
            action="upsert_session",
            reason="当前会话事实或状态变化，写入 session",
            candidate=candidate,
            confidence=candidate.confidence,
            target_scope="session",
        )
