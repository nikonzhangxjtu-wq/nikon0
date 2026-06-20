"""记忆候选排序。"""

from __future__ import annotations

from app.services.memory.v3.types import MemoryReadCandidate, MemoryReadRequest


class MemoryRanker:
    def rank(
        self,
        request: MemoryReadRequest,
        candidates: list[MemoryReadCandidate],
    ) -> list[MemoryReadCandidate]:
        ranked: list[MemoryReadCandidate] = []
        for candidate in candidates:
            score = candidate.score + _score_candidate(request, candidate)
            ranked.append(
                MemoryReadCandidate(
                    text=candidate.text,
                    source_scope=candidate.source_scope,
                    source_id=candidate.source_id,
                    score=score,
                    reason=candidate.reason,
                    kind=candidate.kind,
                    product_model=candidate.product_model,
                    issue_thread_id=candidate.issue_thread_id,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked


def _score_candidate(request: MemoryReadRequest, candidate: MemoryReadCandidate) -> float:
    score = 0.0
    query = request.query or ""
    for values in request.entities.values():
        if any(value and value in candidate.text for value in values):
            score += 100
    if candidate.issue_thread_id:
        score += 80
    if candidate.product_model and candidate.product_model in query:
        score += 60
    if candidate.source_scope == "session":
        score += 40
    elif candidate.source_scope == "profile":
        score += 30
    elif candidate.source_scope == "episodic":
        score += 25
    return score
