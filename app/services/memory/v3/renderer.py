"""把读取候选渲染成 Prompt 可用的 memory_context。"""

from __future__ import annotations

import re

from app.services.memory.v3.types import (
    MemoryReadCandidate,
    MemoryReadRequest,
    MemoryReadResult,
)


class MemoryRenderer:
    def render(
        self,
        request: MemoryReadRequest,
        candidates: list[MemoryReadCandidate],
    ) -> MemoryReadResult:
        selected: list[MemoryReadCandidate] = []
        token_budget = max(60, request.budget_tokens)
        used = 0
        for candidate in candidates:
            text = _redact_pii(candidate.text)
            estimate = _estimate_tokens(text)
            if used + estimate > token_budget and selected:
                continue
            selected.append(
                MemoryReadCandidate(
                    text=text,
                    source_scope=candidate.source_scope,
                    source_id=candidate.source_id,
                    score=candidate.score,
                    reason=candidate.reason,
                    kind=candidate.kind,
                    product_model=candidate.product_model,
                    issue_thread_id=candidate.issue_thread_id,
                )
            )
            used += estimate
        rendered = _render_sections(selected)
        return MemoryReadResult(
            rendered_context=rendered,
            candidates=candidates,
            selected=selected,
            token_estimate=_estimate_tokens(rendered),
            trace={
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "budget_tokens": request.budget_tokens,
            },
        )


def _render_sections(candidates: list[MemoryReadCandidate]) -> str:
    if not candidates:
        return ""
    groups = {
        "session": ("[当前会话记忆]", []),
        "profile": ("[用户稳定信息]", []),
        "episodic": ("[相关历史事件]", []),
    }
    for candidate in candidates:
        title, lines = groups.get(candidate.source_scope, ("[记忆]", []))
        lines.append(f"- {candidate.text}")
        groups[candidate.source_scope] = (title, lines)
    rendered: list[str] = ["[记忆]"]
    for _, (title, lines) in groups.items():
        if lines:
            rendered.append(title)
            rendered.extend(lines)
    return "\n".join(rendered)


def _redact_pii(text: str) -> str:
    return re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text or "")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 2)
