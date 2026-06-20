"""Conversation compaction for context packs.

Phase 1 is deterministic and conservative: keep recent turns verbatim and
extract a structured issue-local summary from older lines. LLM compaction can
replace the summarizer later behind the same interface.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompactedConversation(BaseModel):
    compacted: bool
    issue_summary: str = ""
    summary_lines: list[str] = Field(default_factory=list)
    raw_recent_lines: list[str] = Field(default_factory=list)
    original_chars: int = 0
    rendered_chars: int = 0

    def render(self) -> str:
        if not self.compacted:
            return "\n".join(self.raw_recent_lines)
        lines = ["[Conversation Summary]"]
        if self.issue_summary:
            lines.append(f"issue_summary: {self.issue_summary}")
        if self.summary_lines:
            lines.append("older_context:")
            for line in self.summary_lines:
                lines.append(f"- {line}")
        lines.append("[Recent Conversation]")
        lines.extend(self.raw_recent_lines)
        return "\n".join(lines)


class ConversationCompactor:
    """Conservative issue-local conversation compactor."""

    def __init__(
        self,
        *,
        max_raw_chars: int = 1800,
        recent_line_count: int = 8,
        max_summary_lines: int = 8,
    ) -> None:
        self.max_raw_chars = max(40, int(max_raw_chars))
        self.recent_line_count = max(2, int(recent_line_count))
        self.max_summary_lines = max(1, int(max_summary_lines))

    def compact(self, transcript: str, *, active_issue_summary: str = "") -> CompactedConversation:
        clean = transcript.strip()
        if not clean:
            return CompactedConversation(compacted=False)
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if len(clean) <= self.max_raw_chars:
            return CompactedConversation(
                compacted=False,
                raw_recent_lines=lines,
                original_chars=len(clean),
                rendered_chars=len(clean),
            )
        recent = lines[-self.recent_line_count :]
        older = lines[: -self.recent_line_count]
        summary_lines = self._summary_lines(older)
        compacted = CompactedConversation(
            compacted=True,
            issue_summary=active_issue_summary,
            summary_lines=summary_lines,
            raw_recent_lines=recent,
            original_chars=len(clean),
        )
        compacted.rendered_chars = len(compacted.render())
        return compacted

    def _summary_lines(self, lines: list[str]) -> list[str]:
        selected: list[str] = []
        for line in lines:
            if self._is_salient(line):
                selected.append(line)
            if len(selected) >= self.max_summary_lines:
                break
        if selected:
            return selected
        return lines[-self.max_summary_lines :]

    @staticmethod
    def _is_salient(line: str) -> bool:
        signals = (
            "故障",
            "显示",
            "报修",
            "退款",
            "投诉",
            "型号",
            "电话",
            "订单",
            "已收集",
            "还需要",
            "需要",
            "堵塞",
            "断电",
            "清洁",
            "检查",
            "不能",
            "无法",
            "error",
            "failed",
        )
        return any(signal in line for signal in signals)
