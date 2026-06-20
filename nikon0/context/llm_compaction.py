"""LLM-backed conversation compaction."""

from __future__ import annotations

import json
from typing import Any, Protocol

from nikon0.context.conversation import CompactedConversation, ConversationCompactor


class ChatClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


LLM_CONVERSATION_COMPACTOR_SYSTEM = """你是 nikon0 企业助手的对话压缩器。
你的任务是把旧对话压缩成 issue-local summary。
只输出 JSON，不要输出 Markdown，不要回答用户。

JSON schema:
{
  "issue_summary": "当前 issue 的一句话摘要",
  "summary_lines": ["从原文中抽取或忠实改写的关键事实"]
}

要求：
- 保留用户目标、已确认事实、已执行建议、未解决问题、下一步。
- 不要编造原文没有的信息。
- 不要输出业务承诺。
- 最近对话原文会由 Runtime 另行保留，你只压缩旧历史。
"""


class LlmConversationCompactor:
    """Use LLM to improve issue-local summaries, with deterministic fallback."""

    def __init__(
        self,
        client: ChatClient,
        *,
        deterministic: ConversationCompactor | None = None,
        max_summary_lines: int = 8,
    ) -> None:
        self.client = client
        self.deterministic = deterministic or ConversationCompactor()
        self.max_summary_lines = max(1, int(max_summary_lines))

    async def acompact(self, transcript: str, *, active_issue_summary: str = "") -> CompactedConversation:
        base = self.deterministic.compact(transcript, active_issue_summary=active_issue_summary)
        if not base.compacted:
            return base
        older_text = "\n".join(base.summary_lines)
        messages = [
            {"role": "system", "content": LLM_CONVERSATION_COMPACTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"active_issue_summary: {active_issue_summary or '(empty)'}\n\n"
                    f"旧历史候选内容：\n{older_text}\n\n"
                    "请只输出 JSON。"
                ),
            },
        ]
        try:
            payload = _parse_json(await self.client.complete(messages))
            issue_summary = str(payload.get("issue_summary") or active_issue_summary or base.issue_summary).strip()
            raw_lines = payload.get("summary_lines")
            if not isinstance(raw_lines, list):
                raise ValueError("summary_lines must be a list")
            summary_lines = [str(item).strip() for item in raw_lines if str(item).strip()][: self.max_summary_lines]
            if not summary_lines:
                raise ValueError("summary_lines cannot be empty")
            compacted = CompactedConversation(
                compacted=True,
                issue_summary=issue_summary,
                summary_lines=summary_lines,
                raw_recent_lines=base.raw_recent_lines,
                original_chars=base.original_chars,
            )
            compacted.rendered_chars = len(compacted.render())
            return compacted
        except Exception:
            return base

    def compact(self, transcript: str, *, active_issue_summary: str = "") -> CompactedConversation:
        return self.deterministic.compact(transcript, active_issue_summary=active_issue_summary)


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM compactor output must be a JSON object")
    return data
