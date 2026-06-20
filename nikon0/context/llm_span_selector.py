"""LLM-assisted evidence span selection.

The LLM may choose character offsets, but the returned prompt text is always a
slice of the original evidence. It never replaces evidence with a free summary.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel


class ChatClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class EvidenceSpan(BaseModel):
    text: str
    start: int
    end: int
    source: str = "deterministic"
    reason: str = ""
    summary: str = ""


LLM_SPAN_SELECTOR_SYSTEM = """你是 nikon0 企业助手的证据原文片段选择器。
你只返回原文 span 的字符下标，不总结、不改写、不补充。
只输出 JSON，不要输出 Markdown。

JSON schema:
{
  "start": 0,
  "end": 100,
  "reason": "为什么这个原文片段相关"
}

要求：
- start/end 必须是原文 text 的字符下标。
- 返回片段必须能直接支持回答。
- 不要输出 summary 字段。
- 如果无法判断，选择最相关的短原文片段。
"""


class LlmEvidenceSpanSelector:
    def __init__(self, client: ChatClient, *, max_span_chars: int = 900) -> None:
        self.client = client
        self.max_span_chars = max(40, int(max_span_chars))

    async def select_span(self, *, query: str, text: str) -> EvidenceSpan:
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= self.max_span_chars:
            return EvidenceSpan(text=clean, start=0, end=len(clean), source="deterministic")
        messages = [
            {"role": "system", "content": LLM_SPAN_SELECTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"query:\n{query}\n\n"
                    f"text:\n{clean}\n\n"
                    "请只返回原文 span 的 start/end JSON。"
                ),
            },
        ]
        try:
            payload = _parse_json(await self.client.complete(messages))
            start = int(payload.get("start"))
            end = int(payload.get("end"))
            if start < 0 or end <= start or start >= len(clean):
                raise ValueError("invalid span offsets")
            end = min(end, len(clean), start + self.max_span_chars)
            return EvidenceSpan(
                text=clean[start:end],
                start=start,
                end=end,
                source="llm",
                reason=str(payload.get("reason") or ""),
                summary="",
            )
        except Exception:
            return self._deterministic_span(query=query, text=clean)

    def _deterministic_span(self, *, query: str, text: str) -> EvidenceSpan:
        terms = [term for term in _terms(query) if term in text]
        if not terms:
            return EvidenceSpan(text=text[: self.max_span_chars], start=0, end=min(len(text), self.max_span_chars))
        first_hit = min(text.find(term) for term in terms if text.find(term) >= 0)
        half = self.max_span_chars // 2
        start = max(0, first_hit - half)
        end = min(len(text), start + self.max_span_chars)
        if end - start < self.max_span_chars:
            start = max(0, end - self.max_span_chars)
        return EvidenceSpan(text=text[start:end], start=start, end=end, source="deterministic")


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
        raise ValueError("span selector output must be a JSON object")
    return data


def _terms(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)
