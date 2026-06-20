"""Tool observation context management.

Raw tool results belong in trace/storage. Prompt context should receive compact
observations with status, summaries, data keys, and stable refs for debugging or
refetch, not full payload dumps.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class ToolObservationItem(BaseModel):
    tool: str
    status: str
    summary: str = ""
    data_keys: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str = ""
    raw_result_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolObservationPack(BaseModel):
    items: list[ToolObservationItem] = Field(default_factory=list)

    def render_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


class ToolObservationManager:
    """Convert tool results into prompt-safe observations."""

    def __init__(self, *, max_items: int = 8, summary_char_budget: int = 500) -> None:
        self.max_items = max(1, int(max_items))
        self.summary_char_budget = max(40, int(summary_char_budget))

    def build(self, tool_results: list[dict[str, Any]], *, trace_id: str) -> ToolObservationPack:
        items: list[ToolObservationItem] = []
        start_index = max(0, len(tool_results) - self.max_items)
        for idx, result in enumerate(tool_results[start_index:], start=start_index):
            items.append(self._item(result, trace_id=trace_id, index=idx))
        return ToolObservationPack(items=items)

    def _item(self, result: dict[str, Any], *, trace_id: str, index: int) -> ToolObservationItem:
        service_id = str(result.get("service_id") or "unknown")
        tool_name = str(result.get("tool_name") or "unknown")
        ok = bool(result.get("ok"))
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        summary = self._summary(data, result)
        return ToolObservationItem(
            tool=f"{service_id}.{tool_name}",
            status="success" if ok else "failed",
            summary=summary,
            data_keys=sorted(str(key) for key in data.keys())[:40],
            error_code=result.get("error_code"),
            error_message=str(result.get("error_message") or ""),
            raw_result_ref=f"trace://{trace_id}/tool_results/{index}",
            metadata={
                "provider": result.get("raw", {}).get("provider") if isinstance(result.get("raw"), dict) else None,
                "source_service": result.get("raw", {}).get("source_service") if isinstance(result.get("raw"), dict) else None,
            },
        )

    def _summary(self, data: dict[str, Any], result: dict[str, Any]) -> str:
        candidates = [
            data.get("summary"),
            data.get("reply_text"),
            data.get("context_block"),
            result.get("error_message"),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text[: self.summary_char_budget]
        if data:
            keys = ", ".join(sorted(str(key) for key in data.keys())[:8])
            return f"tool returned data keys: {keys}"[: self.summary_char_budget]
        return ""
