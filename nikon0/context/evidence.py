"""Evidence context management.

This module intentionally does not summarize RAG chunks by default. It selects,
deduplicates, trims raw excerpts, and preserves source metadata so final answer
generation can stay grounded in inspectable text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import Evidence


class PromptEvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    source: dict[str, Any] = Field(default_factory=dict)
    raw_excerpt: str
    confidence: float
    applicability: dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    query: str
    items: list[PromptEvidenceItem] = Field(default_factory=list)
    usage: dict[str, list[str]] = Field(default_factory=dict)

    def render_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


class EvidenceContextManager:
    """Prepare prompt-safe evidence without lossy summarization."""

    def __init__(self, *, max_items: int = 5, excerpt_char_budget: int = 900, span_selector: Any | None = None) -> None:
        self.max_items = max(1, int(max_items))
        self.excerpt_char_budget = max(80, int(excerpt_char_budget))
        self.span_selector = span_selector

    def build(self, *, query: str, evidence: list[Evidence]) -> EvidencePack:
        retrieved_ids = [item.evidence_id for item in evidence]
        included: list[PromptEvidenceItem] = []
        deduplicated_ids: list[str] = []
        seen_keys: set[str] = set()
        for item in sorted(evidence, key=lambda ev: ev.confidence, reverse=True):
            key = self._dedupe_key(item)
            if key in seen_keys:
                deduplicated_ids.append(item.evidence_id)
                continue
            seen_keys.add(key)
            included.append(self._prompt_item(query, item))
            if len(included) >= self.max_items:
                break
        return EvidencePack(
            query=query,
            items=included,
            usage={
                "retrieved_evidence_ids": retrieved_ids,
                "included_evidence_ids": [item.evidence_id for item in included],
                "deduplicated_evidence_ids": deduplicated_ids,
            },
        )

    async def abuild(self, *, query: str, evidence: list[Evidence]) -> EvidencePack:
        if self.span_selector is None or not hasattr(self.span_selector, "select_span"):
            return self.build(query=query, evidence=evidence)
        retrieved_ids = [item.evidence_id for item in evidence]
        included: list[PromptEvidenceItem] = []
        deduplicated_ids: list[str] = []
        seen_keys: set[str] = set()
        for item in sorted(evidence, key=lambda ev: ev.confidence, reverse=True):
            key = self._dedupe_key(item)
            if key in seen_keys:
                deduplicated_ids.append(item.evidence_id)
                continue
            seen_keys.add(key)
            span = await self.span_selector.select_span(query=query, text=item.text)
            included.append(self._prompt_item(query, item, raw_excerpt=span.text))
            if len(included) >= self.max_items:
                break
        return EvidencePack(
            query=query,
            items=included,
            usage={
                "retrieved_evidence_ids": retrieved_ids,
                "included_evidence_ids": [item.evidence_id for item in included],
                "deduplicated_evidence_ids": deduplicated_ids,
            },
        )

    def _prompt_item(self, query: str, evidence: Evidence, raw_excerpt: str | None = None) -> PromptEvidenceItem:
        return PromptEvidenceItem(
            evidence_id=evidence.evidence_id,
            source_type=evidence.source,
            source=self._source_metadata(evidence),
            raw_excerpt=raw_excerpt if raw_excerpt is not None else self._raw_excerpt(query, evidence.text),
            confidence=evidence.confidence,
            applicability=self._applicability(evidence),
        )

    @staticmethod
    def _source_metadata(evidence: Evidence) -> dict[str, Any]:
        payload = evidence.payload
        source = {
            "manual_name": payload.get("manual_name"),
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "product_id": payload.get("product_id"),
            "knowledge_version": payload.get("knowledge_version"),
            "section_path": payload.get("section_path"),
        }
        return {key: value for key, value in source.items() if value not in (None, "", [])}

    @staticmethod
    def _applicability(evidence: Evidence) -> dict[str, Any]:
        payload = evidence.payload
        applicability = {
            "manual_name": payload.get("manual_name"),
            "product_id": payload.get("product_id"),
            "conditions": payload.get("conditions"),
            "limitations": payload.get("limitations"),
        }
        return {key: value for key, value in applicability.items() if value not in (None, "", [])}

    def _raw_excerpt(self, query: str, text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= self.excerpt_char_budget:
            return clean
        query_terms = [term for term in _terms(query) if term in clean]
        if not query_terms:
            return clean[: self.excerpt_char_budget]
        first_hit = min(clean.find(term) for term in query_terms if clean.find(term) >= 0)
        half = self.excerpt_char_budget // 2
        start = max(0, first_hit - half)
        end = min(len(clean), start + self.excerpt_char_budget)
        if end - start < self.excerpt_char_budget:
            start = max(0, end - self.excerpt_char_budget)
        return clean[start:end]

    @staticmethod
    def _dedupe_key(evidence: Evidence) -> str:
        payload = evidence.payload
        manual = str(payload.get("manual_name") or "").strip()
        page = str(payload.get("page") or "").strip()
        normalized_text = re.sub(r"\s+", "", evidence.text)[:160]
        return f"{manual}:{page}:{normalized_text}"


def _terms(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)
