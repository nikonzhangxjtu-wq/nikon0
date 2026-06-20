"""LLM proposal layer for governed memory writes."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import StateUpdate
from nikon0.memory.governance.types import StateUpdateCandidate


class MemoryWriteModelClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class MemoryWriteRequest(BaseModel):
    source_agent: Literal["support", "service"]
    execution_stage: Literal["diagnosis", "service_workflow"]
    message: str
    handoff: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    target_thread_id: str | None = None


class MemoryWriteAgentResult(BaseModel):
    valid: bool
    candidates: list[StateUpdateCandidate] = Field(default_factory=list)
    failure_reason: str = ""
    raw_response: str = ""


class MemoryWriteAgent:
    """Proposes memory candidates; Runtime and MemoryWriteGate remain authoritative."""

    def __init__(self, client: MemoryWriteModelClient, *, min_confidence: float = 0.70) -> None:
        self.client = client
        self.min_confidence = min(1.0, max(0.0, min_confidence))

    async def propose(self, request: MemoryWriteRequest) -> MemoryWriteAgentResult:
        raw = ""
        try:
            raw = await self.client.complete(self._messages(request))
            payload = _parse_json(raw)
            raw_candidates = payload.get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError("candidates must be a list")
            candidates = [self._candidate(item, request) for item in raw_candidates]
            if not candidates:
                raise ValueError("no usable memory candidates")
            return MemoryWriteAgentResult(valid=True, candidates=candidates, raw_response=raw)
        except Exception as exc:  # noqa: BLE001
            return MemoryWriteAgentResult(
                valid=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
                raw_response=raw,
            )

    def _candidate(self, raw: Any, request: MemoryWriteRequest) -> StateUpdateCandidate:
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        update_key = str(raw.get("update_key") or "").strip()
        fields = raw.get("fields")
        confidence = _confidence(raw.get("confidence"))
        evidence_ids = [str(item) for item in raw.get("evidence_ids", []) if str(item) in set(request.evidence_ids)]
        if update_key not in {"product_support", "case_intake"}:
            raise ValueError("candidate update_key is not allowed")
        if not isinstance(fields, dict) or not fields:
            raise ValueError("candidate fields must be a non-empty object")
        if confidence < self.min_confidence:
            raise ValueError("candidate confidence below threshold")
        update = StateUpdate(
            key=update_key,
            value=fields,
            reason=str(raw.get("reason") or "memory write agent proposal"),
            evidence_ids=evidence_ids,
            provenance="model",
            confidence=confidence,
        )
        fingerprint = json.dumps(
            {
                "stage": request.execution_stage,
                "source_agent": request.source_agent,
                "target_thread_id": request.target_thread_id,
                "update": update.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return StateUpdateCandidate(
            update=update,
            target_thread_id=request.target_thread_id,
            provenance="model",
            confidence=confidence,
            idempotency_key=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32],
            source_agent=request.source_agent,
            execution_stage=request.execution_stage,
        )

    @staticmethod
    def _messages(request: MemoryWriteRequest) -> list[dict[str, Any]]:
        payload = {
            "task": "Propose minimal durable memory candidates. Never invent facts.",
            "source_agent": request.source_agent,
            "execution_stage": request.execution_stage,
            "user_message": request.message,
            "support_handoff_or_service_result": request.handoff,
            "result_summary": request.result_summary,
            "allowed_evidence_ids": request.evidence_ids,
            "output_json": {
                "candidates": [
                    {
                        "update_key": "product_support|case_intake",
                        "fields": {"stable_fact": "value"},
                        "confidence": 0.0,
                        "reason": "short factual reason",
                        "evidence_ids": ["only ids from allowed_evidence_ids"],
                    }
                ]
            },
        }
        return [
            {"role": "system", "content": "Return JSON only. You propose memory; Runtime validates and persists."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("memory write response must be an object")
    return value


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
