"""Deterministic final authority for session-memory writes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from nikon0.app.schemas.memory import IssueThread, SessionIssueMemory
from nikon0.memory.governance.types import MemoryConflict, MemoryWriteDecision, StateUpdateCandidate


_PHONE_RE = re.compile(r"^(?:\+?\d[\d -]{6,20})$")
_ORDER_RE = re.compile(r"^[A-Za-z]{2,8}-?[A-Za-z0-9]{3,32}$")
_CRITICAL_HINTS = ("product", "model", "order", "phone", "mobile", "address", "ticket", "case_id")
_PROVENANCE_RANK = {"model": 0, "skill": 1, "workflow": 2, "verified_tool": 3, "user": 4, "runtime": 5}


class MemoryWriteGate:
    """Reject unsafe writes and require confirmation before key facts change."""

    def adapt_updates(self, updates, *, risk_level: str, selected_skill: str | None) -> list[StateUpdateCandidate]:
        provenance = "workflow" if selected_skill == "case_intake" else "skill"
        confidence = 0.92 if provenance == "workflow" else 0.82
        candidates: list[StateUpdateCandidate] = []
        for update in updates:
            fingerprint = json.dumps(update.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, default=str)
            update = update.model_copy(update={"provenance": provenance, "confidence": confidence})
            candidates.append(StateUpdateCandidate(
                update=update,
                provenance=provenance,
                confidence=confidence,
                risk_level=risk_level if risk_level in {"low", "medium", "high"} else "low",
                idempotency_key=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32],
            ))
        return candidates

    def validate(
        self,
        memory: SessionIssueMemory,
        candidates: list[StateUpdateCandidate],
    ) -> list[MemoryWriteDecision]:
        return [self.validate_one(memory, candidate) for candidate in candidates]

    def validate_one(self, memory: SessionIssueMemory, candidate: StateUpdateCandidate) -> MemoryWriteDecision:
        target = None if candidate.create_thread else memory.threads.get(candidate.target_thread_id or memory.active_thread_id or "")
        if target is not None and target.status in {"submitted", "resolved", "cancelled"}:
            return MemoryWriteDecision(
                candidate_id=candidate.candidate_id,
                outcome="reject",
                target_thread_id=target.thread_id,
                reason=f"cannot write terminal thread status={target.status}",
            )
        invalid = self._invalid_field(candidate.update.value)
        if invalid:
            return MemoryWriteDecision(
                candidate_id=candidate.candidate_id,
                outcome="reject",
                target_thread_id=candidate.target_thread_id,
                reason=invalid,
            )
        if candidate.provenance == "model" and self._contains_critical_field(candidate.update.value):
            return MemoryWriteDecision(
                candidate_id=candidate.candidate_id,
                outcome="reject",
                target_thread_id=candidate.target_thread_id,
                reason="model provenance cannot write critical memory fields",
            )
        conflicts = self._find_conflicts(target, candidate)
        if conflicts:
            return MemoryWriteDecision(
                candidate_id=candidate.candidate_id,
                outcome="needs_confirmation",
                target_thread_id=target.thread_id if target else candidate.target_thread_id,
                reason="critical memory conflict requires user confirmation",
                conflicts=conflicts,
            )
        if self._is_no_op(target, candidate):
            return MemoryWriteDecision(
                candidate_id=candidate.candidate_id,
                outcome="no_op",
                target_thread_id=target.thread_id if target else candidate.target_thread_id,
                reason="candidate repeats existing active facts",
            )
        return MemoryWriteDecision(
            candidate_id=candidate.candidate_id,
            outcome="accept",
            target_thread_id=target.thread_id if target else candidate.target_thread_id,
            reason="candidate passed memory governance checks",
            update=candidate.update,
        )

    def _find_conflicts(self, thread: IssueThread | None, candidate: StateUpdateCandidate) -> list[MemoryConflict]:
        if thread is None or not isinstance(candidate.update.value, dict):
            return []
        existing = {fact.kind: fact for fact in thread.facts.values()}
        conflicts: list[MemoryConflict] = []
        for field, incoming in self._flatten(candidate.update.key, candidate.update.value).items():
            current = existing.get(field)
            if current is None or current.value == incoming or not self._is_critical(field):
                continue
            if self._can_replace(current.source, current.confidence, candidate.provenance, candidate.confidence):
                continue
            conflicts.append(MemoryConflict(
                field=field,
                existing_value=current.value,
                incoming_value=incoming,
                existing_provenance=current.source if current.source in _PROVENANCE_RANK else "runtime",
                incoming_provenance=candidate.provenance,
                reason="incoming value conflicts with higher-trust stored fact",
            ))
        return conflicts

    def _is_no_op(self, thread: IssueThread | None, candidate: StateUpdateCandidate) -> bool:
        if thread is None or not isinstance(candidate.update.value, dict):
            return False
        existing = {fact.kind: fact.value for fact in thread.facts.values()}
        flattened = self._flatten(candidate.update.key, candidate.update.value)
        return bool(flattened) and all(existing.get(key) == value for key, value in flattened.items())

    @staticmethod
    def _can_replace(existing_source: str, existing_confidence: float, incoming_source: str, incoming_confidence: float) -> bool:
        return _PROVENANCE_RANK.get(incoming_source, 0) > _PROVENANCE_RANK.get(existing_source, 0) and incoming_confidence >= existing_confidence

    def _invalid_field(self, value: Any, prefix: str = "") -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                failure = self._invalid_field(item, f"{prefix}.{key}" if prefix else str(key))
                if failure:
                    return failure
            return ""
        if not isinstance(value, str):
            return ""
        field = prefix.lower()
        if "phone" in field or "mobile" in field or "电话" in field:
            if not _PHONE_RE.fullmatch(value.strip()):
                return f"invalid phone format for {prefix}"
        if "order" in field or "订单" in field:
            if not _ORDER_RE.fullmatch(value.strip()):
                return f"invalid order id format for {prefix}"
        if len(value) > 2000:
            return f"memory value too long for {prefix}"
        return ""

    def _contains_critical_field(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return any(self._is_critical(key) or (isinstance(item, dict) and self._contains_critical_field(item)) for key, item in value.items())

    @staticmethod
    def _is_critical(field: str) -> bool:
        # The update namespace is not itself a fact. For example,
        # `product_support.last_query` must remain mutable across normal turns.
        leaf = field.rsplit(".", 1)[-1]
        lowered = leaf.lower()
        return any(hint in lowered for hint in _CRITICAL_HINTS) or any(token in leaf for token in ("订单", "电话", "地址", "型号", "工单"))

    @staticmethod
    def _flatten(update_key: str, value: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key == "ticket_payload" and isinstance(item, dict):
                output.update({f"{update_key}.{nested}": nested_value for nested, nested_value in item.items()})
            else:
                output[f"{update_key}.{key}"] = item
        return output
