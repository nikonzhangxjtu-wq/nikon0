"""Session issue memory store."""

from __future__ import annotations

import time
from typing import Any

from nikon0.app.schemas.capability import StateUpdate
from nikon0.app.schemas.memory import EvidenceRef, IssueFact, IssueThread, IssueType, SessionIssueMemory


class InMemorySessionIssueStore:
    """In-memory implementation of the formal SessionIssueMemory model."""

    def __init__(self) -> None:
        self._state: dict[str, SessionIssueMemory] = {}

    def load(self, session_id: str) -> SessionIssueMemory:
        memory = self._state.get(session_id)
        if memory is None:
            memory = SessionIssueMemory(session_id=session_id)
            self._state[session_id] = memory
        return memory.model_copy(deep=True)

    def load_flat(self, session_id: str) -> dict[str, Any]:
        return dict(self.load(session_id).flat_state)

    def apply_updates(
        self,
        session_id: str,
        updates: list[StateUpdate],
        *,
        turn_id: str = "",
        target_thread_id: str | None = None,
        create_thread: bool = False,
    ) -> SessionIssueMemory:
        memory = self._state.get(session_id) or SessionIssueMemory(session_id=session_id)
        memory.turn_count += 1
        memory.updated_at = time.time()
        if updates:
            thread = self._ensure_active_thread(memory, updates, target_thread_id=target_thread_id, create_thread=create_thread)
            if turn_id and turn_id not in thread.last_turn_ids:
                thread.last_turn_ids.append(turn_id)
            for update in updates:
                self._apply_update(memory, thread, update, turn_id=turn_id)
            thread.updated_at = time.time()
            memory.threads[thread.thread_id] = thread
        self._state[session_id] = memory
        return memory.model_copy(deep=True)

    def _ensure_active_thread(
        self,
        memory: SessionIssueMemory,
        updates: list[StateUpdate],
        *,
        target_thread_id: str | None = None,
        create_thread: bool = False,
    ) -> IssueThread:
        if target_thread_id and target_thread_id in memory.threads:
            memory.active_thread_id = target_thread_id
            return memory.threads[target_thread_id]
        active = memory.active_thread()
        if active is not None and not create_thread:
            return active
        issue_type = self._infer_issue_type(updates)
        thread = IssueThread(status="open", issue_type=issue_type)
        memory.active_thread_id = thread.thread_id
        memory.threads[thread.thread_id] = thread
        return thread

    def _apply_update(
        self,
        memory: SessionIssueMemory,
        thread: IssueThread,
        update: StateUpdate,
        *,
        turn_id: str,
    ) -> None:
        memory.flat_state[update.key] = update.value
        if update.key in {"product_support", "case_intake"}:
            memory.active_skill = update.key
        evidence_ref = EvidenceRef(
            turn_id=turn_id,
            source="state_update",
            text=update.reason or update.key,
            payload={"evidence_ids": update.evidence_ids},
        )
        thread.evidence_refs[evidence_ref.evidence_ref_id] = evidence_ref
        if isinstance(update.value, dict):
            for key, value in update.value.items():
                if key == "ticket_payload" and isinstance(value, dict):
                    for ticket_key, ticket_value in value.items():
                        self._upsert_fact(
                            thread,
                            kind=f"{update.key}.{ticket_key}",
                            value=ticket_value,
                            evidence_ref_id=evidence_ref.evidence_ref_id,
                            source=update.provenance,
                            confidence=update.confidence,
                        )
                    product_model = value.get("product_model")
                    if product_model:
                        thread.product_model = str(product_model)
                else:
                    self._upsert_fact(
                        thread,
                        kind=f"{update.key}.{key}",
                        value=value,
                        evidence_ref_id=evidence_ref.evidence_ref_id,
                        source=update.provenance,
                        confidence=update.confidence,
                    )
            self._apply_structured_fields(memory, thread, update)
        else:
            self._upsert_fact(
                thread,
                kind=update.key,
                value=update.value,
                evidence_ref_id=evidence_ref.evidence_ref_id,
                source=update.provenance,
                confidence=update.confidence,
            )
        self._refresh_thread_status(thread, update)

    def _apply_structured_fields(
        self,
        memory: SessionIssueMemory,
        thread: IssueThread,
        update: StateUpdate,
    ) -> None:
        if not isinstance(update.value, dict):
            return
        if update.key == "product_support":
            self._apply_product_support_state(memory, thread, update.value, update.reason)
        elif update.key == "case_intake":
            self._apply_case_intake_state(memory, thread, update.value, update.reason)

    @staticmethod
    def _apply_product_support_state(
        memory: SessionIssueMemory,
        thread: IssueThread,
        value: dict[str, Any],
        reason: str,
    ) -> None:
        product_id = str(value.get("selected_product_id") or "").strip()
        display_name = str(value.get("selected_display_name") or "").strip()
        manual_names = [
            str(item)
            for item in value.get("manual_names", [])
            if str(item).strip()
        ] if isinstance(value.get("manual_names"), list) else []
        resolution = value.get("product_resolution") if isinstance(value.get("product_resolution"), dict) else {}
        if product_id or display_name or resolution:
            active_product = {
                "product_id": product_id or resolution.get("product_id"),
                "display_name": display_name or resolution.get("display_name"),
                "manual_names": manual_names or list(resolution.get("manual_names") or []),
                "source": resolution.get("source") or "product_support",
            }
            memory.active_product = {
                key: item
                for key, item in active_product.items()
                if item not in (None, "", [])
            }
            thread.product_ref = dict(memory.active_product)
            if memory.active_product.get("product_id"):
                thread.product_model = str(memory.active_product["product_id"])
        query = str(value.get("last_query") or "").strip()
        if query:
            thread.user_goal = query
        if not thread.summary:
            subject = display_name or product_id or "商品"
            thread.summary = f"{subject}商品问答"
        if reason and not thread.user_goal:
            thread.user_goal = reason

    @staticmethod
    def _apply_case_intake_state(
        memory: SessionIssueMemory,
        thread: IssueThread,
        value: dict[str, Any],
        reason: str,
    ) -> None:
        missing = value.get("workflow_missing_slots")
        if not isinstance(missing, list):
            missing = value.get("missing_slots")
        thread.missing_info = [str(item) for item in missing or [] if str(item).strip()]
        workflow_keys = {
            "workflow_name",
            "workflow_intent",
            "workflow_status",
            "workflow_missing_slots",
            "requires_approval",
            "handoff_required",
            "next_tool",
            "risk_level",
            "reason",
        }
        snapshot = {
            key: value[key]
            for key in workflow_keys
            if key in value and value[key] not in (None, "")
        }
        if snapshot:
            if "workflow_intent" in snapshot and "intent" not in snapshot:
                snapshot["intent"] = snapshot["workflow_intent"]
            if "workflow_missing_slots" in snapshot and "missing_slots" not in snapshot:
                snapshot["missing_slots"] = snapshot["workflow_missing_slots"]
            thread.workflow_snapshot = snapshot
        ticket = value.get("ticket_payload")
        if isinstance(ticket, dict):
            product_model = str(ticket.get("product_model") or "").strip()
            if product_model:
                thread.product_model = product_model
                if not memory.active_product:
                    memory.active_product = {"display_name": product_model, "source": "case_intake"}
                thread.product_ref = dict(memory.active_product)
            ticket_id = str(ticket.get("ticket_id") or ticket.get("case_id") or "").strip()
            if ticket_id:
                thread.linked_ticket_id = ticket_id
        if not thread.summary:
            intent = str(value.get("workflow_intent") or "").strip() or thread.issue_type
            thread.summary = f"{intent} intake"
        if reason and not thread.user_goal:
            thread.user_goal = reason

    @staticmethod
    def _upsert_fact(
        thread: IssueThread,
        *,
        kind: str,
        value: Any,
        evidence_ref_id: str,
        source: str = "runtime",
        confidence: float = 1.0,
    ) -> None:
        fact = thread.facts.get(kind)
        now = time.time()
        if fact is None:
            thread.facts[kind] = IssueFact(
                kind=kind,
                value=value,
                source=source,
                confidence=confidence,
                evidence_ref_id=evidence_ref_id,
                created_at=now,
                updated_at=now,
            )
            return
        fact.value = value
        fact.source = source
        fact.confidence = confidence
        fact.evidence_ref_id = evidence_ref_id
        fact.updated_at = now
        thread.facts[kind] = fact

    @staticmethod
    def _refresh_thread_status(thread: IssueThread, update: StateUpdate) -> None:
        if update.key != "case_intake" or not isinstance(update.value, dict):
            return
        completed = bool(update.value.get("completed"))
        missing = update.value.get("missing_slots")
        exited = bool(update.value.get("exited"))
        if exited:
            thread.status = "cancelled"
        elif completed:
            thread.status = "submitted"
        elif isinstance(missing, list) and missing:
            thread.status = "waiting_user"
        else:
            thread.status = "diagnosing"
        ticket = update.value.get("ticket_payload")
        if isinstance(ticket, dict):
            intent = str(ticket.get("intent") or "").lower()
            if intent in {"repair", "refund", "complaint"}:
                thread.issue_type = intent  # type: ignore[assignment]
        workflow_intent = str(update.value.get("workflow_intent") or "").lower()
        if workflow_intent in {"repair", "refund", "complaint"}:
            thread.issue_type = workflow_intent  # type: ignore[assignment]

    @staticmethod
    def _infer_issue_type(updates: list[StateUpdate]) -> IssueType:
        for update in updates:
            if update.key == "case_intake" and isinstance(update.value, dict):
                ticket = update.value.get("ticket_payload")
                if isinstance(ticket, dict):
                    intent = str(ticket.get("intent") or "").lower()
                    if intent in {"repair", "refund", "complaint"}:
                        return intent  # type: ignore[return-value]
                return "repair"
        return "unknown"
