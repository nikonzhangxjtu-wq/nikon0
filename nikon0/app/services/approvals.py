"""Approval and handoff stores for HITL workflows."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from nikon0.app.schemas.safety import ApprovalRequest, ApprovalStatus, HandoffRequest


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._approvals: dict[str, ApprovalRequest] = {}
        self._handoffs: dict[str, HandoffRequest] = {}
        self._approval_by_session: dict[str, list[str]] = defaultdict(list)
        self._handoff_by_session: dict[str, list[str]] = defaultdict(list)

    def create_approval(self, approval: ApprovalRequest) -> ApprovalRequest:
        self._approvals[approval.approval_id] = approval
        self._approval_by_session[approval.session_id].append(approval.approval_id)
        return approval

    def update_approval(self, approval_id: str, status: ApprovalStatus) -> ApprovalRequest | None:
        approval = self._approvals.get(approval_id)
        if approval is None:
            return None
        updated = approval.model_copy(update={"status": status})
        self._approvals[approval_id] = updated
        return updated

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        return self._approvals.get(approval_id)

    def list_approvals(self, session_id: str | None = None) -> list[ApprovalRequest]:
        if session_id is None:
            return list(self._approvals.values())
        return [
            self._approvals[item]
            for item in self._approval_by_session.get(session_id, [])
            if item in self._approvals
        ]

    def create_handoff(self, handoff: HandoffRequest) -> HandoffRequest:
        self._handoffs[handoff.handoff_id] = handoff
        self._handoff_by_session[handoff.session_id].append(handoff.handoff_id)
        return handoff

    def get_handoff(self, handoff_id: str) -> HandoffRequest | None:
        return self._handoffs.get(handoff_id)

    def list_handoffs(self, session_id: str | None = None) -> list[HandoffRequest]:
        if session_id is None:
            return list(self._handoffs.values())
        return [
            self._handoffs[item]
            for item in self._handoff_by_session.get(session_id, [])
            if item in self._handoffs
        ]


class JsonlApprovalStore(InMemoryApprovalStore):
    """Small append-only HITL store.

    Reads replay all events into memory. This keeps production debugging simple
    until a database-backed queue is introduced.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    @classmethod
    def default(cls) -> "JsonlApprovalStore":
        return cls(Path("nikon0/infra/runtime/approvals.jsonl"))

    def create_approval(self, approval: ApprovalRequest) -> ApprovalRequest:
        created = super().create_approval(approval)
        self._append({"kind": "approval", "action": "create", "payload": created.model_dump()})
        return created

    def update_approval(self, approval_id: str, status: ApprovalStatus) -> ApprovalRequest | None:
        updated = super().update_approval(approval_id, status)
        if updated is not None:
            self._append({"kind": "approval", "action": "update", "payload": updated.model_dump()})
        return updated

    def create_handoff(self, handoff: HandoffRequest) -> HandoffRequest:
        created = super().create_handoff(handoff)
        self._append({"kind": "handoff", "action": "create", "payload": created.model_dump()})
        return created

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False))
            fp.write("\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                payload = event.get("payload") or {}
                if event.get("kind") == "approval":
                    approval = ApprovalRequest.model_validate(payload)
                    InMemoryApprovalStore.create_approval(self, approval)
                elif event.get("kind") == "handoff":
                    handoff = HandoffRequest.model_validate(payload)
                    InMemoryApprovalStore.create_handoff(self, handoff)
