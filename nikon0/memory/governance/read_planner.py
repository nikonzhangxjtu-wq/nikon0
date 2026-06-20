"""LLM-assisted, guard-railed memory read planning."""

from __future__ import annotations

import json
from typing import Any

from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.llm.client import ChatModelClient
from nikon0.memory.governance.lifecycle import IssueThreadLifecycleManager
from nikon0.memory.governance.types import MemoryReadPlan, ThreadDecision


class MemoryReadPlanner:
    def __init__(self, *, lifecycle: IssueThreadLifecycleManager | None = None, client: ChatModelClient | None = None) -> None:
        self.lifecycle = lifecycle or IssueThreadLifecycleManager()
        self.client = client

    async def plan(self, memory: SessionIssueMemory, message: str, transcript: str) -> tuple[MemoryReadPlan, ThreadDecision]:
        fallback_decision = self.lifecycle.decide(memory, message)
        if self.client is None:
            return self._fallback(memory, fallback_decision, "llm planner disabled")
        try:
            raw = await self.client.complete(self._messages(memory, message, transcript))
            payload = _parse_json(raw)
            decision = self._validate_decision(memory, payload, fallback_decision)
            plan = self._validate_plan(memory, payload, decision)
            return plan, decision
        except Exception as exc:  # noqa: BLE001
            return self._fallback(memory, fallback_decision, f"{type(exc).__name__}: {exc}")

    def _fallback(self, memory: SessionIssueMemory, decision: ThreadDecision, reason: str) -> tuple[MemoryReadPlan, ThreadDecision]:
        ids = [decision.thread_id] if decision.thread_id else []
        return MemoryReadPlan(
            thread_ids=ids,
            include_session_facts=False,
            include_workflow=True,
            source="deterministic",
            reason=decision.reason,
            fallback_reason=reason,
        ), decision

    @staticmethod
    def _messages(memory: SessionIssueMemory, message: str, transcript: str) -> list[dict[str, Any]]:
        open_threads = [
            {
                "thread_id": thread.thread_id,
                "status": thread.status,
                "product_model": thread.product_model,
                "summary": thread.summary,
                "goal": thread.user_goal,
            }
            for thread in memory.threads.values()
            if thread.status not in {"submitted", "resolved", "cancelled"}
        ]
        prompt = {
            "task": "Choose memory thread action and minimum read scope. Never invent facts.",
            "user_message": message,
            "active_thread_id": memory.active_thread_id,
            "open_threads": open_threads,
            "recent_conversation": transcript[-1000:],
            "output_json": {
                "action": "continue_active|switch_open_thread|create_thread|needs_clarification",
                "thread_id": "must be an existing open thread for continue/switch",
                "thread_ids_to_read": ["existing ids only"],
                "include_session_facts": False,
                "include_workflow": True,
                "confidence": 0.0,
                "reason": "short explanation",
            },
        }
        return [
            {"role": "system", "content": "Return only valid JSON. You recommend; the runtime validates and decides."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

    @staticmethod
    def _validate_decision(memory: SessionIssueMemory, payload: dict[str, Any], fallback: ThreadDecision) -> ThreadDecision:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        action = str(payload.get("action") or "")
        thread_id = str(payload.get("thread_id") or "") or None
        valid_open = {thread.thread_id for thread in memory.threads.values() if thread.status not in {"submitted", "resolved", "cancelled"}}
        if confidence < 0.65 or action not in {"continue_active", "switch_open_thread", "create_thread", "needs_clarification"}:
            return fallback
        if action in {"continue_active", "switch_open_thread"} and thread_id not in valid_open:
            return fallback
        return ThreadDecision(action=action, thread_id=thread_id, confidence=confidence, source="llm", reason=str(payload.get("reason") or "llm memory decision"))

    @staticmethod
    def _validate_plan(memory: SessionIssueMemory, payload: dict[str, Any], decision: ThreadDecision) -> MemoryReadPlan:
        valid = {thread.thread_id for thread in memory.threads.values() if thread.status not in {"resolved", "cancelled"}}
        requested = payload.get("thread_ids_to_read") or []
        ids = [str(item) for item in requested if str(item) in valid][:2]
        if decision.thread_id and decision.thread_id not in ids:
            ids.insert(0, decision.thread_id)
        return MemoryReadPlan(
            thread_ids=ids,
            include_session_facts=bool(payload.get("include_session_facts")),
            include_workflow=bool(payload.get("include_workflow", True)),
            include_ticket_history=bool(payload.get("include_ticket_history")),
            source=decision.source,
            confidence=decision.confidence,
            reason=decision.reason,
        )


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("memory planner output must be object")
    return value
