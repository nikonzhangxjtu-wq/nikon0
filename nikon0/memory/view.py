"""Runtime memory view builder.

The model should consume a small, governed memory view instead of raw session
storage. This P0 builder keeps only current session focus and open issue
summaries; deeper retrieval and compression can be layered on later.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nikon0.app.schemas.memory import IssueThread, SessionIssueMemory
from nikon0.memory.governance.types import MemoryReadPlan


class MemoryView(BaseModel):
    session_id: str
    active_product: dict[str, Any] = Field(default_factory=dict)
    active_skill: str | None = None
    active_thread: dict[str, Any] = Field(default_factory=dict)
    open_threads: list[dict[str, Any]] = Field(default_factory=list)
    session_facts: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        lines = ["[Memory View]"]
        lines.append(f"session_id: {self.session_id}")
        if self.active_skill:
            lines.append(f"active_skill: {self.active_skill}")
        if self.active_product:
            lines.append("active_product:")
            for key in ("product_id", "display_name", "manual_names", "source"):
                if key in self.active_product:
                    lines.append(f"- {key}: {self.active_product[key]}")
        if self.active_thread:
            lines.append("active_issue:")
            for key in ("thread_id", "status", "issue_type", "summary", "user_goal"):
                value = self.active_thread.get(key)
                if value:
                    lines.append(f"- {key}: {value}")
            missing = self.active_thread.get("missing_info") or []
            if missing:
                lines.append(f"- missing_info: {missing}")
            workflow = self.active_thread.get("workflow_snapshot") or {}
            if workflow:
                lines.append("- workflow:")
                for key in ("workflow_name", "intent", "workflow_status", "requires_approval", "handoff_required"):
                    if key in workflow:
                        lines.append(f"  - {key}: {workflow[key]}")
            facts = self.active_thread.get("facts") or []
            if facts:
                lines.append("- facts:")
                for fact in facts:
                    lines.append(f"  - {fact}")
        if self.open_threads:
            lines.append("open_issue_summaries:")
            for item in self.open_threads:
                label = item.get("summary") or item.get("issue_type") or item.get("thread_id")
                lines.append(f"- {item.get('thread_id')}: {label} ({item.get('status')})")
        if self.session_facts:
            lines.append("session_facts:")
            for key, value in self.session_facts.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)


class MemoryViewBuilder:
    """Build a compact model-facing view from formal session memory."""

    def __init__(self, *, max_open_threads: int = 5, max_facts: int = 8, char_budget: int = 1600) -> None:
        self.max_open_threads = max(1, int(max_open_threads))
        self.max_facts = max(1, int(max_facts))
        self.char_budget = max(400, int(char_budget))

    def build(self, memory: SessionIssueMemory | None, *, read_plan: MemoryReadPlan | None = None) -> MemoryView:
        if memory is None:
            return MemoryView(session_id="")
        active_thread = memory.active_thread()
        selected_ids = set(read_plan.thread_ids) if read_plan is not None else set()
        if read_plan is not None and active_thread is not None and active_thread.thread_id not in selected_ids:
            active_thread = None
        view = MemoryView(
            session_id=memory.session_id,
            active_product=dict(memory.active_product),
            active_skill=memory.active_skill,
            active_thread=self._thread_summary(active_thread) if active_thread is not None else {},
            open_threads=[
                self._thread_summary(thread, include_facts=False)
                for thread in self._open_threads(memory, active_thread, selected_ids)
            ],
            session_facts=self._session_facts(memory) if read_plan is None or read_plan.include_session_facts else {},
        )
        return self._trim_to_budget(view)

    def _open_threads(
        self,
        memory: SessionIssueMemory,
        active_thread: IssueThread | None,
        selected_ids: set[str] | None = None,
    ) -> list[IssueThread]:
        active_id = active_thread.thread_id if active_thread is not None else None
        threads = [
            thread
            for thread in memory.threads.values()
            if thread.thread_id != active_id
            and thread.status not in {"resolved", "cancelled"}
            and (not selected_ids or thread.thread_id in selected_ids)
        ]
        threads.sort(key=lambda item: item.updated_at, reverse=True)
        return threads[: self.max_open_threads]

    def _thread_summary(self, thread: IssueThread | None, *, include_facts: bool = True) -> dict[str, Any]:  # 把活跃线程压成active_thread 摘要
        if thread is None:
            return {}
        data: dict[str, Any] = {
            "thread_id": thread.thread_id,
            "status": thread.status,
            "issue_type": thread.issue_type,
            "product_model": thread.product_model,
            "product_ref": dict(thread.product_ref),
            "summary": thread.summary,
            "user_goal": thread.user_goal,
            "missing_info": list(thread.missing_info),
            "workflow_snapshot": dict(thread.workflow_snapshot),
        }
        if include_facts:
            facts = []
            for fact in sorted(thread.facts.values(), key=lambda item: item.updated_at, reverse=True)[: self.max_facts]:
                facts.append(f"{fact.kind}={fact.value}")
            data["facts"] = facts
        return {key: value for key, value in data.items() if value not in (None, "", [], {})}

    def _session_facts(self, memory: SessionIssueMemory) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        for fact in sorted(memory.session_facts.values(), key=lambda item: item.updated_at, reverse=True)[: self.max_facts]:
            facts[fact.kind] = fact.value
        return facts

    def _trim_to_budget(self, view: MemoryView) -> MemoryView:
        rendered = view.render()
        if len(rendered) <= self.char_budget:
            return view
        active = dict(view.active_thread)
        facts = list(active.get("facts") or [])
        while facts and len(MemoryView(**view.model_dump()).render()) > self.char_budget:
            facts.pop()
            active["facts"] = facts
            view = view.model_copy(update={"active_thread": active})
        if len(view.render()) <= self.char_budget:
            return view
        return view.model_copy(update={"open_threads": []})
