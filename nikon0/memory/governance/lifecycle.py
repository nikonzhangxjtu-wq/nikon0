"""Thread lifecycle decisions with deterministic safety guards."""

from __future__ import annotations

from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.knowledge.product_resolver import ProductResolver
from nikon0.memory.governance.types import ThreadDecision


class IssueThreadLifecycleManager:
    def __init__(self, resolver: ProductResolver | None = None) -> None:
        self.resolver = resolver or ProductResolver()

    def decide(self, memory: SessionIssueMemory, message: str) -> ThreadDecision:
        active = memory.active_thread()
        # Explicit product identity in the current message must take precedence
        # over the resolver's session-product shortcut.
        resolved = self.resolver.resolve(message)
        product_id = resolved.product_id
        open_threads = [thread for thread in memory.threads.values() if thread.status not in {"submitted", "resolved", "cancelled"}]
        if active is not None and active.status in {"submitted", "resolved", "cancelled"}:
            return ThreadDecision(action="create_thread", reason="active thread is terminal")
        if product_id:
            for thread in open_threads:
                if thread.product_model == product_id and thread.thread_id != (active.thread_id if active else None):
                    if _looks_like_reference(message):
                        return ThreadDecision(action="switch_open_thread", thread_id=thread.thread_id, reason="explicit follow-up matches open product thread")
            if active is not None and active.product_model and active.product_model != product_id:
                return ThreadDecision(action="create_thread", reason="message identifies a different product")
        if active is not None and _looks_like_reference(message):
            return ThreadDecision(action="continue_active", thread_id=active.thread_id, reason="follow-up signal for active thread")
        if active is not None:
            return ThreadDecision(action="continue_active", thread_id=active.thread_id, reason="active thread remains usable")
        return ThreadDecision(action="create_thread", reason="session has no active thread")


def _looks_like_reference(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("刚才", "上次", "继续", "还是", "那个", "之前", "again", "still", "previous", "that one"))
