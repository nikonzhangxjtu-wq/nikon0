"""Minimal context governance for Phase 1."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.context.runtime import ContextRuntime


class ContextGovernance:
    """Builds a small governed context and records governance decisions."""

    def __init__(self, *, context_runtime: ContextRuntime | None = None) -> None:
        self.context_runtime = context_runtime or ContextRuntime()

    def govern(self, context: AgentContext) -> AgentContext:
        pack = self.context_runtime.build_pack(context)
        return self._attach_pack(context, pack)

    async def agovern(self, context: AgentContext) -> AgentContext:
        pack = await self.context_runtime.build_pack_async(context)
        return self._attach_pack(context, pack)

    @staticmethod
    def _attach_pack(context: AgentContext, pack) -> AgentContext:
        context.context_pack = pack
        context.governed_context = pack.render()
        event = {
            "message_chars": len(context.request.message.strip()),
            "image_count": len(context.request.images),
            "transcript_chars": len(context.transcript_context.strip()),
            "memory_chars": len(context.memory_context.strip()),
            "memory_preview": context.memory_context.strip()[:240],
            "section_count": len(pack.sections),
            "section_names": [section.name for section in pack.sections],
            "budget_report": pack.budget_report.model_dump(),
            "strategy": "context_pack_v1",
        }
        context.trace.context_events.append(event)
        context.trace.add_event("context.govern", "built context pack", **event)
        return context
