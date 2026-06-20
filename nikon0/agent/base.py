"""Agent protocol and registry."""

from __future__ import annotations

from typing import Protocol

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import AgentMatch, AgentResult


class Agent(Protocol):
    name: str
    description: str

    async def can_handle(self, context: AgentContext) -> AgentMatch:
        ...

    async def run(self, context: AgentContext) -> AgentResult:
        ...


class AgentRegistry:
    """Registry for runtime-selectable agents."""

    def __init__(self, agents: list[Agent] | None = None) -> None:
        self._agents = tuple(agents or [])

    def list(self) -> tuple[Agent, ...]:
        return self._agents

    def get(self, name: str) -> Agent | None:
        lowered = name.lower()
        for agent in self._agents:
            if agent.name.lower() == lowered:
                return agent
        return None

    async def best_match(self, context: AgentContext) -> tuple[Agent | None, AgentMatch]:
        best_agent: Agent | None = None
        best_match = AgentMatch(matched=False, confidence=0.0, reason="no agent matched")
        for agent in self._agents:
            match = await agent.can_handle(context)
            context.trace.add_event(
                "agent.match",
                f"{agent.name}: {match.reason}",
                matched=match.matched,
                confidence=match.confidence,
            )
            if match.matched and match.confidence > best_match.confidence:
                best_agent = agent
                best_match = match
        return best_agent, best_match
