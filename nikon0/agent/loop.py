"""Agent loop with bounded plan/act/tool-result iterations."""

from __future__ import annotations

from dataclasses import dataclass, field

from nikon0.agent.base import AgentRegistry
from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import AgentResult
from nikon0.tools.runtime import ToolRuntime


@dataclass(frozen=True)
class AgentLoopStep:
    turn: int
    selected_agent: str
    tool_call_count: int
    tool_result_count: int
    stop_reason: str = ""


@dataclass(frozen=True)
class AgentLoopResult:
    result: AgentResult
    turn_count: int
    stop_reason: str
    steps: list[AgentLoopStep] = field(default_factory=list)


class AgentLoop:
    """Minimal loop inspired by query.ts: plan/act, observe tools, repeat."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        tool_runtime: ToolRuntime,
        max_turns: int = 4,
    ) -> None:
        self.agent_registry = agent_registry
        self.tool_runtime = tool_runtime
        self.max_turns = max(1, int(max_turns))

    async def run(self, context: AgentContext) -> AgentLoopResult:
        context.max_turns = self.max_turns
        steps: list[AgentLoopStep] = []
        last_result = AgentResult(status="failed", answer_draft="Agent loop did not run.")

        for turn in range(1, self.max_turns + 1):
            context.loop_turn = turn
            context.trace.add_event("loop.turn_start", f"turn {turn} started", turn=turn)
            agent, match = await self.agent_registry.best_match(context)
            if agent is None:
                context.trace.add_event("loop.stop", "no agent matched", turn=turn, reason=match.reason)
                return AgentLoopResult(
                    result=AgentResult(
                        status="failed",
                        answer_draft="当前没有可用 Agent 处理该请求。",
                        risk_level="low",
                    ),
                    turn_count=turn,
                    stop_reason="no_agent",
                    steps=steps,
                )

            context.selected_agent = agent.name
            if agent.name not in context.trace.selected_agents:
                context.trace.selected_agents.append(agent.name)
            context.trace.add_event("agent.select", f"selected {agent.name}", turn=turn, reason=match.reason)
            last_result = await agent.run(context)

            if not last_result.tool_calls:
                step = AgentLoopStep(
                    turn=turn,
                    selected_agent=agent.name,
                    tool_call_count=0,
                    tool_result_count=0,
                    stop_reason="no_tool_calls",
                )
                steps.append(step)
                context.trace.add_event("loop.stop", "no tool calls requested", turn=turn)
                return AgentLoopResult(
                    result=last_result,
                    turn_count=turn,
                    stop_reason="no_tool_calls",
                    steps=steps,
                )

            before_results = len(context.tool_results)
            for tool_call in last_result.tool_calls:
                tool_result = await self.tool_runtime.call(context, tool_call)
                if context.retry_tool_errors and not tool_result.ok:
                    context.trace.add_event(
                        "tool.retry",
                        f"retrying {tool_call.service_id}.{tool_call.tool_name} after failure",
                        service_id=tool_call.service_id,
                        tool_name=tool_call.tool_name,
                        error_code=tool_result.error_code,
                    )
                    context.tool_results.append(tool_result.model_dump())
                    tool_result = await self.tool_runtime.call(context, tool_call)
                context.tool_results.append(tool_result.model_dump())
            produced_results = len(context.tool_results) - before_results
            steps.append(
                AgentLoopStep(
                    turn=turn,
                    selected_agent=agent.name,
                    tool_call_count=len(last_result.tool_calls),
                    tool_result_count=produced_results,
                )
            )
            context.trace.add_event(
                "loop.observe_tools",
                "tool results added to context",
                turn=turn,
                tool_call_count=len(last_result.tool_calls),
                tool_result_count=produced_results,
            )

        context.trace.add_event("loop.stop", "max turns reached", max_turns=self.max_turns)
        return AgentLoopResult(
            result=last_result,
            turn_count=self.max_turns,
            stop_reason="max_turns",
            steps=steps,
        )
