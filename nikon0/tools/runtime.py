"""Tool runtime with registry, permission checks, and trace events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol
from uuid import uuid4

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import (
    PermissionDecision,
    ToolCallRequest,
    ToolCallResult,
    ToolSpec,
)
from nikon0.app.schemas.safety import ApprovalRequest
from nikon0.tools.mcp_gateway import McpGatewayTool


class Tool(Protocol):
    spec: ToolSpec

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        ...


@dataclass(frozen=True)
class ToolHookEvent:
    stage: str
    tool_name: str
    message: str


PreToolHook = Callable[[AgentContext, ToolCallRequest], PermissionDecision]
PostToolHook = Callable[[AgentContext, ToolCallRequest, ToolCallResult], str]
FailureToolHook = Callable[[AgentContext, ToolCallRequest, str], str]


@dataclass
class HookRunner:
    pre_tool: tuple[PreToolHook, ...] = field(default_factory=tuple)
    post_tool: tuple[PostToolHook, ...] = field(default_factory=tuple)
    on_failure: tuple[FailureToolHook, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> "HookRunner":
        return cls(
            pre_tool=(_audit_pre_tool,),
            post_tool=(_audit_post_tool,),
            on_failure=(_audit_failure_tool,),
        )

    def run_pre(self, context: AgentContext, request: ToolCallRequest) -> tuple[PermissionDecision, list[ToolHookEvent]]:
        events: list[ToolHookEvent] = []
        for hook in self.pre_tool:
            decision = hook(context, request)
            events.append(ToolHookEvent("pre_tool", request.tool_name, decision.reason))
            if not decision.allowed:
                return decision, events
        return PermissionDecision(allowed=True, reason="pre_tool hooks passed"), events

    def run_post(self, context: AgentContext, request: ToolCallRequest, result: ToolCallResult) -> list[ToolHookEvent]:
        return [
            ToolHookEvent("post_tool", request.tool_name, hook(context, request, result))
            for hook in self.post_tool
        ]

    def run_failure(self, context: AgentContext, request: ToolCallRequest, reason: str) -> list[ToolHookEvent]:
        return [
            ToolHookEvent("on_failure", request.tool_name, hook(context, request, reason))
            for hook in self.on_failure
        ]


def _audit_pre_tool(context: AgentContext, request: ToolCallRequest) -> PermissionDecision:
    _ = context
    return PermissionDecision(
        allowed=True,
        reason=f"pre_tool accepted {request.service_id}.{request.tool_name}",
    )


def _audit_post_tool(context: AgentContext, request: ToolCallRequest, result: ToolCallResult) -> str:
    _ = context
    return f"post_tool recorded ok={result.ok} for {request.service_id}.{request.tool_name}"


def _audit_failure_tool(context: AgentContext, request: ToolCallRequest, reason: str) -> str:
    _ = context
    return f"on_failure recorded {request.service_id}.{request.tool_name}: {reason}"


class ToolRegistry:
    """Runtime tool snapshot."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = tuple(tools or [])

    def list(self, service_id: str | None = None) -> list[ToolSpec]:
        specs = [tool.spec for tool in self._tools]
        if service_id is None:
            return specs
        return [spec for spec in specs if spec.service_id == service_id]

    def get(self, service_id: str, tool_name: str) -> Tool | None:
        for tool in self._tools:
            if tool.spec.service_id == service_id and tool.spec.tool_name == tool_name:
                return tool
        return None


class ToolPermissionPolicy:
    """Phase 1 permission policy for tool calls."""

    def check(self, request: ToolCallRequest) -> PermissionDecision:
        if request.requires_approval:
            return PermissionDecision(
                allowed=False,
                reason="tool call requires approval; approval flow is not enabled yet",
                blocked_action=request.tool_name,
            )
        if request.risk_level == "high":
            return PermissionDecision(
                allowed=False,
                reason="high-risk tool call is blocked by phase1 policy",
                blocked_action=request.tool_name,
            )
        return PermissionDecision(allowed=True, reason="tool call allowed by phase1 policy")


class ToolRuntime:
    """Executes tool requests through a controlled lifecycle."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        permission_policy: ToolPermissionPolicy | None = None,
        hook_runner: HookRunner | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry(default_tools())
        self.permission_policy = permission_policy or ToolPermissionPolicy()
        self.hook_runner = hook_runner or HookRunner.default()

    async def list_tools(self, service_id: str | None = None) -> list[ToolSpec]:
        return self.registry.list(service_id)

    async def call_step(self, context: AgentContext, request: ToolCallRequest) -> ToolCallResult:
        """Execute a skill-internal tool step and record the observed result."""
        result = await self.call(context, request)
        context.tool_results.append(result.model_dump())
        return result

    async def call(self, context: AgentContext, request: ToolCallRequest) -> ToolCallResult:
        scoped_tools = context.allowed_tool_names
        full_name = f"{request.service_id}.{request.tool_name}"
        if scoped_tools and full_name not in scoped_tools:
            message = f"tool not allowed for selected agent: {full_name}"
            context.trace.add_event("tool.agent_scope_denied", message, allowed_tools=sorted(scoped_tools))
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code="agent_tool_scope_denied",
                error_message=message,
            )
        context.trace.add_event(
            "tool.preflight",
            f"{request.service_id}.{request.tool_name}",
            risk_level=request.risk_level,
            requires_approval=request.requires_approval,
        )
        hook_decision, hook_events = self.hook_runner.run_pre(context, request)
        for event in hook_events:
            context.trace.add_event(f"tool.{event.stage}", event.message, tool_name=event.tool_name)
        decision = hook_decision if not hook_decision.allowed else self.permission_policy.check(request)
        if not decision.allowed:
            approval_request = None
            if request.requires_approval:
                approval_request = ApprovalRequest(
                    approval_id=f"approval_{uuid4().hex}",
                    trace_id=context.trace.trace_id,
                    session_id=context.request.session_id,
                    approval_type="tool_call",
                    title="Tool call requires approval",
                    reason=decision.reason,
                    risk_level=request.risk_level,
                    requested_action=f"{request.service_id}.{request.tool_name}",
                    payload={"arguments": request.arguments},
                )
            context.trace.tool_calls.append(
                {
                    "service_id": request.service_id,
                    "tool_name": request.tool_name,
                    "ok": False,
                    "blocked": True,
                    "reason": decision.reason,
                    "approval_id": approval_request.approval_id if approval_request else None,
                }
            )
            context.trace.add_event(
                "tool.permission_denied",
                decision.reason,
                blocked_action=decision.blocked_action,
                approval_id=approval_request.approval_id if approval_request else None,
            )
            for event in self.hook_runner.run_failure(context, request, decision.reason):
                context.trace.add_event(f"tool.{event.stage}", event.message, tool_name=event.tool_name)
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                data={"approval_request": approval_request.model_dump()} if approval_request else {},
                error_code="approval_required" if approval_request else "permission_denied",
                error_message=decision.reason,
            )

        tool = self.registry.get(request.service_id, request.tool_name)
        if tool is None:
            message = "tool is not registered"
            context.trace.add_event("tool.not_found", message, service_id=request.service_id, tool_name=request.tool_name)
            for event in self.hook_runner.run_failure(context, request, message):
                context.trace.add_event(f"tool.{event.stage}", event.message, tool_name=event.tool_name)
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code="tool_not_found",
                error_message=message,
            )

        try:
            result = await tool.call(request)
        except Exception as exc:  # noqa: BLE001
            failure_reason = str(exc)
            for event in self.hook_runner.run_failure(context, request, failure_reason):
                context.trace.add_event(f"tool.{event.stage}", event.message, tool_name=event.tool_name)
            result = ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

        context.trace.tool_calls.append(
            {
                "service_id": result.service_id,
                "tool_name": result.tool_name,
                "ok": result.ok,
                "error_code": result.error_code,
                "provider": result.raw.get("provider"),
                "source_service": result.raw.get("source_service"),
            }
        )
        context.trace.add_event(
            "tool.result",
            f"{result.service_id}.{result.tool_name} returned {'ok' if result.ok else 'error'}",
            ok=result.ok,
            error_code=result.error_code,
        )
        for event in self.hook_runner.run_post(context, request, result):
            context.trace.add_event(f"tool.{event.stage}", event.message, tool_name=event.tool_name)
        return result


class EchoTool:
    """A harmless tool used to verify ToolRuntime behavior."""

    spec = ToolSpec(
        service_id="mock",
        tool_name="echo",
        description="Returns the provided arguments for runtime verification.",
        risk_level="low",
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={"echo": request.arguments},
            raw={"provider": "mock"},
        )


def default_tools() -> list[Tool]:
    """Default runtime tools.

    MCP tools are registered as specs, but they are only called when a Skill
    explicitly returns a matching ToolCallRequest.
    """

    from nikon0.tools.case_intake import ExtractCaseSlotsTool
    from nikon0.tools.memory import ReadSessionMemoryTool, WriteSessionFactTool
    from nikon0.tools.product import ResolveProductTool, SearchProductManualTool, ValidateAnswerGroundingTool
    from nikon0.mcp.provider import McpCapabilityProvider, McpToolPolicy

    tools: list[Tool] = [
        EchoTool(),
        ResolveProductTool(),
        SearchProductManualTool(),
        ValidateAnswerGroundingTool(),
        ExtractCaseSlotsTool(),
    ]
    try:
        from app.services.mcp_gateway.client import McpGatewayClient

        tools.extend(
            McpCapabilityProvider(
                McpGatewayClient(),
                policies=[
                    McpToolPolicy(service_id="case-intake", tool_name="get_case_intake_status", risk_level="low"),
                    McpToolPolicy(service_id="case-intake", tool_name="collect_case_intake", risk_level="medium"),
                    McpToolPolicy(service_id="case-intake", tool_name="try_cancel_case_intake", risk_level="medium"),
                ],
                allowed_tools={
                    "case-intake.get_case_intake_status",
                    "case-intake.collect_case_intake",
                    "case-intake.try_cancel_case_intake",
                },
            ).discover_tools()
        )
    except Exception:
        pass

    if not any(tool.spec.service_id == "case-intake" and tool.spec.tool_name == "collect_case_intake" for tool in tools):
        tools.extend(
            [
                McpGatewayTool(
                    service_id="case-intake",
                    tool_name="get_case_intake_status",
                    description="Query case-intake pending state through MCP Gateway.",
                    risk_level="low",
                ),
                McpGatewayTool(
                    service_id="case-intake",
                    tool_name="collect_case_intake",
                    description="Collect case-intake fields through MCP Gateway.",
                    risk_level="medium",
                ),
                McpGatewayTool(
                    service_id="case-intake",
                    tool_name="try_cancel_case_intake",
                    description="Cancel case-intake collection through MCP Gateway.",
                    risk_level="medium",
                ),
            ]
        )
    tools.extend(
        [
            ReadSessionMemoryTool(),
            WriteSessionFactTool(),
        ]
    )
    return tools
