"""Normalize MCP capabilities into nikon0 ToolRuntime tools."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import RiskLevel, ToolCallRequest, ToolCallResult, ToolSpec


class McpClientProtocol(Protocol):
    def search_services(self, query: str = "") -> list[dict[str, Any]]:
        ...

    def list_tools(self, service_id: str) -> list[dict[str, Any]]:
        ...

    def call_tool(self, *, service_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class McpToolPolicy(BaseModel):
    service_id: str
    tool_name: str
    risk_level: RiskLevel | None = None
    requires_approval: bool | None = None
    capability_tags: list[str] = Field(default_factory=list)
    timeout_ms: int | None = None

    @property
    def key(self) -> str:
        return f"{self.service_id}.{self.tool_name}"


class McpToolAdapter:
    """A ToolRuntime-compatible adapter over an MCP tool."""

    def __init__(
        self,
        *,
        client: McpClientProtocol,
        service_id: str,
        tool_name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        risk_level: RiskLevel = "low",
        requires_approval: bool = False,
        capability_tags: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.client = client
        self.requires_approval = requires_approval
        self.timeout_ms = timeout_ms
        schema = dict(input_schema or {})
        schema["x-provider"] = "mcp"
        schema["x-source-service"] = service_id
        schema["x-capability-tags"] = list(capability_tags or [])
        if timeout_ms is not None:
            schema["x-timeout-ms"] = timeout_ms
        if requires_approval:
            schema["x-requires-approval"] = True
        self.spec = ToolSpec(
            service_id=service_id,
            tool_name=tool_name,
            description=description,
            risk_level=risk_level,
            input_schema=schema,
        )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        try:
            payload = self.client.call_tool(
                service_id=request.service_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code=type(exc).__name__,
                error_message=str(exc),
                raw={"provider": "mcp", "source_service": request.service_id},
            )
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data=payload if isinstance(payload, dict) else {"value": payload},
            raw={"provider": "mcp", "source_service": request.service_id},
        )


class McpCapabilityProvider:
    """Discovers MCP tools and exposes them as ToolRuntime tools."""

    def __init__(
        self,
        client: McpClientProtocol,
        *,
        policies: list[McpToolPolicy] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.client = client
        self.policies = {policy.key: policy for policy in policies or []}
        self.allowed_tools = set(allowed_tools or [])

    def discover_tools(self) -> list[McpToolAdapter]:
        tools: list[McpToolAdapter] = []
        for service in self.client.search_services(""):
            service_id = _service_id(service)
            if not service_id:
                continue
            for raw_tool in self.client.list_tools(service_id):
                tool_name = _tool_name(raw_tool)
                if not tool_name:
                    continue
                key = f"{service_id}.{tool_name}"
                if self.allowed_tools and key not in self.allowed_tools:
                    continue
                policy = self.policies.get(key)
                tools.append(
                    McpToolAdapter(
                        client=self.client,
                        service_id=service_id,
                        tool_name=tool_name,
                        description=str(raw_tool.get("description") or ""),
                        input_schema=_input_schema(raw_tool),
                        risk_level=_risk_level(raw_tool, policy),
                        requires_approval=bool(policy.requires_approval) if policy and policy.requires_approval is not None else False,
                        capability_tags=list(policy.capability_tags) if policy else [],
                        timeout_ms=policy.timeout_ms if policy else None,
                    )
                )
        return tools


def _service_id(service: dict[str, Any]) -> str:
    return str(service.get("service_id") or service.get("id") or "").strip()


def _tool_name(raw_tool: dict[str, Any]) -> str:
    return str(raw_tool.get("tool_name") or raw_tool.get("name") or "").strip()


def _input_schema(raw_tool: dict[str, Any]) -> dict[str, Any]:
    schema = raw_tool.get("input_schema") or raw_tool.get("inputSchema") or raw_tool.get("schema")
    return dict(schema) if isinstance(schema, dict) else {}


def _risk_level(raw_tool: dict[str, Any], policy: McpToolPolicy | None) -> RiskLevel:
    if policy and policy.risk_level:
        return policy.risk_level
    value = str(raw_tool.get("risk_level") or raw_tool.get("riskLevel") or "low")
    return value if value in {"low", "medium", "high"} else "low"
