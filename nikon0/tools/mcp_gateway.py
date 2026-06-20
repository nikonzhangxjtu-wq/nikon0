"""MCP Gateway backed tools for nikon0."""

from __future__ import annotations

from typing import Any

from nikon0.app.schemas.capability import RiskLevel, ToolCallRequest, ToolCallResult, ToolSpec


class McpGatewayTool:
    """A generic nikon0 Tool backed by the existing MCP Gateway client."""

    def __init__(
        self,
        *,
        service_id: str,
        tool_name: str,
        description: str = "",
        risk_level: RiskLevel = "low",
        client: Any | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self.spec = ToolSpec(
            service_id=service_id,
            tool_name=tool_name,
            description=description,
            risk_level=risk_level,
            input_schema=input_schema or {},
        )
        if client is None:
            from app.services.mcp_gateway.client import McpGatewayClient

            client = McpGatewayClient()
        self._client = client

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        try:
            payload = self._client.call_tool(
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
            )
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data=payload if isinstance(payload, dict) else {"value": payload},
            raw={"provider": "mcp_gateway"},
        )
