"""Tests for nikon0 MCP provider normalization."""

from __future__ import annotations

import asyncio

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import ToolCallRequest
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.mcp.provider import McpCapabilityProvider, McpToolPolicy
from nikon0.tools.runtime import ToolRegistry, ToolRuntime


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def search_services(self, query: str = "") -> list[dict]:
        assert query == ""
        return [{"service_id": "orders", "name": "Order MCP"}]

    def list_tools(self, service_id: str) -> list[dict]:
        assert service_id == "orders"
        return [
            {
                "name": "lookup_order",
                "description": "Lookup order status.",
                "input_schema": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                "risk_level": "high",
            }
        ]

    def call_tool(self, *, service_id: str, tool_name: str, arguments: dict) -> dict:
        self.calls.append((service_id, tool_name, arguments))
        return {"status": "paid", "order_id": arguments["order_id"]}


def _context() -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="mcp-provider-s1", message="查订单"),
        trace=ExecutionTrace(
            trace_id="trace-mcp-provider",
            session_id="mcp-provider-s1",
            user_message="查订单",
        ),
    )


def test_mcp_provider_discovers_tools_and_applies_local_policy() -> None:
    provider = McpCapabilityProvider(
        FakeMcpClient(),
        policies=[
            McpToolPolicy(
                service_id="orders",
                tool_name="lookup_order",
                risk_level="medium",
                requires_approval=False,
                capability_tags=["order", "read"],
            )
        ],
    )

    tools = provider.discover_tools()

    assert len(tools) == 1
    spec = tools[0].spec
    assert spec.service_id == "orders"
    assert spec.tool_name == "lookup_order"
    assert spec.risk_level == "medium"
    assert spec.input_schema["x-provider"] == "mcp"
    assert spec.input_schema["x-source-service"] == "orders"
    assert spec.input_schema["x-capability-tags"] == ["order", "read"]


def test_mcp_tool_adapter_calls_through_tool_runtime_with_trace_metadata() -> None:
    client = FakeMcpClient()
    provider = McpCapabilityProvider(
        client,
        policies=[
            McpToolPolicy(service_id="orders", tool_name="lookup_order", risk_level="low"),
        ],
    )
    runtime = ToolRuntime(registry=ToolRegistry(provider.discover_tools()))
    context = _context()

    result = asyncio.run(
        runtime.call(
            context,
            ToolCallRequest(
                service_id="orders",
                tool_name="lookup_order",
                arguments={"order_id": "A1001"},
            ),
        )
    )

    assert result.ok is True
    assert result.data == {"status": "paid", "order_id": "A1001"}
    assert result.raw["provider"] == "mcp"
    assert result.raw["source_service"] == "orders"
    assert client.calls == [("orders", "lookup_order", {"order_id": "A1001"})]
    assert context.trace.tool_calls[-1]["provider"] == "mcp"
    assert context.trace.tool_calls[-1]["source_service"] == "orders"


def test_mcp_provider_can_filter_allowed_tools() -> None:
    provider = McpCapabilityProvider(
        FakeMcpClient(),
        allowed_tools={"orders.lookup_order"},
    )

    assert [tool.spec.tool_name for tool in provider.discover_tools()] == ["lookup_order"]

    blocked_provider = McpCapabilityProvider(
        FakeMcpClient(),
        allowed_tools={"orders.cancel_order"},
    )

    assert blocked_provider.discover_tools() == []
