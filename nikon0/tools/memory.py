"""Memory tools with explicit, auditable state patch payloads."""

from __future__ import annotations

from typing import Any

from nikon0.app.schemas.capability import ToolCallRequest, ToolCallResult, ToolSpec


class ReadSessionMemoryTool:
    spec = ToolSpec(
        service_id="memory",
        tool_name="read_session_memory",
        description="Return the provided session memory snapshot for skill-local reasoning.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {"session_state": {"type": "object"}},
        },
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        session_state = request.arguments.get("session_state")
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={"session_state": session_state if isinstance(session_state, dict) else {}},
        )


class WriteSessionFactTool:
    spec = ToolSpec(
        service_id="memory",
        tool_name="write_session_fact",
        description="Prepare a normalized session-memory fact patch for Runtime to apply.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
                "reason": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key", "value"],
        },
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        key = str(request.arguments.get("key") or "").strip()
        if not key:
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code="invalid_arguments",
                error_message="key is required",
            )
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={
                "state_update": {
                    "key": key,
                    "value": request.arguments.get("value"),
                    "reason": str(request.arguments.get("reason") or "written by memory tool"),
                    "evidence_ids": _string_list(request.arguments.get("evidence_ids")),
                }
            },
        )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
