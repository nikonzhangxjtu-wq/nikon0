"""通过 MCP Gateway 调用售后工单服务的 Skill 适配器。"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.mcp_gateway.client import McpGatewayClient, McpGatewayError
from app.services.skills.case_intake_types import CaseIntakeResult


class GatewayCaseIntakeSkill:
    """保持 ``CaseIntakeSkill`` 的接口形状，内部改走 MCP Gateway。

    Pipeline 只需要 ``run``、``has_pending_intake``、``try_cancel_intake``，
    不应关心下游到底是本地规则 Skill，还是 gateway 后面的 MCP 服务。
    """

    def __init__(
        self,
        *,
        client: McpGatewayClient | None = None,
        service_id: str | None = None,
        collect_tool: str | None = None,
        status_tool: str | None = None,
        cancel_tool: str | None = None,
    ) -> None:
        self._client = client or McpGatewayClient()
        self._service_id = service_id or settings.mcp_case_intake_service_id
        self._collect_tool = collect_tool or settings.mcp_case_intake_collect_tool
        self._status_tool = status_tool or settings.mcp_case_intake_status_tool
        self._cancel_tool = cancel_tool or settings.mcp_case_intake_cancel_tool

    def has_pending_intake(self, session_id: str) -> bool:
        if not (session_id or "").strip():
            return False
        try:
            payload = self._call(
                self._status_tool,
                {"session_id": session_id},
            )
        except McpGatewayError as exc:
            print(f"[WARN] Gateway case intake status failed: {exc}")
            return False
        return bool(payload.get("has_pending"))

    def try_cancel_intake(self, session_id: str, question: str) -> bool:
        if not (session_id or "").strip():
            return False
        try:
            payload = self._call(
                self._cancel_tool,
                {"session_id": session_id, "question": question},
            )
        except McpGatewayError as exc:
            print(f"[WARN] Gateway case intake cancel failed: {exc}")
            return False
        return bool(payload.get("cancelled"))

    def run(
        self,
        *,
        question: str,
        session_id: str,
        conversation_history: str = "",
        enrichment: str = "",
    ) -> CaseIntakeResult:
        try:
            payload = self._call(
                self._collect_tool,
                {
                    "question": question,
                    "session_id": session_id,
                    "conversation_history": conversation_history,
                    "enrichment": enrichment,
                },
            )
        except McpGatewayError as exc:
            return CaseIntakeResult(
                completed=False,
                reply_text="当前工单服务暂时不可用，请稍后再试或转人工客服处理。",
                missing_slots=[],
                ticket_payload={},
                context_block=f"[工单收集状态]\nstatus: gateway_error\nreason: {exc}",
            )
        return self._result_from_payload(payload)

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._client.call_tool(
            service_id=self._service_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    @staticmethod
    def _result_from_payload(payload: dict[str, Any]) -> CaseIntakeResult:
        ticket_raw = payload.get("ticket_payload")
        ticket_payload = {
            str(k): str(v)
            for k, v in ticket_raw.items()
        } if isinstance(ticket_raw, dict) else {}
        return CaseIntakeResult(
            completed=bool(payload.get("completed")),
            exited=bool(payload.get("exited")),
            reply_text=str(payload.get("reply_text") or ""),
            missing_slots=[
                str(item)
                for item in payload.get("missing_slots", [])
                if isinstance(item, str)
            ],
            ticket_payload=ticket_payload,
            context_block=str(payload.get("context_block") or ""),
            react_trace=tuple(
                str(item)
                for item in payload.get("react_trace", [])
                if isinstance(item, str)
            ),
        )

