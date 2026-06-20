from __future__ import annotations

from app.services.skills.gateway_case_intake import GatewayCaseIntakeSkill


class FakeGatewayClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def call_tool(self, *, service_id: str, tool_name: str, arguments: dict) -> dict:
        self.calls.append((service_id, tool_name, arguments))
        if tool_name == "collect_case_intake":
            return {
                "completed": True,
                "exited": False,
                "reply_text": "已为你完成售后受理信息收集。",
                "missing_slots": [],
                "ticket_payload": {"status": "ready", "product_model": "DW-123"},
                "context_block": "[工单收集状态]\nstatus: ready",
                "react_trace": [],
            }
        if tool_name == "get_case_intake_status":
            return {"has_pending": True}
        if tool_name == "try_cancel_case_intake":
            return {"cancelled": True}
        raise AssertionError(f"unexpected tool {tool_name}")


def test_gateway_case_intake_skill_run_maps_tool_result() -> None:
    client = FakeGatewayClient()
    skill = GatewayCaseIntakeSkill(client=client)

    result = skill.run(
        question="型号 DW-123 无法启动，手机号 13800138000",
        session_id="sid1",
        conversation_history="上一轮",
        enrichment="记忆",
    )

    assert result.completed is True
    assert result.ticket_payload["status"] == "ready"
    assert result.context_block.startswith("[工单收集状态]")
    service_id, tool_name, args = client.calls[0]
    assert service_id == "case-intake"
    assert tool_name == "collect_case_intake"
    assert args["session_id"] == "sid1"
    assert args["conversation_history"] == "上一轮"


def test_gateway_case_intake_skill_status_and_cancel() -> None:
    client = FakeGatewayClient()
    skill = GatewayCaseIntakeSkill(client=client)

    assert skill.has_pending_intake("sid1") is True
    assert skill.try_cancel_intake("sid1", "算了，取消工单") is True

