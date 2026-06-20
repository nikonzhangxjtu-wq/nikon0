from __future__ import annotations

import json

from app.services.mcp_gateway.client import McpGatewayClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_call_tool_parses_gateway_text_json_payload() -> None:
    calls: list[dict] = []

    def fake_post(url: str, **kwargs):
        _ = url
        calls.append(kwargs["json"])
        downstream = json.dumps({"completed": True, "ticket_payload": {"status": "ready"}}, ensure_ascii=False)
        return FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": "x",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            # 当前 gateway 会把下游 MCP 的 text 再作为 JSON 字符串返回。
                            "text": json.dumps(downstream, ensure_ascii=False),
                        }
                    ]
                },
            }
        )

    client = McpGatewayClient(endpoint="http://gateway.test/mcp", post=fake_post)

    result = client.call_tool(
        service_id="case-intake",
        tool_name="collect_case_intake",
        arguments={"question": "报修"},
    )

    assert result == {"completed": True, "ticket_payload": {"status": "ready"}}
    sent_args = calls[0]["params"]["arguments"]
    assert sent_args["service_id"] == "case-intake"
    assert sent_args["tool_name"] == "collect_case_intake"
    assert sent_args["question"] == "报修"


def test_call_tool_parses_gateway_quoted_text_with_literal_newlines() -> None:
    def fake_post(url: str, **kwargs):
        _ = (url, kwargs)
        return FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": "x",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": '"{\n  \\"completed\\": true,\n  \\"reply_text\\": \\"ok\\"\n}"',
                        }
                    ]
                },
            }
        )

    client = McpGatewayClient(endpoint="http://gateway.test/mcp", post=fake_post)

    result = client.call_tool(
        service_id="case-intake",
        tool_name="collect_case_intake",
        arguments={"question": "报修"},
    )

    assert result == {"completed": True, "reply_text": "ok"}
