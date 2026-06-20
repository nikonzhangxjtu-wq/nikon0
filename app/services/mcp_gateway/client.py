"""MCP Gateway JSON-RPC client.

本客户端只负责和统一 gateway 通信，不包含任何客服业务判断。
业务语义应放在具体 Skill/Provider 适配层里。
"""

from __future__ import annotations

import json
from typing import Any, Callable

import requests

from app.core.config import settings


class McpGatewayError(RuntimeError):
    """Gateway 调用失败。"""


class McpGatewayClient:
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        bearer_token: str | None = None,
        timeout_sec: int | None = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoint = (endpoint or settings.mcp_gateway_endpoint).strip()
        self.bearer_token = (bearer_token if bearer_token is not None else settings.mcp_gateway_bearer_token).strip()
        self.timeout_sec = int(timeout_sec or settings.mcp_gateway_timeout_sec)
        self._post = post or requests.post

    def call_tool(self, *, service_id: str, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        args["service_id"] = service_id
        args["tool_name"] = tool_name
        return self.call_catalog_tool("call_mcp_tool", args)

    def search_services(self, query: str = "") -> list[dict[str, Any]]:
        payload = self.call_catalog_tool("search_mcp_services", {"query": query})
        return self._list_payload(payload, keys=("services", "items", "results"))

    def list_tools(self, service_id: str) -> list[dict[str, Any]]:
        payload = self.call_catalog_tool("list_mcp_tools", {"service_id": service_id})
        return self._list_payload(payload, keys=("tools", "items", "results"))

    def call_catalog_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.endpoint:
            raise McpGatewayError("MCP gateway endpoint is empty")
        payload = {
            "jsonrpc": "2.0",
            "id": name,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        try:
            response = self._post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise McpGatewayError(f"gateway_request_failed:{exc}") from exc
        if not isinstance(body, dict):
            raise McpGatewayError("gateway_response_not_object")
        if body.get("error"):
            raise McpGatewayError(str(body["error"]))
        result = body.get("result")
        if not isinstance(result, dict):
            return {}
        return self._extract_result_payload(result)

    @classmethod
    def _extract_result_payload(cls, result: dict[str, Any]) -> dict[str, Any]:
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if not isinstance(content, list) or not content:
            return result
        first = content[0]
        if not isinstance(first, dict):
            return result
        text = first.get("text")
        if not isinstance(text, str):
            return result
        parsed = cls._parse_nested_json_text(text)
        return parsed if isinstance(parsed, dict) else {"text": text}

    @staticmethod
    def _parse_nested_json_text(text: str) -> Any:
        """解析 gateway 常见的双层 JSON 文本。

        case-intake 下游返回 JSON；gateway 的 ``extractText`` 会把它作为字符串再
        放进 content.text，因此这里最多解两层，得到真正的 dict。
        """
        value: Any = text
        for _ in range(2):
            if not isinstance(value, str):
                break
            stripped = value.strip()
            if not stripped:
                break
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                if stripped.startswith('"') and stripped.endswith('"'):
                    unwrapped = stripped[1:-1].replace('\\"', '"')
                    try:
                        value = json.loads(unwrapped)
                        continue
                    except json.JSONDecodeError:
                        pass
                break
        return McpGatewayClient._normalize_escaped_strings(value)

    @staticmethod
    def _list_payload(payload: dict[str, Any], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if all(key in payload for key in ("service_id", "id")):
            return [payload]
        return []

    @staticmethod
    def _normalize_escaped_strings(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("\\n", "\n")
        if isinstance(value, list):
            return [McpGatewayClient._normalize_escaped_strings(item) for item in value]
        if isinstance(value, dict):
            return {
                key: McpGatewayClient._normalize_escaped_strings(item)
                for key, item in value.items()
            }
        return value
