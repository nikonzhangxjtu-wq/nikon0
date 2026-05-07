"""MCP 订单状态 Provider：通过 HTTP JSON-RPC 调用 MCP Server 查询订单进度。"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.core.config import settings
from app.services.order_status_skill import OrderStatusHit, OrderStatusProvider


def _to_hit(item: dict[str, Any]) -> OrderStatusHit | None:
    if not isinstance(item, dict):
        return None
    order_id = str(item.get("order_id", "")).strip()
    status = str(item.get("status", "")).strip()
    if not order_id and not status:
        return None
    return OrderStatusHit(
        order_id=order_id,
        status=status,
        logistics_status=str(item.get("logistics_status", "")).strip(),
        eta=str(item.get("eta", "")).strip(),
        updated_at=str(item.get("updated_at", "")).strip(),
        can_refund=str(item.get("can_refund", "")).strip(),
        note=str(item.get("note", "")).strip(),
    )


class MCPOrderProvider(OrderStatusProvider):
    """通过 HTTP JSON-RPC 调 MCP Server tool 查询订单状态。"""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        api_key: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        ep = (endpoint or settings.mcp_order_endpoint).strip()
        if not ep:
            raise ValueError("MCP order endpoint 不能为空")
        self._endpoint = ep
        self._tool_name = (settings.mcp_order_tool_name or "get_order_status").strip() or "get_order_status"
        self._api_key = (api_key or settings.mcp_order_api_key).strip()
        self._timeout_sec = timeout_sec or settings.mcp_order_timeout_sec

    def search_order_status(self, query: str, *, top_k: int = 3) -> list[OrderStatusHit]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": self._tool_name,
                "arguments": {"query": query, "top_k": top_k},
            },
        }

        resp = requests.post(
            self._endpoint,
            json=payload,
            headers=headers,
            timeout=self._timeout_sec,
        )
        resp.raise_for_status()
        body = resp.json()

        # MCP JSON-RPC 响应：result.structuredContent 或 result.content[0].text
        result = body.get("result", {})
        if result.get("isError", False):
            raise RuntimeError(f"MCP tool 调用失败: {self._tool_name}")

        rows = _rows_from_result(result)
        out: list[OrderStatusHit] = []
        for row in rows:
            hit = _to_hit(row)
            if hit is not None:
                out.append(hit)
        return out


def _rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        hits = structured.get("hits")
        if isinstance(hits, list):
            return [x for x in hits if isinstance(x, dict)]

    content = result.get("content")
    if isinstance(content, list):
        for c in content:
            text = c.get("text") if isinstance(c, dict) else ""
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        hits = parsed.get("hits")
                        if isinstance(hits, list):
                            return [x for x in hits if isinstance(x, dict)]
                except json.JSONDecodeError:
                    continue
    return []
