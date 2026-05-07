"""订单状态 MCP Server（streamable-http）。

业务场景：给主业务 pipeline 的 MCPOrderProvider 提供标准 MCP tool：
`get_order_status(query, top_k)`。

运行：
    cd /Users/nikonzhang/compeletion
    python3.10 -m venv .venv-mcp-order && source .venv-mcp-order/bin/activate
    pip install -r examples/mcp_order_status/requirements.txt
    python examples/mcp_order_status/server.py

默认 endpoint（`MCP_ORDER_HOST`/`PORT` 或 `FASTMCP_HOST`/`PORT`；需传给 `FastMCP(host=, port=)`，仅设环境变量不够）：
    http://127.0.0.1:8010/mcp

主业务 .env 示例：
    MCP_ORDER_ENDPOINT=http://127.0.0.1:8010/mcp
    MCP_ORDER_TOOL_NAME=get_order_status
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("mcp_order_status")


def _listen_host_port() -> tuple[str, int]:
    """官方 mcp 的 FastMCP 会把 host/port 显式传入 Settings；环境变量不会覆盖这些 kwargs，必须自己读 env 再传给 FastMCP。"""
    host = os.getenv("MCP_ORDER_HOST") or os.getenv("FASTMCP_HOST") or "127.0.0.1"
    raw = os.getenv("MCP_ORDER_PORT") or os.getenv("FASTMCP_PORT")
    port = int(raw) if raw else 8010
    return host, port


_listen_host, _listen_port = _listen_host_port()

mcp = FastMCP(
    "OrderStatusDemo",
    stateless_http=True,
    json_response=True,
    host=_listen_host,
    port=_listen_port,
)

_ORDERS: dict[str, dict[str, str]] = {
    "OD20260507001": {
        "order_id": "OD20260507001",
        "status": "已发货",
        "logistics_status": "运输中（杭州分拨中心）",
        "eta": "2026-05-09",
        "updated_at": "2026-05-07 19:20",
        "can_refund": "可联系客服尝试拦截退款，是否成功以物流状态为准",
        "note": "建议等待下一物流节点；若 24 小时未更新可催单。",
        "phone_last4": "8000",
    },
    "OD20260507002": {
        "order_id": "OD20260507002",
        "status": "待发货",
        "logistics_status": "仓库已接单，待揽收",
        "eta": "2026-05-10",
        "updated_at": "2026-05-07 18:05",
        "can_refund": "通常可申请取消订单或退款，需以平台状态为准",
        "note": "若着急可先联系平台客服催发货。",
        "phone_last4": "8000",
    },
    "OD20260507003": {
        "order_id": "OD20260507003",
        "status": "已签收",
        "logistics_status": "本人签收",
        "eta": "已送达",
        "updated_at": "2026-05-06 14:12",
        "can_refund": "如商品存在问题，可按平台规则申请售后",
        "note": "如未收到包裹，请尽快联系快递或平台客服核实签收信息。",
        "phone_last4": "1234",
    },
}

_ORDER_RE = re.compile(r"OD\d{11,20}|\b\d{8,20}\b", re.IGNORECASE)


def _public_order(row: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in row.items() if k != "phone_last4"}


def _extract_order_ids(query: str) -> list[str]:
    ids: list[str] = []
    for m in _ORDER_RE.finditer(query or ""):
        oid = m.group(0).upper()
        if oid not in ids:
            ids.append(oid)
    return ids


@mcp.tool()
def get_order_status(query: str, top_k: int = 3) -> dict[str, list[dict[str, str]]]:
    """按用户查询文本查订单状态。参数：query=用户问题或订单号，top_k=返回上限。"""
    log.info("get_order_status called")
    order_ids = _extract_order_ids(query)
    hits: list[dict[str, str]] = []

    for oid in order_ids:
        row = _ORDERS.get(oid)
        if row is not None:
            hits.append(_public_order(row))

    # 如果没有明确订单号命中，但 query 中带手机号后四位，则返回该手机号近期订单。
    if not hits:
        for last4 in re.findall(r"\b\d{4}\b", query or ""):
            for row in _ORDERS.values():
                if row.get("phone_last4") == last4:
                    hits.append(_public_order(row))
                    if len(hits) >= top_k:
                        break
            if hits:
                break

    return {"hits": hits[: max(1, top_k)]}


@mcp.tool()
def list_recent_orders(phone_last4: str, top_k: int = 5) -> dict[str, list[dict[str, str]]]:
    """按手机号后四位列出近期订单，用于用户没提供订单号的场景。"""
    log.info("list_recent_orders called")
    key = (phone_last4 or "").strip()[-4:]
    hits = [_public_order(row) for row in _ORDERS.values() if row.get("phone_last4") == key]
    return {"hits": hits[: max(1, top_k)]}


@mcp.tool()
def suggest_after_sales_action(order_id: str, user_request: str = "") -> dict[str, Any]:
    """根据订单状态给售后建议。"""
    log.info("suggest_after_sales_action called")
    row = _ORDERS.get((order_id or "").strip().upper())
    if row is None:
        return {"ok": False, "suggestion": "未查到该订单，请核对订单号。"}

    status = row.get("status", "")
    if "待发货" in status:
        suggestion = "当前待发货，可优先申请取消订单；若仍需要商品，可联系平台催发货。"
    elif "已发货" in status:
        suggestion = "当前已发货，建议先关注物流节点；如需退款，可联系客服尝试拦截。"
    elif "签收" in status:
        suggestion = "当前已签收，如商品异常请准备照片/视频并发起售后。"
    else:
        suggestion = "建议联系平台客服核实订单状态后处理。"

    if user_request:
        suggestion += f" 用户诉求：{user_request.strip()[:80]}。"
    return {"ok": True, "order": _public_order(row), "suggestion": suggestion}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
