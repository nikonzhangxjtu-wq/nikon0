# MCP Order Status Demo

这个目录提供一个真正的 MCP Server（`streamable-http`），供主业务里的 `MCPOrderProvider` 调用。

## 运行

```bash
cd /Users/nikonzhang/compeletion
python3.10 -m venv .venv-mcp-order
source .venv-mcp-order/bin/activate
pip install -r examples/mcp_order_status/requirements.txt
python examples/mcp_order_status/server.py
```

默认 endpoint：

```text
http://127.0.0.1:8010/mcp
```

可用环境变量（本示例读取 `MCP_ORDER_HOST`/`MCP_ORDER_PORT` 或 `FASTMCP_HOST`/`FASTMCP_PORT`，并**显式传入** `FastMCP(host=, port=)`。原因：官方 SDK 构造 `Settings` 时带了默认 `port=8000`，会压住仅靠环境变量改端口的用法；未设置端口时本示例默认 **8010**。`run()` 不支持 `host=`/`port=`。）：

```bash
export MCP_ORDER_HOST=127.0.0.1
export MCP_ORDER_PORT=8010
python examples/mcp_order_status/server.py
```

或直接：

```bash
export FASTMCP_HOST=127.0.0.1
export FASTMCP_PORT=8010
python examples/mcp_order_status/server.py
```

## 主业务 .env

```bash
ORDER_STATUS_SKILL_ENABLED=true
MCP_ORDER_ENDPOINT=http://127.0.0.1:8010/mcp
MCP_ORDER_TOOL_NAME=get_order_status
```

## 暴露的 MCP tools

- `get_order_status(query: str, top_k: int = 3)`
- `list_recent_orders(phone_last4: str, top_k: int = 5)`
- `suggest_after_sales_action(order_id: str, user_request: str = "")`

主业务当前默认调用：`get_order_status`。

## 测试问题

```text
请帮我查订单 OD20260507001 到哪了
```

## 说明

当前订单数据在 `server.py` 的 `_ORDERS` 内存字典里，真实业务接入时可替换为数据库/API 查询。
