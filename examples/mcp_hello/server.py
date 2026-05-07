"""最小 MCP 服务（stdio）：理解「Host 连子进程、JSON-RPC、Tools / Prompts」。

MCP 是什么（一句话）
    Host（如 Cursor）启动本脚本当子进程，通过 **stdin 写 / stdout 读** JSON-RPC
    消息；你在这里用 FastMCP 注册 **tools**（模型可调用的函数）、**prompts**
    （可复用的提示模板）等。

本地运行
    cd 仓库根目录
    python3.10 -m venv .venv && source .venv/bin/activate   # 或 conda 3.10+
    pip install -r examples/mcp_hello/requirements.txt
    python examples/mcp_hello/server.py

在 Cursor 里挂上这个 MCP（User Settings → MCP，或项目 `.cursor/mcp.json`）
    把下面路径换成你本机绝对路径；command 用你装好 ``mcp`` 的 Python。

    {
      "mcpServers": {
        "hello-demo": {
          "command": "/ABS/PATH/TO/venv/bin/python",
          "args": ["/ABS/PATH/TO/compeletion/examples/mcp_hello/server.py"]
        }
      }
    }

调试
    npx -y @modelcontextprotocol/inspector
    在 Inspector 里选 stdio，command 填上面的 python，args 填 server.py。

注意：不要在服务器里向 stdout 打 ``print`` 调试（会破坏 JSON-RPC）；用 logging 到 stderr。
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("mcp_hello")

mcp = FastMCP("HelloDemo")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回 text，用于验证整条 MCP 调用链路。"""
    log.info("tool echo called")
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """返回 a + b。"""
    return a + b


@mcp.prompt()
def summarize_brief(topic: str) -> str:
    """给模型用的提示模板：请简短总结某个主题。"""
    return f"请用不超过三句话、中文，总结这个主题：{topic}"


if __name__ == "__main__":
    mcp.run()
