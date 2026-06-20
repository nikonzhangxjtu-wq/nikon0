"""售后工单 MCP Server（streamable-http）。

这个服务把项目内已有的 ``CaseIntakeSkill`` 包装成 MCP tool，供
``mcp-gateway`` 统一发现和转发。这里不复制工单规则逻辑，而是直接复用
业务 Skill，避免后续客服主流程和 MCP 工具两边出现两套不一致的槽位规则。

运行：
    cd /Users/nikonzhang/compeletion
    conda run -n kefu python examples/mcp_case_intake/server.py

默认 endpoint：
    http://127.0.0.1:8011/mcp
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 工具服务第一版追求稳定和低延迟，默认关闭工单 ReAct 子流程；
# 如需实验多步推理，可在启动前显式设置 CASE_INTAKE_REACT_ENABLED=true。
os.environ.setdefault("CASE_INTAKE_REACT_ENABLED", "false")

from app.services.skills.case_intake_skill import CaseIntakeSkill  # noqa: E402


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("mcp_case_intake")


def _listen_host_port() -> tuple[str, int]:
    """读取监听地址；FastMCP 需要显式传 host/port，不能只依赖环境变量。"""
    host = os.getenv("MCP_CASE_INTAKE_HOST") or os.getenv("FASTMCP_HOST") or "127.0.0.1"
    raw_port = os.getenv("MCP_CASE_INTAKE_PORT") or os.getenv("FASTMCP_PORT")
    return host, int(raw_port) if raw_port else 8011


_host, _port = _listen_host_port()

mcp = FastMCP(
    "CaseIntakeService",
    stateless_http=True,
    json_response=True,
    host=_host,
    port=_port,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[
            "127.0.0.1",
            "127.0.0.1:*",
            "localhost",
            "localhost:*",
            "host.docker.internal",
            "host.docker.internal:*",
        ],
    ),
)


def _skill() -> CaseIntakeSkill:
    return CaseIntakeSkill()


@mcp.tool()
def collect_case_intake(
    question: str,
    session_id: str = "",
    conversation_history: str = "",
    enrichment: str = "",
) -> dict[str, Any]:
    """收集售后/报修/退款工单信息。

    参数：
    - question: 用户当前发言。
    - session_id: 会话 ID；相同 session 会延续未完成工单草稿。
    - conversation_history: 可选历史对话文本，用于补全槽位。
    - enrichment: 可选结构化记忆或外部补充信息。
    """
    log.info("collect_case_intake called session_id=%s", session_id or "__default__")
    result = _skill().run(
        question=question,
        session_id=session_id or "__mcp_default__",
        conversation_history=conversation_history,
        enrichment=enrichment,
    )
    # MCP 返回 JSON 友好的结构化结果；最终是否写入客服回答由上层 agent/pipeline 决定。
    return {
        "completed": result.completed,
        "exited": result.exited,
        "reply_text": result.reply_text,
        "missing_slots": list(result.missing_slots),
        "ticket_payload": dict(result.ticket_payload),
        "context_block": result.context_block,
        "react_trace": list(result.react_trace),
    }


@mcp.tool()
def get_case_intake_status(session_id: str) -> dict[str, Any]:
    """查询指定 session 是否存在未完成工单草稿。"""
    sid = session_id or "__mcp_default__"
    pending = _skill().has_pending_intake(sid)
    return {"session_id": sid, "has_pending": pending}


@mcp.tool()
def try_cancel_case_intake(session_id: str, question: str) -> dict[str, Any]:
    """仅当 question 明确表达取消时，取消未完成工单草稿。"""
    sid = session_id or "__mcp_default__"
    cancelled = _skill().try_cancel_intake(sid, question or "")
    return {
        "cancelled": cancelled,
        "session_id": sid,
        "reply_text": "已取消当前工单草稿。" if cancelled else "当前未识别到取消工单意图。",
    }


@mcp.tool()
def cancel_case_intake(session_id: str) -> dict[str, Any]:
    """取消指定 session 的未完成工单草稿。"""
    sid = session_id or "__mcp_default__"
    skill = _skill()
    cancelled = skill.try_cancel_intake(sid, "取消工单")
    return {
        "cancelled": cancelled,
        "session_id": sid,
        "reply_text": "已取消当前工单草稿。" if cancelled else "当前没有可取消的工单草稿。",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
