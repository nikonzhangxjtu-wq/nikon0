"""极简内存会话辅助（V1）。

刻意保持简单；生产环境建议换 Redis 等持久化存储。
"""

from __future__ import annotations

import uuid


def ensure_session_id(session_id: str | None) -> str:
    """若已有有效 session_id 则原样返回，否则生成新的。"""
    if session_id and session_id.strip():
        return session_id.strip()
    return f"kf_session_{uuid.uuid4().hex[:12]}"
