"""Very small in-memory session helper for V1.

This is intentionally simple. For production you should move to Redis
or another persistent store.
"""

from __future__ import annotations

import uuid


def ensure_session_id(session_id: str | None) -> str:
    """Return existing session id or generate a new one."""
    if session_id and session_id.strip():
        return session_id.strip()
    return f"kf_session_{uuid.uuid4().hex[:12]}"
