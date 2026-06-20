"""Trace and transcript persistence schemas."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from nikon0.app.schemas.trace import ExecutionTrace


TranscriptRole = Literal["user", "assistant", "system", "tool"]


class TranscriptEntry(BaseModel):
    session_id: str
    trace_id: str
    role: TranscriptRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class StoredTrace(BaseModel):
    trace_id: str
    session_id: str
    trace: ExecutionTrace
    created_at: float = Field(default_factory=time.time)
