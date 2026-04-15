"""Schema definitions for the `/chat` API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming request body for the competition-compatible endpoint."""

    question: str = Field(..., description="User question string, non-empty.")
    images: list[str] = Field(default_factory=list, description="Optional base64 image list (0-3).")
    session_id: Optional[str] = Field(default=None, description="Conversation session id.")
    stream: bool = Field(default=False, description="Whether to stream response.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("`question` must be a non-empty string.")
        return value.strip()

    @field_validator("images")
    @classmethod
    def validate_images(cls, value: list[str]) -> list[str]:
        if len(value) > 3:
            raise ValueError("`images` supports up to 3 items.")
        return value


class ChatResponseData(BaseModel):
    """Business payload for successful responses."""

    answer: str
    session_id: str
    timestamp: int


class ChatResponse(BaseModel):
    """Standardized API response envelope."""

    code: int
    msg: str
    data: ChatResponseData

    @staticmethod
    def success(answer: str, session_id: str) -> "ChatResponse":
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        return ChatResponse(
            code=0,
            msg="success",
            data=ChatResponseData(answer=answer, session_id=session_id, timestamp=now_ts),
        )
