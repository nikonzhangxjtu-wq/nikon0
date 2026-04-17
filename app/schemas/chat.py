"""`/chat` 接口相关的 Pydantic 模型定义。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """与赛题一致的请求体。"""

    question: str = Field(..., description="用户问题，非空字符串。")
    images: list[str] = Field(default_factory=list, description="可选 Base64 图片列表，0～3 张。")
    session_id: Optional[str] = Field(default=None, description="会话 ID，多轮对话用。")
    stream: bool = Field(default=False, description="是否流式返回。")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("`question` 必须为非空字符串。")
        return value.strip()

    @field_validator("images")
    @classmethod
    def validate_images(cls, value: list[str]) -> list[str]:
        if len(value) > 3:
            raise ValueError("`images` 最多 3 张。")
        return value


class ChatResponseData(BaseModel):
    """成功时返回的业务数据。"""

    answer: str
    session_id: str
    timestamp: int


class ChatResponse(BaseModel):
    """统一响应外壳。"""

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
