"""Chat API for nikon0."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from nikon0.agent.runtime import build_default_runtime
from nikon0.app.schemas.agent import AgentRequest, AgentResponse
from nikon0.app.schemas.safety import ApprovalRequest, ApprovalStatus, HandoffRequest

router = APIRouter(prefix="/api/v1", tags=["chat"])
runtime = build_default_runtime()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message.")
    session_id: str = Field(default="default", description="Session id for issue-state tracking.")
    user_id: str | None = None
    images: list[str] = Field(default_factory=list)
    channel: str = "web"
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message", "session_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value


@router.post("/chat", response_model=AgentResponse)
async def chat(req: ChatRequest) -> AgentResponse:
    return await runtime.run(
        AgentRequest(
            session_id=req.session_id,
            user_id=req.user_id,
            message=req.message,
            images=req.images,
            channel=req.channel,
            metadata=req.metadata,
        )
    )


@router.get("/approvals", response_model=list[ApprovalRequest])
async def list_approvals(session_id: str | None = None) -> list[ApprovalRequest]:
    return runtime.approval_store.list_approvals(session_id)


@router.post("/approvals/{approval_id}/{status}", response_model=ApprovalRequest | None)
async def update_approval(approval_id: str, status: ApprovalStatus) -> ApprovalRequest | None:
    return runtime.approval_store.update_approval(approval_id, status)


@router.get("/handoffs", response_model=list[HandoffRequest])
async def list_handoffs(session_id: str | None = None) -> list[HandoffRequest]:
    return runtime.approval_store.list_handoffs(session_id)
